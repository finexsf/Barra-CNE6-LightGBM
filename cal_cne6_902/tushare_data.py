import pandas as pd
import numpy as np
import tushare as ts
import os
import time
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')
pd.set_option('future.no_silent_downcasting', True)
ts.set_token(token='fcc268ebc60170db2757a9d5ca830fb4a5a5b36b51ee13bd629ba982')
pro = ts.pro_api()

from utils import save_df, read_df, gen_periods, name_exchange, name_exchange_anti, industry_d

def get_trade_date(start_date=None, end_date=None, count=None, all_dates=None):
    """
    获取一段时间内的交易日列表。
    start_date:开始日期: pandas._libs.tslibs.timestamps.Timestamp
    end_date:结束日期: pandas._libs.tslibs.timestamps.Timestamp
    count:数量，int
    如果三个参数都给出，默认使用start_date和end_date参数。
    """
    assert [start_date, end_date, count].count(None) <= 1
    trade_date = []
    if start_date != None and end_date == None:
        this_count = 0
        for d in all_dates:
            if d >= start_date:
                this_count += 1
                trade_date.append(d)
            if this_count == count:
                end_date = d
                break
    elif start_date == None and end_date != None:
        this_count = 0
        for d in reversed(all_dates):
            if d <= end_date:
                this_count += 1
                trade_date.append(d)
            if this_count == count:
                start_date = d
                break
    elif start_date != None and end_date != None:
        for d in all_dates:
            if start_date <= d <= end_date:
                trade_date.append(d)
    trade_date = list(set(trade_date))
    trade_date.sort()
    return trade_date

# 公布日对齐交易日
def pubDate_align_tradedate(df: pd.DataFrame, end_date, all_dates, pubDate_col='pubDate'):
    df.loc[:, pubDate_col] = pd.to_datetime(df[pubDate_col])
    # 获取交易日历
    trade_dates = sorted(get_trade_date(start_date=df[pubDate_col].min(), end_date=end_date, all_dates=all_dates))
    trade_dates_np = np.array(trade_dates)
    
    # 创建日期对齐函数
    def align_date_vectorized(dates):
        indices = np.searchsorted(trade_dates_np, dates, side='left')
        # 使用np.where但确保安全
        mask = indices < len(trade_dates_np)
        result = np.empty(len(dates), dtype=object)
        result[mask] = trade_dates_np[indices[mask]]
        result[~mask] = pd.NaT
        return result
    
    # 应用向量化对齐
    df = df.copy()
    valid_mask = df[pubDate_col].notna()
    df.loc[valid_mask, 'aligned_date'] = align_date_vectorized(df.loc[valid_mask, pubDate_col])
    df.loc[~valid_mask, 'aligned_date'] = pd.NaT
    
    # 创建交易日历DataFrame
    unique_codes = df['code'].unique()
    trade_df = pd.MultiIndex.from_product(
        [trade_dates, unique_codes], 
        names=['time', 'code']
    ).to_frame(index=False)
    
    # 合并数据
    merged = trade_df.merge(
        df,
        left_on=['code', 'time'],
        right_on=['code', 'aligned_date'],
        how='left'
    )
    
    # 前向填充
    merged.sort_values(['code', 'time'], inplace=True)
    cols_to_fill = [col for col in merged.columns if col not in ['code', 'time', 'aligned_date']]
    merged[cols_to_fill] = merged.groupby('code')[cols_to_fill].ffill()
    
    merged.drop(columns=['aligned_date'], inplace=True)
    return merged

def get_trade_cal(start_date, end_date):
    print("获取所有交易日: ")
    path = '可用交易日_SZSE.csv'
    if ignore_exists and os.path.exists(path):
        trade_cal_df = read_df(path)
    else:
        trade_cal_df = pro.trade_cal(exchange='SZSE', start_date=str(start_date), end_date=str(end_date), is_open='1')
        trade_cal_df.to_csv(path, index=False)
    all_dates = sorted(pd.to_datetime(trade_cal_df['cal_date'], format='%Y%m%d').to_list())
    print(f"{len(all_dates)}个交易日")
    return all_dates

def get_all_ts_codes(start_date, end_date):
    # 筛选日期内都上市的股票, 后续还要过滤掉季度财报不完整的股票保存到all_ts_codes.csv
    print("获取所有股票代码: ")
    path_in_date = '日期内可用股票.csv'
    if ignore_exists and os.path.exists(path_in_date):
        stock_basic_df_in_date = read_df(path_in_date)
    else:
        stock_basic_df = pro.stock_basic()
        stock_basic_df_in_date = stock_basic_df[(stock_basic_df['list_date'].astype(int)<=start_date)]
        save_df(stock_basic_df_in_date, path_in_date)
    all_ts_codes = stock_basic_df_in_date['ts_code'].to_list()
    print(f"{len(all_ts_codes)}个股票")
    return all_ts_codes

def get_daily(all_ts_codes, start_date, end_date):
    # get_price()
    # A股日线行情
    print("get_price: ")
    daily_dir = 'daily'
    daily_df_path = 'daily.csv'
    if ignore_exists and os.path.exists(daily_df_path):
        pass
    else:
        os.makedirs(daily_dir, exist_ok=True)
        daily_df_list = []
        with tqdm(all_ts_codes, desc="get_price: ") as pbar:
            for ts_code in pbar:
                pbar.set_description(f"{ts_code}")
                path = os.path.join(daily_dir, f'{ts_code}.csv')
                if ignore_exists:
                    if os.path.exists(path):
                        df = read_df(path)
                        if df.empty:
                            continue
                        daily_df_list.append(df)
                        continue
                empty_max_try = 5
                empty_try = 0
                while empty_try < empty_max_try:
                    try:
                        df = pro.daily(**{
                            "ts_code": ts_code,
                            "trade_date": "",
                            "start_date": str(start_date),
                            "end_date": str(end_date),
                            "limit": "",
                            "offset": ""
                        }, fields=[
                            'ts_code', #str	股票代码
                            'trade_date', #str	交易日期
                            'open', #float	开盘价
                            'close', #float	收盘价
                            'pre_close', #float	昨收价【除权价，前复权】
                        ])
                        if df.empty:
                            pbar.set_description(f'空数据。重试{empty_try+1}')
                            time.sleep(4)
                            empty_try += 1
                        else:
                            break
                    except Exception as ex:
                        pbar.set_description(f'{ex}。重试')
                        time.sleep(2)
                if not df.empty:
                    save_df(df, path)
                    daily_df_list.append(df)
            
        # daily_dir = 'daily'
        # daily_df_list = []
        # for i in tqdm(os.listdir(daily_dir)):
        #     ip = os.path.join(daily_dir, i)
        #     df = pd.read_excel(ip, sheet_name='Sheet1')
        #     daily_df_list.append(df)
        daily_df_all = pd.concat(daily_df_list)
        daily_df_all.to_csv(daily_df_path, index=False)
    print("完成")
def get_hs300(start_date, end_date):
    # 沪深300 指数
    # 000300.SH和399300.SZ一样
    print("get_hs300")
    path = 'index_daily_399300.SZ.csv'
    if ignore_exists and os.path.exists(path):
        pass
    else:
        empty_max_try = 5
        empty_try = 0
        while empty_try < empty_max_try:
            try:
                index_daily_df = pro.index_daily(**{
                    "ts_code": "399300.SZ",
                    "trade_date": "",
                    "start_date": str(start_date),
                    "end_date": str(end_date),
                })
                if index_daily_df.empty:
                    print(f'空数据。重试{empty_try+1}')
                    time.sleep(4)
                    empty_try += 1
                else:
                    break
            except Exception as ex:
                print(f'{ex}。重试')
                time.sleep(2)
        save_df(index_daily_df, path)

def get_daily_basic(all_ts_codes, start_date, end_date):
    '''get_valuation()
    每日指标'''
    print("get_valuation: ")
    daily_basic_dir = 'daily_basic'
    daily_basic_path = 'daily_basic.csv'
    if ignore_exists and os.path.exists(daily_basic_path):
        daily_basic_df_all = read_df(daily_basic_path)
    else:
        os.makedirs(daily_basic_dir, exist_ok=True)
        daily_basic_df_all_list = []
        with tqdm(all_ts_codes, desc="get_valuation: ") as pbar:
            for ts_code in pbar:
                pbar.set_description(f"{ts_code}")
                path = os.path.join(daily_basic_dir, f'{ts_code}.csv')
                if ignore_exists:
                    if os.path.exists(path):
                        daily_basic_df = read_df(path)
                        if daily_basic_df.empty:
                            continue
                        daily_basic_df_all_list.append(daily_basic_df)
                        continue
                empty_max_try = 5
                empty_try = 0
                while empty_try < empty_max_try:
                    try:
                        daily_basic_df = pro.daily_basic(**{
                            "ts_code": ts_code,
                            "trade_date": "",
                            "start_date": str(start_date),
                            "end_date": str(end_date),
                            "limit": "",
                            "offset": ""
                        }, fields=[
                            'ts_code',#str	TS股票代码
                            'trade_date',#str	交易日期
                            'close',#float	当日收盘价
                            'turnover_rate',#float	换手率（%）
                            'turnover_rate_f',#float	换手率（自由流通股）
                            'volume_ratio',#float	量比
                            'pe',#float	市盈率（总市值/净利润， 亏损的PE为空）
                            'pe_ttm',#float	市盈率（TTM，亏损的PE为空）
                            'pb',#float	市净率（总市值/净资产）
                            'ps',#float	市销率
                            'ps_ttm',#float	市销率（TTM）
                            'dv_ratio',#float	股息率 （%）
                            'dv_ttm',#float	股息率（TTM）（%）
                            'total_share',#float	总股本 （万股）
                            'float_share',#float	流通股本 （万股）
                            'free_share',#float	自由流通股本 （万）
                            'total_mv',#float	总市值 （万元）
                            'circ_mv',#float	流通市值（万元）
                        ])
                        if daily_basic_df.empty:
                            pbar.set_description(f'空数据。重试{empty_try+1}')
                            time.sleep(4)
                            empty_try += 1
                        else:
                            break
                    except Exception as ex:
                        pbar.set_description(f'{ex}。重试')
                        time.sleep(2)
                # daily_basic_df = daily_basic_df[daily_basic_df['trade_date'].astype(int).between(20190101, 20250101, inclusive='both')]
                if not daily_basic_df.empty:
                    save_df(daily_basic_df, path)
                    daily_basic_df_all_list.append(daily_basic_df)

        # daily_basic_df_all_list = []
        # for i in tqdm(all_ts_codes):
        #     ip = os.path.join(daily_basic_dir, f'{i}.csv')
        #     df = pd.read_excel(ip, sheet_name='Sheet1')
        #     daily_basic_df_all_list.append(df)
        daily_basic_df_all = pd.concat(daily_basic_df_all_list)
        daily_basic_df_all.sort_values(['ts_code', 'trade_date'], inplace=True)
        daily_basic_df_all.reset_index(drop=True, inplace=True)
        daily_basic_df_all.to_csv(daily_basic_path, index=False)
    print("完成")
    return daily_basic_df_all

def get_bak_basic(all_ts_code, start_date, end_date):
    # get_industry()
    # 基础信息 或者 股票历史列表
    print("get_industry: ")
    path = 'all_industry.csv'
    if ignore_exists and os.path.exists(path):
        pass
    else:
        bak_basic_dir = 'bak_basic'
        os.makedirs(bak_basic_dir, exist_ok=True)
        bak_basic_list = []
        all_industry = pd.DataFrame()
        all_dates = sorted(pd.to_numeric(pd.read_csv('可用交易日_SZSE.csv')['cal_date']).to_list())
        all_dates_dt = pd.to_datetime(all_dates, format='%Y%m%d')
        with tqdm(all_ts_code, desc="get_industry: ") as pbar:
            for ts_code in pbar:
                pbar.set_description(f"{ts_code}")
                bak_basic_path = os.path.join(bak_basic_dir, f'{ts_code}.csv')
                if ignore_exists:
                    if os.path.exists(bak_basic_path):
                        bak_basic_df = read_df(bak_basic_path)
                        if bak_basic_df.empty:
                            continue
                        bak_basic_list.append(bak_basic_df)
                        continue
                empty_max_try = 5
                empty_try = 0
                while empty_try < empty_max_try:
                    try:
                        bak_basic_df = pro.bak_basic(**{
                            "trade_date": "",
                            "ts_code": ts_code,
                            "limit": "",
                            "offset": ""
                        }, fields=[
                            'ts_code',
                            'trade_date',
                            "industry",
                        ])
                        if bak_basic_df.empty:
                            pbar.set_description(f'空数据。重试{empty_try+1}')
                            time.sleep(4)
                            empty_try += 1
                        else:
                            break
                    except Exception as ex:
                        pbar.set_description(f'{ex}。重试')
                        time.sleep(2)
                if not bak_basic_df.empty:
                    # 处理缺失时间
                    bak_basic_df.drop(bak_basic_df[bak_basic_df['trade_date'].isnull()].index, inplace=True)

                    bak_basic_df = bak_basic_df[bak_basic_df['trade_date'].astype(int).between(start_date, end_date, inclusive='both')]
                    
                    bak_basic_df['trade_date'] = pd.to_datetime(bak_basic_df['trade_date'])
                    # # 1. 将日期列设为索引
                    bak_basic_df = bak_basic_df.set_index('trade_date')
                    # 2. 用 `a` 重建索引（补全缺失日期）
                    full_dates = pd.Index(all_dates_dt, name='trade_date')
                    bak_basic_df = bak_basic_df.reindex(full_dates)
                    # 3. 后向填充缺失值
                    bak_basic_df = bak_basic_df.bfill()
                    # 4. 重置索引（可选）
                    bak_basic_df = bak_basic_df.reset_index()
                    save_df(bak_basic_df, bak_basic_path)
                    bak_basic_list.append(bak_basic_df)
        
        all_industry = pd.concat(bak_basic_list)
        # 归类到申万一级分类。dd可用deepseek自动归类生成
        ins_table = {}
        for k, v in industry_d.items():
            for l3 in v:
                ins_table[l3] = k
        L1_list = []
        all_industry.fillna('其他', inplace=True)
        for i in all_industry['industry']:
            L1_list.append(ins_table[i])
        all_industry.insert(len(all_industry.columns), 'L1', L1_list)
        all_industry.to_csv(path, index=False)
    print("完成")

def get__by_ts_code(typ, ts_code):
    '''用于财务数据查漏'''
    if typ==0:
        df = pro.income_vip(**{
            "ts_code": ts_code,
            "ann_date": "",
            "f_ann_date": "",
            "start_date": "",
            "end_date": "",
            "period": "",
            "report_type": "",
            "comp_type": "",
            "is_calc": "",
            "limit": "",
            "offset": ""
        }, fields=[
            'ts_code', #AAAAAstr	Y	TS代码
            'ann_date', #AAAAAstr	Y	公告日期
            'f_ann_date', #AAAAAstr	Y	实际公告日期
            'end_date', #AAAAAstr	Y	报告期
            'report_type', #AAAAAstr	Y	报告类型 见底部表
            'comp_type', #AAAAAstr	Y	公司类型(1一般工商业2银行3保险4证券)
            'end_type', #AAAAAstr	Y	报告期类型
            'total_revenue', #AAAAAfloat	Y	营业总收入
            'revenue', #AAAAAfloat	Y	营业收入
            'total_cogs', #AAAAAfloat	Y	营业总成本
            'oper_cost', #AAAAAfloat	Y	减:营业成本
            'n_income_attr_p', #AAAAAfloat	Y	净利润(不含少数股东损益)
            'oth_compr_income', #AAAAAfloat	Y	其他综合收益
            'compr_inc_attr_p', #AAAAAfloat	Y	归属于母公司(或股东)的综合收益总额
            'total_opcost', #AAAAAfloat	N	营业总成本（二）
        ])
    elif typ==1:
        df = pro.balancesheet_vip(**{
            "ts_code": ts_code,
            "ann_date": "",
            "f_ann_date": "",
            "start_date": "",
            "end_date": "",
            "period": "",
            "report_type": "",
            "comp_type": "",
            "limit": "",
            "offset": ""
        }, fields=[
            'ts_code', #AAAAAAstr	Y	TS股票代码
            'ann_date', #AAAAAAstr	Y	公告日期
            'f_ann_date', #AAAAAAstr	Y	实际公告日期
            'end_date', #AAAAAAstr	Y	报告期
            'report_type', #AAAAAAstr	Y	报表类型
            'comp_type', #AAAAAAstr	Y	公司类型(1一般工商业2银行3保险4证券)
            'end_type', #AAAAAAstr	Y	报告期类型
            'total_share', #AAAAAAfloat	Y	期末总股本
            'st_borr', #AAAAAAfloat	Y	短期借款
            'total_assets', #AAAAAAfloat	Y	资产总计
            'non_cur_liab_due_1y', #AAAAAAfloat	Y	一年内到期的非流动负债
            'lt_payable', #AAAAAAfloat	Y	长期应付款
            'specific_payables', #AAAAAAfloat	Y	专项应付款
            'total_ncl', #AAAAAAfloat	Y	非流动负债合计
            'total_liab', #AAAAAAfloat	Y	负债合计
            'oth_eqt_tools_p_shr', #AAAAAAfloat	Y	其他权益工具(优先股)
        ])
    elif typ==2:
        df = pro.cashflow_vip(**{
            "ts_code": ts_code,
            "ann_date": "",
            "f_ann_date": "",
            "start_date": "",
            "end_date": "",
            "period": "",
            "report_type": "",
            "comp_type": "",
            "is_calc": "",
            "limit": "",
            "offset": ""
        }, fields=[
            'ts_code', #AAAAAAstr	Y	TS股票代码
            'ann_date', #AAAAAAstr	Y	公告日期
            'f_ann_date', #AAAAAAstr	Y	实际公告日期
            'end_date', #AAAAAAstr	Y	报告期
            'comp_type', #AAAAAAstr	Y	公司类型(1一般工商业2银行3保险4证券)
            'report_type', #AAAAAAstr	Y	报表类型
            'end_type', #AAAAAAstr	Y	报告期类型
            'net_profit', #AAAAAAfloat	Y	净利润
            'n_cashflow_act', #AAAAAAfloat	Y	经营活动产生的现金流量净额
            'n_cashflow_inv_act', #float	Y	投资活动产生的现金流量净额
            'c_pay_acq_const_fiolta', #AAAAAAfloat	Y	购建固定资产、无形资产和其他长期资产支付的现金
            'n_incr_cash_cash_equ', #AAAAAAfloat	Y	现金及现金等价物净增加额
            'c_cash_equ_end_period', #AAAAAAfloat	Y	期末现金及现金等价物余额
            'depr_fa_coga_dpba', #AAAAAAfloat	Y	固定资产折旧、油气资产折耗、生产性生物资产折旧
            'amort_intang_assets', #AAAAAAfloat	Y	无形资产摊销
            'lt_amort_deferred_exp', #AAAAAAfloat	Y	长期待摊费用摊销
        ])
    return df

def get_income_by_season(periods, all_ts_codes):
    # get_basic()
    # get_income()
    # 利润表
    print("get_income_by_season: ")
    income_by_season_dir = 'income_by_season'
    os.makedirs(income_by_season_dir, exist_ok=True)
    income_df_list = []
    with tqdm(periods, desc="get_income_by_season: ") as pbar:
        for period in pbar:
            pbar.set_description(f"{period}")
            ip = os.path.join(income_by_season_dir, f'{period}.csv')
            if ignore_exists:
                if os.path.exists(ip):
                    income_df = read_df(ip)
                    if income_df.empty:
                        continue
                    income_df_list.append(income_df)
                    continue
            empty_max_try = 5
            empty_try = 0
            while empty_try < empty_max_try:
                try:
                    income_df = pro.income_vip(**{
                        "ts_code": "",
                        "ann_date": "",
                        "f_ann_date": "",
                        "start_date": "",
                        "end_date": "",
                        "period": period,
                        "report_type": "",
                        "comp_type": "",
                        "is_calc": "",
                        "limit": "",
                        "offset": ""
                    }, fields=[
                        'ts_code', #AAAAAstr	Y	TS代码
                        'ann_date', #AAAAAstr	Y	公告日期
                        'f_ann_date', #AAAAAstr	Y	实际公告日期
                        'end_date', #AAAAAstr	Y	报告期
                        'report_type', #AAAAAstr	Y	报告类型 见底部表
                        'comp_type', #AAAAAstr	Y	公司类型(1一般工商业2银行3保险4证券)
                        'end_type', #AAAAAstr	Y	报告期类型
                        'total_revenue', #AAAAAfloat	Y	营业总收入
                        'revenue', #AAAAAfloat	Y	营业收入
                        'total_cogs', #AAAAAfloat	Y	营业总成本
                        'oper_cost', #AAAAAfloat	Y	减:营业成本
                        'n_income_attr_p', #AAAAAfloat	Y	净利润(不含少数股东损益)
                        'oth_compr_income', #AAAAAfloat	Y	其他综合收益
                        'compr_inc_attr_p', #AAAAAfloat	Y	归属于母公司(或股东)的综合收益总额
                        'total_opcost', #AAAAAfloat	N	营业总成本（二）
                    ])
                    if income_df.empty:
                        pbar.set_description(f'空数据。重试{empty_try+1}')
                        time.sleep(4)
                        empty_try += 1
                    else:
                        break
                except Exception as ex:
                    pbar.set_description(f'{ex}。重试')
                    time.sleep(2)
            # print(income_df)
            if not income_df.empty:
                save_df(income_df, ip)
                income_df_list.append(income_df)
    return income_df_list

def get_balancesheet_by_season(periods, all_ts_codes):
    # get_basic()
    # get_balance()
    # 资产负债表
    print("get_balancesheet_by_season: ")
    balance_by_season_dir = 'balance_by_season'
    os.makedirs(balance_by_season_dir, exist_ok=True)
    balancesheet_df_list = []
    with tqdm(periods, desc="get_balancesheet_by_season: ") as pbar:
        for period in pbar:
            pbar.set_description(f"{period}")
            ip = os.path.join(balance_by_season_dir, f'{period}.csv')
            if ignore_exists:
                if os.path.exists(ip):
                    balancesheet_df = read_df(ip)
                    if balancesheet_df.empty:
                        continue
                    balancesheet_df_list.append(balancesheet_df)
                    continue
            empty_max_try = 5
            empty_try = 0
            while empty_try < empty_max_try:
                try:
                    balancesheet_df = pro.balancesheet_vip(**{
                        "ts_code": '',
                        "ann_date": "",
                        "f_ann_date": "",
                        "start_date": "",
                        "end_date": "",
                        "period": period,
                        "report_type": "",
                        "comp_type": "",
                        "limit": "",
                        "offset": ""
                    }, fields=[
                        'ts_code', #AAAAAAstr	Y	TS股票代码
                        'ann_date', #AAAAAAstr	Y	公告日期
                        'f_ann_date', #AAAAAAstr	Y	实际公告日期
                        'end_date', #AAAAAAstr	Y	报告期
                        'report_type', #AAAAAAstr	Y	报表类型
                        'comp_type', #AAAAAAstr	Y	公司类型(1一般工商业2银行3保险4证券)
                        'end_type', #AAAAAAstr	Y	报告期类型
                        'total_share', #AAAAAAfloat	Y	期末总股本
                        'st_borr', #AAAAAAfloat	Y	短期借款
                        'total_assets', #AAAAAAfloat	Y	资产总计
                        'non_cur_liab_due_1y', #AAAAAAfloat	Y	一年内到期的非流动负债
                        'lt_payable', #AAAAAAfloat	Y	长期应付款
                        'specific_payables', #AAAAAAfloat	Y	专项应付款
                        'total_ncl', #AAAAAAfloat	Y	非流动负债合计
                        'total_liab', #AAAAAAfloat	Y	负债合计
                        'oth_eqt_tools_p_shr', #AAAAAAfloat	Y	其他权益工具(优先股)
                    ])
                    if balancesheet_df.empty:
                        pbar.set_description(f'空数据。重试{empty_try+1}')
                        time.sleep(4)
                        empty_try += 1
                    else:
                        break
                except Exception as ex:
                    pbar.set_description(f'{ex}。重试')
                    time.sleep(2)
            if not balancesheet_df.empty:
                save_df(balancesheet_df, ip)
                balancesheet_df_list.append(balancesheet_df)
    return balancesheet_df_list

def get_cashflow_by_season(periods, all_ts_codes):
    # get_basic()
    # get_cashflow()
    # 现金流量表
    print("get_cashflow_by_season: ")
    cashflow_by_season_dir = 'cashflow_by_season'
    os.makedirs(cashflow_by_season_dir, exist_ok=True)
    cashflow_df_list = []
    with tqdm(periods, desc="get_cashflow_by_season: ") as pbar:
        for period in pbar:
            pbar.set_description(f"{period}")
            ip = os.path.join(cashflow_by_season_dir, f'{period}.csv')
            if ignore_exists:
                if os.path.exists(ip):
                    cashflow_df = read_df(ip)
                    if cashflow_df.empty:
                        continue
                    cashflow_df_list.append(cashflow_df)
                    continue
            empty_max_try = 5
            empty_try = 0
            while empty_try < empty_max_try:
                try:
                    cashflow_df = pro.cashflow_vip(**{
                        "ts_code": '',
                        "ann_date": "",
                        "f_ann_date": "",
                        "start_date": "",
                        "end_date": "",
                        "period": period,
                        "report_type": "",
                        "comp_type": "",
                        "is_calc": "",
                        "limit": "",
                        "offset": ""
                    }, fields=[
                        'ts_code', #AAAAAAstr	Y	TS股票代码
                        'ann_date', #AAAAAAstr	Y	公告日期
                        'f_ann_date', #AAAAAAstr	Y	实际公告日期
                        'end_date', #AAAAAAstr	Y	报告期
                        'comp_type', #AAAAAAstr	Y	公司类型(1一般工商业2银行3保险4证券)
                        'report_type', #AAAAAAstr	Y	报表类型
                        'end_type', #AAAAAAstr	Y	报告期类型
                        'net_profit', #AAAAAAfloat	Y	净利润
                        'n_cashflow_act', #AAAAAAfloat	Y	经营活动产生的现金流量净额
                        'n_cashflow_inv_act', #float	Y	投资活动产生的现金流量净额
                        'c_pay_acq_const_fiolta', #AAAAAAfloat	Y	购建固定资产、无形资产和其他长期资产支付的现金
                        'n_incr_cash_cash_equ', #AAAAAAfloat	Y	现金及现金等价物净增加额
                        'c_cash_equ_end_period', #AAAAAAfloat	Y	期末现金及现金等价物余额
                        'depr_fa_coga_dpba', #AAAAAAfloat	Y	固定资产折旧、油气资产折耗、生产性生物资产折旧
                        'amort_intang_assets', #AAAAAAfloat	Y	无形资产摊销
                        'lt_amort_deferred_exp', #AAAAAAfloat	Y	长期待摊费用摊销
                    ])
                    if cashflow_df.empty:
                        pbar.set_description(f'空数据。重试{empty_try+1}')
                        time.sleep(4)
                        empty_try += 1
                    else:
                        break
                except Exception as ex:
                    pbar.set_description(f'{ex}。重试')
                    time.sleep(2)
            # print(cashflow_df)
            if not cashflow_df.empty:
                save_df(cashflow_df, ip)
                cashflow_df_list.append(cashflow_df)
    return cashflow_df_list

def combine_finance(all_ts_codes, periods, income_df_list, balancesheet_df_list, cashflow_df_list):
    '''季度财务报表合为一个'''
    print("合并财务报表")
    combined_df_path = 'finance.csv'
    df3 = []
    for df_list in [income_df_list, balancesheet_df_list, cashflow_df_list]:
        df1 = pd.concat(df_list)
        df1 = df1[df1['ts_code'].astype(str).isin(all_ts_codes)]
        df3.append(df1)
    # 补充数据
    df3_paths = [r'income', r'balance', r'cashflow']
    still_lacks = []
    count = 0
    for index in range(len(df3)):
        # 去重日期
        df3[index].drop_duplicates(subset=['ts_code', 'end_date'], keep='last', ignore_index=True, inplace=True)
        with tqdm(df3[index].groupby('ts_code'), desc=f"合并财务报表{df3_paths[index]}: ") as pbar:
            for ts_code, g in pbar:
                pbar.set_description(f"{ts_code}")
                if g['end_date'].shape[0] != 25:
                    while 1:
                        try:
                            found_df = get__by_ts_code(index, ts_code)
                            break
                        except Exception as ex:
                            pbar.set_description(f'{ex}。重试')
                            time.sleep(2)
                    for dt in set(periods).difference(set(g['end_date'].astype(str))):
                        row = found_df[found_df['end_date'].astype(str)==dt]
                        if row.empty:
                            still_lacks.append([ts_code, dt])
                            count += 1
                        else:
                            df3[index] =pd.concat([df3[index], row], ignore_index=True)
    # 过滤掉缺少数据的
    still_lacks = pd.DataFrame(still_lacks, columns=['ts_code', 'date'])
    still_lacks.drop_duplicates(subset=['ts_code'], keep='last', ignore_index=True, inplace=True)
    all_ts_codes = sorted(list(set(all_ts_codes) - set(still_lacks['ts_code'].astype(str))))
    all_ts_codes_df = pd.DataFrame(all_ts_codes, columns=['ts_code'])
    all_ts_codes_df.to_csv('已下载股票.csv', index=False)
    for index in range(len(df3)):
        # 去重日期, 保证横向拼接之前行数一致
        df3[index].drop_duplicates(subset=['ts_code', 'end_date'], keep='last', ignore_index=True, inplace=True)
        df3[index] = df3[index][df3[index]['ts_code'].astype(str).isin(all_ts_codes_df['ts_code'].to_list())]
        # 排序, 保证横向拼接之前顺序一致
        df3[index].sort_values(['ts_code', 'end_date', 'ann_date'], inplace=True)
        # 重置索引, 保证横向拼接之前索引一致
        df3[index].reset_index(drop=True, inplace=True)

    # 合并income，balance, cashflow
    # 横向合并
    combined_df = pd.concat(df3, axis=1)
    # 去重列名
    combined_df = combined_df.loc[:, ~combined_df.columns.duplicated(keep='first')]
    combined_df.sort_values(['ts_code', 'end_date', 'ann_date'], inplace=True)
    save_df(combined_df, combined_df_path)
    return combined_df


def calc_pcf_ratio(all_ts_codes, all_dates, start_date, end_date, combined_df: pd.DataFrame, daily_basic_df_all: pd.DataFrame):
    '''计算pcf_ratio和计算多个ttm'''
    print('计算pcf_ratio和计算多个ttm')
    combined_df = combined_df[combined_df['ts_code'].isin(all_ts_codes)]
    combined_df.rename(columns=name_exchange_anti, inplace=True)
    # 首先确保数据按ts_code和end_date排序
    combined_df.sort_values(['code', 'statDate'], inplace=True)
    # 计算np_parent_company_owners
    combined_df.loc[:, 'compr_inc_attr_p'] = combined_df['compr_inc_attr_p'].fillna(0)
    combined_df.loc[:, 'oth_compr_income'] = combined_df['oth_compr_income'].fillna(0)
    combined_df.loc[:, 'np_parent_company_owners'] = combined_df['compr_inc_attr_p'] - combined_df['oth_compr_income']
    combined_df.loc[:, 'pubDate'] = pd.to_datetime(combined_df['pubDate'], format='%Y%m%d')
    combined_df['statDate'] = pd.to_datetime(combined_df['statDate'], format='%Y%m%d')
    need_ttm = [
        'operating_revenue',  #年内累加
        'net_profit',  #年内累加
        'cash_equivalent_increase',  #年内不累加
        'total_operating_revenue',   #年内累加
        'total_operating_cost',   #年内累加
        'np_parent_company_owners',   #年内累加
        'operating_cost',   #年内累加
    ]
    # 定义计算4季度和的函数
    # def calculate_4_season_sum(group):
    #     for column in need_ttm:
    #         if column in [
    #             'net_profit',
    #         ]:
    #             group_0 = group.copy()
    #             group = group[group['statDate'].dt.month.isin([6, 12])]
    #             sum_month = 6
    #             rolling_count = 2
    #         else:
    #             sum_month = 3
    #             rolling_count = 4
    #         quarterly_column = '不累加_' + column
    #         ttm_sum_column = column + '(TTM)'
    #         if column in ['cash_equivalent_increase']:
    #             #年内不累加
    #             group[quarterly_column] = group[column]
    #         else:
    #             group[quarterly_column] = np.where(
    #                 group['statDate'].dt.month==sum_month,
    #                 group[column],
    #                 group[column] - group[column].shift(1)
    #             )

    #         # 一行代码计算4季度滚动和（不足4季度为NaN）
    #         group[ttm_sum_column] = group[quarterly_column].transform(
    #             lambda s: s.rolling(rolling_count, min_periods=rolling_count).sum()
    #         )
    #         group.drop(quarterly_column, axis=1, inplace=True)
    #         if column in [
    #             'net_profit',
    #         ]:
    #             group_0[ttm_sum_column] = pd.NA
    #             group_0.update(group)
    #             group = group_0
    #     return group
    def calculate_4_season_sum(group):
        # 创建副本以避免操作原始数据
        group = group.copy()
        
        for column in need_ttm:
            if column in ['net_profit']:
                # 保存原始数据的副本
                group_0 = group.copy()
                # 筛选6月和12月的数据
                mask = group['statDate'].dt.month.isin([6, 12])
                group_filtered = group[mask].copy()
                sum_month = 6
                rolling_count = 2
            else:
                group_filtered = group.copy()
                sum_month = 3
                rolling_count = 4
            
            quarterly_column = '不累加_' + column
            ttm_sum_column = column + '(TTM)'
            
            if column in ['cash_equivalent_increase']:
                # 年内不累加
                group_filtered.loc[:, quarterly_column] = group_filtered[column]
            else:
                # 使用 .loc 进行安全赋值
                group_filtered.loc[:, quarterly_column] = np.where(
                    group_filtered['statDate'].dt.month == sum_month,
                    group_filtered[column],
                    group_filtered[column] - group_filtered[column].shift(1)
                )

            # 计算4季度滚动和
            rolling_sum = group_filtered[quarterly_column].transform(
                lambda s: s.rolling(rolling_count, min_periods=rolling_count).sum()
            )
            group_filtered.loc[:, ttm_sum_column] = rolling_sum
            
            # 删除临时列，不使用 inplace
            group_filtered = group_filtered.drop(columns=[quarterly_column])
            
            if column in ['net_profit']:
                # 将结果合并回原始数据
                group_0 = group_0.drop(columns=[ttm_sum_column], errors='ignore')
                group_0 = group_0.merge(
                    group_filtered[['statDate', ttm_sum_column]],
                    on='statDate',
                    how='left'
                )
                group = group_0
            else:
                group = group_filtered
        return group
    print('计算季度财务报表ttm')
    combined_df = combined_df.groupby('code', group_keys=False).apply(calculate_4_season_sum)

    combined_df_onee = combined_df[['code', 'pubDate', 'statDate', 'cash_equivalent_increase(TTM)']]
    combined_df.rename(columns=name_exchange, inplace=True)
    combined_df_path = 'finance.csv'
    save_df(combined_df, combined_df_path)

    print('发布日对齐交易日')
    zzz = pubDate_align_tradedate(combined_df_onee, end_date=pd.to_datetime(str(end_date)), all_dates=all_dates, pubDate_col='statDate')
    zzz = zzz[(zzz['time'] <= pd.to_datetime(str(end_date))) & (zzz['time'] >= pd.to_datetime(str(start_date)))]
    def fill_dates(group):
        # # 1. 将日期列设为索引
        group = group.set_index('time')
        # 2. 用 `a` 重建索引（补全缺失日期）
        full_dates = pd.Index(all_dates, name='time')
        group = group.reindex(full_dates)
        # 3. 后向填充缺失值
        group = group.bfill()
        # 4. 重置索引（可选）
        group = group.reset_index()
        return group
    zzz = zzz.groupby('code', group_keys=False).apply(fill_dates)
    zzz.sort_values(['code', 'time'], inplace=True)
    zzz.drop_duplicates(subset=['code', 'time'], keep='last', inplace=True)
    zzz.reset_index(drop=True, inplace=True)
    zzz.rename(columns=name_exchange, inplace=True)
    daily_basic_df_all['trade_date'] = pd.to_datetime(daily_basic_df_all['trade_date'].astype(str))
    daily_basic_df_all = daily_basic_df_all.merge(
        zzz[['ts_code', 'trade_date', 'cash_equivalent_increase(TTM)']],
        on=['ts_code', 'trade_date'],
        how='left'
    )
    daily_basic_df_all['pcf_ratio'] = daily_basic_df_all['close'] * daily_basic_df_all['total_share']*10000/daily_basic_df_all['cash_equivalent_increase(TTM)']
    daily_basic_df_all.drop(columns=['cash_equivalent_increase(TTM)'], inplace=True)
    daily_basic_path = 'daily_basic.csv'
    daily_basic_df_all.to_csv(daily_basic_path, index=False)
    print('完成')

def get_report_rc(all_ts_codes, start_date, end_date):
    '''
    get_report()
    卖方盈利预测数据
    '''
    print("get_report_rc: ")
    report_rc_df_all_path = 'report_rc.csv'
    if ignore_exists and os.path.exists(report_rc_df_all_path):
        pass
    else:
        report_rc_dir = 'report_rc'
        os.makedirs(report_rc_dir, exist_ok=True)
        report_rc_list = []
        with tqdm(all_ts_codes, desc=f"get_report_rc: ") as pbar:
            for ts_code in pbar:
                pbar.set_description(f"{ts_code}")
                path = os.path.join(report_rc_dir, f'{ts_code}.csv')
                if ignore_exists:
                    if os.path.exists(path):
                        report_rc_df = read_df(path)
                        if report_rc_df.empty:
                            continue
                        report_rc_list.append(report_rc_df)
                        continue
                empty_max_try = 5
                empty_try = 0
                while empty_try < empty_max_try:
                    try:
                        report_rc_df = pro.report_rc(**{
                                "ts_code": ts_code,
                                "report_date": "",
                                "start_date": "",
                                "end_date": "",
                                "limit": "",
                                "offset": ""
                            }, fields=[
                                "ts_code",#股票代码
                                "report_date",#研报日期
                                "quarter",#预测报告期
                                "np",#预测净利润（万元）
                                "eps",#预测每股收益（元）
                                "rd",#预测股息率
                                "roe",#预测净资产收益率
                                ])
                        if report_rc_df.empty:
                            pbar.set_description(f'空数据。重试{empty_try+1}')
                            time.sleep(4)
                            empty_try += 1
                        else:
                            break
                    except Exception as ex:
                        pbar.set_description(f'{ex}。重试')
                        time.sleep(2)
                if not report_rc_df.empty:
                    save_df(report_rc_df, path)
                    report_rc_list.append(report_rc_df)
        # report_rc_list = []
        # for index, ts_code in enumerate(tqdm(all_ts_codes)):
        #     tp = os.path.join(report_rc_dir, f'report_rc_{ts_code}.csv')
        #     if not os.path.getsize(tp):
        #         continue
        #     rdf = pd.read_excel(tp)
        #     report_rc_list.append(rdf)

        report_rc_df_all = pd.concat(report_rc_list, ignore_index=True)
        report_rc_df_all.sort_values(['ts_code', 'report_date'], inplace=True)
        report_rc_df_all.to_csv(report_rc_df_all_path, index=False)
    print("完成")

def main(start_date, end_date):
    print(f'正在获取数据，时间：{start_date} 到 {end_date}')
    # 所有交易日
    all_dates = get_trade_cal(start_date, end_date)
    # 所有股票代码
    all_ts_codes = get_all_ts_codes(start_date, end_date)
    # import random
    # random.seed(111)
    # all_ts_codes = sorted(random.sample(all_ts_codes, 100))
    # print(f'随机选择{len(all_ts_codes)}只股票')
    # 所有季度
    periods = gen_periods(start_date, end_date)

    get_daily(all_ts_codes, start_date, end_date)
    get_hs300(start_date, end_date)
    daily_basic_df_all = get_daily_basic(all_ts_codes, start_date, end_date)
    get_bak_basic(all_ts_codes, start_date, end_date)

    get_report_rc(all_ts_codes, start_date, end_date)

    combined_df_path = 'finance.csv'
    if ignore_exists and os.path.exists(combined_df_path):
        print('读取finance.csv')
        combined_df = read_df(combined_df_path)
    else:
        income_df_list = get_income_by_season(periods, all_ts_codes)
        balancesheet_df_list = get_balancesheet_by_season(periods, all_ts_codes)
        cashflow_df_list = get_cashflow_by_season(periods, all_ts_codes)
        combined_df = combine_finance(all_ts_codes, periods, income_df_list, balancesheet_df_list, cashflow_df_list)
    calc_pcf_ratio(all_ts_codes, all_dates, start_date, end_date, combined_df, daily_basic_df_all)

if __name__ == '__main__':
    # 可修改
    # 是否跳过已下载的文件，适合中断重试时使用
    ignore_exists = True
    # 数据起始日期，之前的不会下载
    start_date = 20120101
    # 数据结束日期，之后的不会下载
    end_date = 20250101
    os.makedirs(f'tushare-{start_date}_{end_date}', exist_ok=True)
    os.chdir(f'tushare-{start_date}_{end_date}')
    main(start_date, end_date)