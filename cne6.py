import pandas as pd
import numpy as np
from rich.progress import track
from IPython.display import clear_output
import talib
from joblib import Parallel, delayed
from functools import wraps
import os
import warnings
import matplotlib.pyplot as plt
import math
# 绘制每个风格的三级因子IC序列
import matplotlib.colors as mcolors

pd.set_option('future.no_silent_downcasting', True)
warnings.filterwarnings('ignore')

from utils import name_exchange_anti, read_df, save_df

# 可修改
# 这里的日期需要存在于已下载的数据中。起始日期延迟五年，结束日期早一个月。比如：已下载数据是20100101到20250101，那这里的start_date不能早于20150101，end_date不能大于20241201。
# 起始冗余是为了需要历史五年数据的因子。结束冗余是为了后面训练模型需要计算股票未来收益
# 数据范围：20120104 ~ 20241230。开始日期比数据延后5年，就是20170104，结束时间提前一个月，就是20241130。也就是说可以正常计算出20170104到20241130的因子数据
start_date = 20190101  # 20170104
end_date = 20241101  # 20241130
# tushare_data.py运行结果文件夹。如果tushare_data.py运行只下载了100只股票，计算因子也只会计算这100个股票的。
数据文件夹 = 'tushare-20120101_20250101'


os.chdir(数据文件夹)
start_date = pd.to_datetime(str(start_date))
end_date = pd.to_datetime(str(end_date))
global_end_date = end_date
# 所有日期
all_dates = sorted(pd.to_datetime(pd.read_csv('可用交易日_SZSE.csv')['cal_date'], format='%Y%m%d').to_list())

def __start_end_date__(start_date, end_date, count, all_dates=all_dates):
    '''
    __start_end_date__是一个转化函数，将start_date、end_date、count三个参数转化为start_date、end_date两个参数
    start_date, end_date: pandas._libs.tslibs.timestamps.Timestamp
    count: 交易日数量。
    由get_trade_date函数衍生而来
    '''
    assert [start_date, end_date, count].count(None) <= 1
    if start_date != None and end_date == None:
        this_count = 0
        for d in all_dates:
            if d >= start_date:
                this_count += 1
            if this_count == count:
                end_date = d
                break
    elif start_date == None and end_date != None:
        this_count = 0
        for d in reversed(all_dates):
            if d <= end_date:
                this_count += 1
            if this_count == count:
                start_date = d
                break
    return start_date,  end_date
# 1


def get_trade_date(start_date=None, end_date=None, count=None, all_dates=all_dates):
    """
    获取一段时间内的交易日列表。
    start_date, end_date, count三个参数任意给定两个，则返回对应的交易日。
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


# 中位数去极值法：将超过上下限的极端值用上下限值代替，避免数据中的极端值对回归结果产生过多影响
def MAD_winsorize(x, multiplier=5):
    """MAD去除极端值"""
    x_M = np.nanmedian(x)
    x_MAD = np.nanmedian(np.abs(x-x_M))
    upper = x_M + multiplier * x_MAD
    lower = x_M - multiplier * x_MAD
    x[x > upper] = upper
    x[x < lower] = lower
    return x

# 带半衰期的权重计算
def get_exponent_weight(window, half_life, is_standardize=True):
    L, Lambda = 0.5**(1/half_life), 0.5**(1/half_life)
    W = []
    for i in range(window):
        W.append(Lambda)
        Lambda *= L
    W = np.array(W[::-1])
    if is_standardize:
        W /= np.sum(W)
    return W

# 公布日对齐交易日
def pubDate_align_tradedate(df: pd.DataFrame, end_date=global_end_date, all_dates=all_dates, pubDate_col='pubDate'):
    df.loc[:, pubDate_col] = pd.to_datetime(df[pubDate_col])
    # 获取交易日历
    trade_dates = sorted(get_trade_date(start_date=df[pubDate_col].min(), end_date=end_date, all_dates=all_dates))  # 获取从最早公布日到指定结束日的所有交易日
    trade_dates_np = np.array(trade_dates)
    
    # 创建日期对齐函数，将每个“公布日”对齐到不早于它的第一个交易日
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
    
    # 创建交易日历DataFrame：完整交易日-股票代码表
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

def try_except(func):
    '''在出错时不会抛出异常，而是直接返回NAN，常用于数据处理、批量计算等场景中，保证流程不中断'''
    @wraps(func)
    def decorated(*args, **kargs):
        try:
            return func(*args, **kargs)
        except:
            return np.nan
    return decorated


# 面板数据的rolling.apply
'''
面板数据包含个体列、时间列以及数值列.
如果我们需要在时间维度，对所有个体的数据计算移动窗口，再返回面板格式的数据，则可以使用这个函数。

此函数的操作步骤分成三步：
收到数据后先转化矩阵数据，以时间列为index，个体列为columns。
对矩阵数据进行移动窗口（rolling），然后施加某种操作（apply）。
操作完成后还原成面板数据，并去除缺失值。
'''
def panel_rolling_apply(
    df, time_col, id_col, value_col, window, apply_func, rolling_kargs={},
    dropna=True, fillna_value=None, fillna_method='ffill', parallel=False, min_periods=None
):
    """面板数据转换成矩阵数据，rolling apply，然后再转换回面板数据。支持并行。"""
    if min_periods is None:
        min_periods = window

    @try_except
    def __apply_func(group):
        group_name = group.index[-1]
        if len(group) < min_periods:
            return pd.Series(np.nan, index=group.columns, name=group_name)
        group = apply_func(group, axis=0)
        group.name = group_name
        return group

    tmp = pd.pivot_table(df, values=value_col, index=time_col, columns=id_col)

    tmp_rolling = tmp.rolling(window, **rolling_kargs)

    if parallel:
        tmp = Parallel(12)(delayed(__apply_func)(group)
                           for group in tmp_rolling)
        tmp = pd.concat(tmp, axis=1).T
        tmp.index.name = time_col
    else:
        tmp = tmp_rolling.apply(apply_func)

    if fillna_value is not None:
        tmp = tmp.fillna(fillna_value)
    else:
        if fillna_method is not None:
            if fillna_method == 'ffill':
                tmp = tmp.ffill()
            elif fillna_method == 'bfill':
                tmp = tmp.bfill()
            else:
                raise NotImplementedError

    if dropna:
        tmp = tmp.dropna(how='all')

    return pd.melt(tmp.reset_index(), id_vars=time_col, value_name=value_col)\
        .dropna().reset_index(drop=True)


# 得到披露截止日期
def __discDate(x):
    bias = {3: [0, 5, 1], 6: [0, 9, 1], 9: [0, 11, 1], 12: [1, 5, 1]}
    m = x.month
    d = pd.Timestamp(x.year+bias[m][0], bias[m][1], bias[m][2])
    return d


# 获取行情数据
# 该函数默认获取最近250个交易日，所有股票的后复权的OHLCV数据，获取的字段可以自定义。底层数据包含所有股票、指数的行情。
def get_price(type, codes=None, start_date=None, end_date=None, count=250, fields=None):
    """
    type: stock: 股票; index: 指数(比如沪深300)
    codes: list,或者,'all-stock'取所有股票,'all-index'取所有指数。
    start_date, end_date:str %Y%m%d 格式为'YYYYMMDD'
    count: 交易日数量。
    fields:可取['open', 'high', 'low', 'close', 'volume', 'money', 'pre_close']
    fq:默认后复权post
    """
    if type=='stock':
        df = daily_df_s  # 股票行情数据
    elif type=='index':
        df = SHSZ300_df  # 指数行情数据
    all__dates = df['time'].unique()
    all__dates = all__dates.tolist()
    all__dates.sort()
    start_date, end_date = __start_end_date__(start_date, end_date, count, all_dates=all__dates)
    # 筛选时间区间：只保留在起止日期之间的数据
    df = df[df['time'].between(start_date, end_date, inclusive='both')]
    # 筛选代码：如果 codes 为 'all-stock'、'all-index' 或 None，则不筛选，保留全部；否则只保留在 codes 列表中的股票或指数
    if codes=='all-stock' or codes=='all-index' or codes==None:
        pass
    else:
        df = df[df['code'].isin(codes)]
    # 筛选字段：如果 fields 为 None，保留所有字段；否则只保留 'time', 'code' 以及用户指定的字段
    if fields == None:
        pass
    else:
        fields = ['time', 'code'] + fields
        df = df[fields]
    return df


# 获取股票的估值数据
def get_valuation(codes=None, start_date=None, end_date=None, count=250, fields:list=None):
    """
    该函数默认获取最近250个交易日，所有股票的估值数据。
    codes: list股票代码列表；默认None获取所有股票数据；
    start_date, end_date: str %Y-%m-%d or %Y%m%d 或者 datetime 格式；
    count: int, 所取数据的交易日的数量，默认最近250个交易日；
    fields: list, 数据字段，默认全取。
    filter: 自定义过滤器
    """
    df = daily_basic_df_s
    all__dates = df['time'].unique()
    all__dates = all__dates.tolist()
    all__dates.sort()
    start_date, end_date = __start_end_date__(start_date, end_date, count, all_dates=all__dates)

    df = df[df['time'].between(start_date, end_date, inclusive='both')]
    if codes=='all-stock' or codes==None:
        pass
    else:
        df = df[df['code'].isin(codes)]
    if fields == None:
        pass
    else:
        fields = ['time', 'code'] + fields
        df = df[fields]
    return df


# 获取股票所属行业
def get_industry(codes=None, date=None):
    """
    给定日期、股票代码列表、行业类别列表，则能返回股票所属行业
    codes: list, 股票代码；
    date: 日期
    """
    df = industry_df_s
    if date == None:
        pass
    else:
        if isinstance(date, int):
            date = str(date)
        if isinstance(date, str):
            date = date.replace('-', '')
            date = pd.to_datetime(date, format='%Y%m%d')
        df = df[df['time']==date]
    if codes=='all-stock' or codes==None:
        pass
    else:
        df = df[df['code'].isin(codes)]
    return df


# 股票财务数据并表查询函数
def get_basic(codes=None, start_date=None, end_date=None, count=4, fields=None, ttm_dict={}):
    """
    获取股票财务数据后自动合并报表
    codes: 股票代码，list，默认获取全部。
    start_date/end_date: 开始和结束日期，默认为空。
    count: 默认为4，代表取最近4个季度的数据。
    fields: 相关字段，默认不取。可以使用`get_field_descrition`获取字段说明。
    ttm_dict: 指定需要计算ttm的字段（keys）、ttm聚合方法（values）构成的dict。结构如下。
        - {ttm_field1: agg_method1, ttm_field2: agg_method2, ...}
        - 常用的TTM聚合方法有['mean', 'sum', 'last', ...]或者自定义函数，可操作长度为4的Series。
        - ttm_dict和fields是取并集的，也即在此出现的字段，将自动合并至fields。
        最终计算的字段 = fields ∪ ttm_dict.keys()
    """
    df = combined_df
    if start_date==None and end_date==None:
        pass
    else:
        all__dates = df['statDate'].unique()
        all__dates = all__dates.tolist()
        all__dates.sort()
        start_date, end_date = __start_end_date__(start_date, end_date, count, all_dates=all__dates)
        df = df[df['statDate'].between(start_date, end_date, inclusive='both')]
    if codes=='all-stock' or codes==None:
        pass
    else:
        df = df[df['code'].isin(codes)]
    if fields == None and ttm_dict == {}:
        pass
    else:
        fields_0 = ['code', 'pubDate', 'statDate']
        if fields:
            fields_0 = fields_0 + fields
        if ttm_dict:
            for k, v in ttm_dict.items():
                fields_0.append(k + '(TTM)')  # 把每个 TTM 字段名加上后缀 (TTM) 加入字段列表
        df = df[fields_0]
    return df


# 本地数据的提取函数
def get_report(codes=None, start_date=None, end_date=None, count=365, year=None, fields=None):
    """
    获取股票研究报告数据。
    codes：list，股票代码列表；
    start_date/end_date:datetime or str，研究报告发布的开始日期、结束日期；
    count：自然日天数，当start_date/end_date起止日期不全时用来推算区间
    year：int or str or list[str|int]，指定提取预期年份，如2023年预期数据。
    fields：list，需要提取的字段列表，默认全取。可以通过get_field_description获取字段说明。
    """
    df = report_rc_df_s
    start_date, end_date = __start_end_date__(start_date, end_date, count)
    df = df[df['time'].between(start_date, end_date, inclusive='both')]
    if codes=='all-stock' or codes==None:
        pass
    else:
        df = df[df['code'].isin(codes)]
    if year == None:
        pass
    else:
        # 如果 year 不为 None，则只保留 quarter 字段以指定年份开头的数据（如2023Q1、2023Q2等）
        df = df[df['quarter'].astype(str).str.startswith(str(year))]
    if fields == None:
        pass
    else:
        fields = ['time', 'code'] + fields
        df = df[fields]
    return df


# 数据清洗
def clean_BARRA(factor):
    '''
    数据清洗包含以下步骤：
    首先，风格因子数据与价格数据对齐。（对于我的数据库而言，价格数据比较全面）
    因子暴露缺失值采用前值填充。
    在每一个横截面上，去除1分位以及99分位以外的极端值。
    在每一个横截面上，去均值、标准差。
    '''
    def __clean_factor(x):
        # quantile winsorize 极值处理
        x = x.clip(x.quantile(0.01), x.quantile(0.99), axis=1)  # 将每列（因子）小于1分位和大于99分位的值分别截断到1分位和99分位
        # fill nan 缺失值填充
        x = x.apply(lambda x: x.fillna(x.median()), axis=0)  # 用该列的中位数填充缺失值
        # demean/std z-score标准化
        x = x.apply(lambda x: (x - x.mean()) / (x.std()+1e-6), axis=0)  # 每列减去均值再除以标准差
        return x
    price = get_price(
        type='stock',
        start_date=factor.time.min(), end_date=factor.time.max(), 
        fields=['pre_close', 'close']).set_index(['code', 'time'])
    f = factor.copy().set_index(['code', 'time'])
    f, _ = f.align(price, join='right', axis=0)  # 将因子数据和价格数据按索引对齐（以价格数据为准，右对齐），保证因子数据的完整性
    f = f.groupby('code').ffill()
    f = f.groupby('time', group_keys=False).apply(__clean_factor)
    return f.reset_index().sort_values(['code', 'time']).reset_index(drop=True)

# SIZE

# 流通市值: circulating_market_cap

# Size（一级因子）
# Size（二级因子）
# LNCAP（三级因子）: 规模。流通市值的自然对数。
# Mid Cap（二级因子）
# MIDCAP（三级因子）: 中市值。首先取 Size 因子暴露的立方，然后以加权回归的方式对 Size 因子正交，最后进行去极值和标准化处理。
# 按照渤海证券的研报中说明，此处的加权回归是指横截面回归，权重为市值开根号。而此处的去极值为MAD方法，采用5倍MAD去除极端值。


def cal_Size(codes=None, start_date=None, end_date=None, count=None):
    print('\n正在计算Size因子')
    def __reg(df):
        y = df['sub_MIDCAP'].values
        X = np.c_[np.ones((len(y), 1)), df['LNCAP'].values]
        W = np.diag(np.sqrt(df['circulating_market_cap']))
        beta = np.linalg.pinv(X.T@W@X)@X.T@W@y
    # 去除极端值
        resi = MAD_winsorize(y - X@beta, multiplier=5)
    # 标准化
        resi -= np.nanmean(resi)
        resi /= np.nanstd(resi)
        return pd.Series(resi, index=df['code'])
    # 获取数据
    tmp = get_valuation(codes=codes, start_date=start_date,
                        end_date=end_date, count=count, fields=['circulating_market_cap'])
    # 第一个三级因子
    tmp['LNCAP'] = np.log(tmp['circulating_market_cap']+1)
    tmp['sub_MIDCAP'] = tmp['LNCAP']**3
    # 截面回归正交化处理
    MIDCAP = tmp.groupby('time').apply(__reg, include_groups=False)
    MIDCAP.name = 'MIDCAP'
    tmp = tmp.merge(MIDCAP.reset_index())
    # tmp['Size'] = (tmp['LNCAP']+tmp['MIDCAP'])/2
    print('完成计算Size因子\n')
    return tmp[['code', 'time', 'LNCAP', 'MIDCAP']]

# Volatility
# Volatility（一级因子）
# Beta（二级因子）
# BETA（三级因子）:股票收益率对沪深 300 收益率进行时间序列回归，取回归系数，回归时间窗口为 252 个交易日，半衰期 63 个交易日。

# Residual Volatility（二级因子）
# Hist sigma（三级因子）:在计算 BETA 所进行的时间序列回归中，取回归残差收益率的波动率。
# Daily std（三级因子）:日收益率在过去 252 个交易日的波动率，半衰期 42 个交易日。
# Cumulative range（三级因子）: 累计收益范围（按月收益计算）,计算方法见下方。
# Z(T)为过去T个月的累计对数收益率（每个月包含21个交易日）

def cal_Volatility(codes=None, start_date=None, end_date=None, count=250, window=252, half_life=63):
    print('正在计算Volatility因子')
    if codes is None:
        codes = 'all-stock'
    s, end_date = __start_end_date__(
        start_date=start_date, end_date=end_date, count=count)
    start_date, _ = __start_end_date__(
        start_date=None, end_date=s, count=window)
    price = get_price(type='stock', codes=codes, start_date=start_date, end_date=end_date, count=count, fields=['pre_close', 'close'])
    hs300 = get_price(type='index', codes=['399300.SZ'], start_date=start_date, end_date=end_date, fields=['pre_close', 'close'])
    price = pd.concat([price, hs300]).reset_index(drop=True)
    # 计算每只股票和指数的日收益率
    price['ret'] = price['close'] / price['pre_close'] - 1
    ret = pd.pivot_table(price, values='ret', index='time', columns='code')

    L, Lambda = 0.5**(1/half_life), 0.5**(1/half_life)
    W = []
    for i in range(window):
        W.append(Lambda)
        Lambda *= L
    W = W[::-1]

    # 计算BETA（回归系数），Hist_sigma（回归残差波动率）
    beta, hist_sigma = [], []
    for i in track(range(len(ret)-window+1), description='正在计算beta...'):
        tmp = ret.iloc[i:i+window, :].copy()
        W_full = np.diag(W)
        
        # 筛选出具有完整数据的股票（在当前窗口内没有 NaN 的列）
        Y_full = tmp.dropna(axis=1)
        if '399300.SZ' in Y_full.columns:  # 399300.SZ代表沪深300指数
            Y_full = Y_full.drop(columns='399300.SZ')
        idx_full, Y_full = Y_full.columns, Y_full.values
        X_full = np.c_[np.ones((window, 1)), tmp.loc[:, '399300.SZ'].values]
        # 求加权最小二乘
        theta_full = np.linalg.pinv(
            X_full.T@W_full@X_full)@X_full.T@W_full@Y_full
        hist_sigma_full = pd.Series(
            np.std(Y_full - X_full@theta_full, axis=0), index=idx_full, name=tmp.index[-1])
        beta_full = pd.Series(theta_full[1], index=idx_full, name=tmp.index[-1])

        beta_lack, hist_sigma_lack = {}, {}
        # 不具有完整数据的股票（存在缺失值）
        for c in set(tmp.columns) - set(idx_full) - set('399300.SZ'):
            tmp_ = tmp.loc[:, [c, '399300.SZ']].copy()
            tmp_.loc[:, 'W'] = W
            tmp_ = tmp_.dropna()
            W_lack = np.diag(tmp_['W'])
            if len(tmp_) < half_life:  # half_life：最小有效观测数阈值
                continue
            X_lack = np.c_[np.ones(len(tmp_)), tmp_['399300.SZ'].values]
            Y_lack = tmp_[c].values
            beta_tmp = np.linalg.pinv(
                X_lack.T@W_lack@X_lack)@X_lack.T@W_lack@Y_lack
            hist_sigma_lack[c] = np.std(Y_lack - X_lack@beta_tmp)
            beta_lack[c] = beta_tmp[1]
        beta_lack = pd.Series(beta_lack, name=tmp.index[-1])
        beta.append(pd.concat([beta_full, beta_lack]).sort_index())
        hist_sigma_lack = pd.Series(hist_sigma_lack, name=tmp.index[-1])
        hist_sigma.append(
            pd.concat([hist_sigma_full, hist_sigma_lack]).sort_index())
    beta = pd.concat(beta, axis=1).T
    beta = pd.melt(beta.reset_index(), id_vars='index').dropna()
    beta.columns = ['time', 'code', 'BETA']
    hist_sigma = pd.concat(hist_sigma, axis=1).T
    hist_sigma = pd.melt(hist_sigma.reset_index(), id_vars='index').dropna()
    hist_sigma.columns = ['time', 'code', 'Hist_sigma']
    factor = pd.merge(beta, hist_sigma)

    # 采用EWMA方法计算日标准差Daily std
    # Daily std定义为日收益率在过去 252 个交易日的波动率，半衰期 42 个交易日
    # init_std = ret.std(axis=0)
    L = 0.5**(1/42)
    init_var = ret.var(axis=0)
    tmp = init_var.copy()
    daily_std = {}
    for t, k in track(ret.iterrows(), description='正在计算Daily std...'):
        tmp = tmp*L + k**2*(1-L)
        daily_std[t] = np.sqrt(tmp)
        tmp = tmp.fillna(init_var)
    daily_std = pd.DataFrame(daily_std).T
    daily_std.index.name = 'time'
    daily_std = daily_std.loc[s:end_date, :]
    daily_std = pd.melt(daily_std.reset_index(),
                        id_vars='time', value_name='Daily_std').dropna()

    factor = factor.merge(daily_std)

    # Cumulative range 累计收益范围（按月收益计算）
    close = pd.pivot_table(price, values='close', index='time',
                            columns='code').fillna(method='ffill', limit=10)
    pre_close = pd.pivot_table(price, values='pre_close', index='time',
                                columns='code').fillna(method='ffill', limit=10)
    idx = close.index
    CMRA = {}
    for i in track(range(len(close)-window+1), description='正在计算CMRA...'):
        close_ = close.iloc[i:i+window, :]
        pre_close_ = pre_close.iloc[i, :]
        pre_close_.name = pre_close_.name - pd.Timedelta(days=1)
        close_ = pd.concat([close_, pre_close_.to_frame().T]).sort_index(
        ).iloc[list(range(0, 253, 21)), :]
        r_tau = close_.pct_change().dropna(how='all')
        Z_T = np.log(r_tau+1).cumsum(axis=0)
        CMRA[idx[i+window-1]] = Z_T.max(axis=0) - Z_T.min(axis=0)  # Z_t 为过去T个月的累计对数收益率（每个月包含21个交易日）

    CMRA = pd.DataFrame(CMRA).T
    CMRA.index.name = 'time'
    CMRA = pd.melt(CMRA.reset_index(), id_vars='time',
                    value_name='Cumulative_range').dropna()
    factor = factor.merge(CMRA)
    factor.loc[factor['code']=='399300.SZ', 'BETA'] = factor.loc[factor['code']=='399300.SZ', 'BETA'].apply(lambda x: x[0])
    clear_output()
    print('完成计算Volatility因子\n')
    return factor.reset_index(drop=True).sort_values(by=['code', 'time'])


# Liquidity
# Liquidity（一级因子）
# Liquidity（二级因子）
# Monthly share turnover（三级因子）：月换手率。对最近21个交易日的股票换手率求和，然后取对数。
# Quarterly share turnover（三级因子）：季换手率。计算公式定义如下（T=3）：
# Annual share turnover（三级因子）：年换手率。将上式中的T变成12。
# Annualized traded value ratio（三级因子）：年化交易量比率。对日换手率进行加权求和，时间窗口为252个交易日，半衰期为63个交易日。


def cal_Liquidity(codes=None, start_date=None, end_date=None, count=250):
    print('正在计算Liquidity因子')
    '''
    流动性因子计算主要使用了股票的换手率数据 turnover_ratio，而换手率数据保存在估值数据表格valuation中。
    我们先使用以下代码提取并整理数据。
    注意，这里的换手率为百分比形式，我们需要对其进行转换。
    此外，利用pandas数据透视表，可以将面板数据转化为矩阵形式。
    '''
    s, end_date = __start_end_date__(
        start_date=start_date, end_date=end_date, count=count)
    start_date, _ = __start_end_date__(start_date=None, end_date=s, count=252)
    tmp = get_valuation(codes=codes, start_date=start_date,
                        end_date=end_date, count=count, fields=['turnover_ratio'])
    tmp['turnover_ratio'] /= 100.
    tmp = pd.pivot_table(tmp, index='time', columns='code',
                         values='turnover_ratio')
    
    # 月、季、年换手率
    monthly_share_turnover = np.log(tmp.rolling(21).sum())
    idx = list(range(20, 252, 21))
    quarterly_share_turnover, annual_share_turnover = {}, {}

    for i in track(range(len(tmp) - 251), description='正在计算Monthly_share_turnover, Quarterly_share_turnover, Annual_share_turnover...'):
        t = tmp.index[i+251]
        mst = np.exp(monthly_share_turnover.iloc[i:i+252, :].iloc[idx, :])
        quarterly_share_turnover[t] = np.log(mst.iloc[-3:, :].mean(axis=0))
        annual_share_turnover[t] = np.log(mst.mean(axis=0))
    quarterly_share_turnover = pd.DataFrame(quarterly_share_turnover).T
    annual_share_turnover = pd.DataFrame(annual_share_turnover).T
    quarterly_share_turnover.index.name = 'time'
    annual_share_turnover.index.name = 'time'
    monthly_share_turnover = monthly_share_turnover.loc[s:end_date, :]
    monthly_share_turnover = pd.melt(monthly_share_turnover.reset_index(
    ), id_vars='time', value_name='Monthly_share_turnover').dropna()
    quarterly_share_turnover = pd.melt(quarterly_share_turnover.reset_index(
    ), id_vars='time', value_name='Quarterly_share_turnover').dropna()
    annual_share_turnover = pd.melt(annual_share_turnover.reset_index(
    ), id_vars='time', value_name='Annual_share_turnover').dropna()

    factor = monthly_share_turnover.merge(
        quarterly_share_turnover).merge(annual_share_turnover)

    # 年化交易量比率
    # 在此因子的计算过程中，考虑到股票停牌的影响，我对计算方法做了些许调整。即先计算换手率的加权平均，再对平均日换手率年化处理。
    window, half_life = 252, 63
    L, Lambda = 0.5**(1/half_life), 0.5**(1/half_life)
    W = []
    for i in range(window):
        W.append(Lambda)
        Lambda *= L
    W = np.array(W[::-1])/np.mean(W)

    annualized_traded_value_ratio = []
    for i in track(range(len(tmp)-251), description='正在计算Annualized_traded_value_ratio...'):
        tmp_ = tmp.iloc[i:i+252, :].copy()
        annualized_traded_value_ratio.append(
            pd.Series(np.nanmean(tmp_.values*W.reshape(-1, 1), axis=0),
                      index=tmp.columns, name=tmp_.index[-1])
        )
    annualized_traded_value_ratio = pd.concat(
        annualized_traded_value_ratio, axis=1).T * window
    annualized_traded_value_ratio.index.name = 'time'
    annualized_traded_value_ratio = pd.melt(annualized_traded_value_ratio.reset_index(
    ), id_vars='time', value_name='Annualized_traded_value_ratio').dropna()
    factor = factor.merge(annualized_traded_value_ratio)
    print('完成计算Liquidity因子\n')
    return factor


# Momentum（一级因子）
# Short Term reversal（二级因子）
# Short Term reversal（三级因子）：短期反转。最近一个月的加权累积对数日收益率。
# Seasonality（二级因子）
# Seasonality（三级因子）：季节因子。过去五年的已实现次月收益率的平均值。
# Industry Momentum（二级因子）
# Industry Momentum（三级因子）：行业动量。该指标描述个股相对中信一级行业的强度。（但我的数据库中没有中信一级行业，所以改用申万一级行业）
# Momentum（二级因子）
# Relative strength（三级因子）：相对于市场的强度。
# Historical alpha（三级因子）：在BETA计算所进行的时间序列回归中取回归截距项。

def cal_Momentum(codes=None, start_date=None, end_date=None, count=250):
    print('正在计算Momentum因子')
    s, end_date = __start_end_date__(
        start_date=start_date, end_date=end_date, count=count)
    if codes is None:
        codes_ = 'all-stock'
    # 短期反转
    s1, _ = __start_end_date__(start_date=None, end_date=s, count=41)
    price1 = get_price(type='stock', codes=codes_, start_date=s1,
                        end_date=end_date, fields=['close', 'pre_close'])
    price1['ret'] = price1['close'] / price1['pre_close'] - 1
    ret = pd.pivot_table(price1, values='ret', index='time', columns='code')  # 得到收益率矩阵
    
    # 计算因子
    r_n = ret.rolling(21).mean().dropna(how='all')

    W = get_exponent_weight(window=21, half_life=5)
    STREV = []
    for i in range(len(r_n)-20):
        tmp = np.log(1 + r_n.iloc[i:i+21, :].copy())
        tmp = pd.Series(np.sum(W.reshape(-1, 1)*tmp.values, axis=0),
                        name=tmp.index[-1], index=tmp.columns)
        STREV.append(tmp)
    STREV = pd.concat(STREV, axis=1).T
    STREV.index.name = 'time'
    STREV = pd.melt(STREV.reset_index(), id_vars='time',
                    value_name='Short_Term_reversal').dropna()
    # Seasonality
    # 季节因子被定义为过去五年的已实现次月收益率的平均值。我们直接获取后复权收盘价即可实现计算：
    trade_date = get_trade_date(start_date=s, end_date=end_date)
    season = {}
    for td in track(trade_date, description='正在计算季节性⋯⋯'):
        r_y = []
        for i in range(1, 6):
            td_shift = get_trade_date(
                start_date=td-pd.Timedelta(days=365*i), count=21)
            s_, e_ = td_shift[0], td_shift[-1]
            p_ = get_price(type='stock', codes=codes_, start_date=s_,
                            end_date=e_, fields=['close'])
            p_ = pd.pivot_table(
                p_, index='time', columns='code', values='close').ffill()
            r_y.append(p_.iloc[-1, :] / p_.iloc[0, :] - 1)
        season[td] = pd.concat(r_y, axis=1).mean(axis=1)
    season = pd.DataFrame(season).T
    season.index.name = 'time'
    season = pd.melt(season.reset_index(), id_vars='time',
                        value_name='Seasonality')

    factor = pd.merge(STREV, season)
    
    # Industry Momentum 行业动量
    # 首先计算个股相对强度 Relative Strength
    s2, _ = __start_end_date__(start_date=None, end_date=s, count=126)
    price = get_price(type='stock', codes=codes_, start_date=s2,
                        end_date=end_date, fields=['pre_close', 'close'])
    price['ret'] = price['close'] / price['pre_close'] - 1
    ret = pd.pivot_table(price, index='time', columns='code', values='ret')
    W = get_exponent_weight(window=126, half_life=21)
    RS = {}
    for i in track(range(len(ret)-125), description='正在计算个股强度⋯⋯'):
        tmp = ret.iloc[i:i+126, :].copy()
        # 缺失值在10%以内
        tmp = tmp.loc[:, tmp.isnull().sum(axis=0) / 252 <= 0.1].fillna(0.)
        tmp = np.log(1 + tmp)
        RS[i] = pd.Series(np.sum(W.reshape(-1, 1)
                                        * tmp.values, axis=0), index=tmp.columns)
    RS = pd.DataFrame(RS).T
    RS.index.name = 'time'
    RS = pd.melt(RS.reset_index(), id_vars='time',
                    value_name='RS').dropna().reset_index(drop=True)

    val = get_valuation(start_date=s, end_date=end_date,
                        fields=['circulating_market_cap'])
    RS = pd.merge(RS, val)
    
    # 改过
    '''
    新写法.dt.to_period('M') 按时间的月周期分组，然后取每组中最小的 time（即该月在数据里最早的时间戳），并把这个最早时间戳赋给组内所有行，结果是 Timestamp 类型（每行的 mon 为该月的“最早交易/记录时间”）
    旧写法产生字符串 "YYYY-MM-01"并非实际交易日，新写法产生真实的时间戳（数据中该月的第一个时间点），更适合基于交易日的分组/对齐
    '''
    # RS['mon'] = RS['time'].apply(lambda x: x.strftime("%Y-%m-01"))  # 把每个时间戳转换为该月的第一天的字符
    RS['mon'] = RS.groupby(RS['time'].dt.to_period('M'))['time'].transform('min')
    RS['c'] = np.sqrt(RS['circulating_market_cap'])

    # 然后计算行业相对强度
    def __industry_RS(x):
        ind_RS = x.groupby('L1').apply(
            lambda y: y['RS'].dot(y['c']) / y['c'].sum(), include_groups=False
        )
        ind_RS.name = 'ind_RS'
        ind_RS = ind_RS.reset_index()
        x = pd.merge(x, ind_RS)
        x['Industry_Momentum'] = x['ind_RS'] - x['RS']
        return x[['code', 'Industry_Momentum']].set_index('code')
    
    # 最后计算行业动量
    INDMOM = []
    for m, tmp_RS in track(RS.groupby('mon'), description='正在计算行业动量⋯⋯'):
        ind = get_industry(date=m)[['code', 'L1']]
        tmp_RS = pd.merge(tmp_RS, ind)
        INDMOM.append(tmp_RS.groupby('time').apply(
            __industry_RS, include_groups=False).reset_index())
    INDMOM = pd.concat(INDMOM).reset_index(drop=True)
    factor = factor.merge(INDMOM)

    # Relative strength 非滞后相对强度
    '''
    对股票的对数收益率进行半衰指数加权求和，时间窗口252个交易日，半衰期126个交易日。
    然后，以11个交易日为窗口，滞后11个交易日，取非滞后相对强度的等权平均值。
    '''
    s3, _ = __start_end_date__(start_date=None, end_date=s, count=262)
    W = get_exponent_weight(window=252, half_life=126)

    price = get_price(type='stock', codes=codes_, start_date=s3,
                        end_date=end_date, fields=['pre_close', 'close'])
    price['ret'] = np.log(price['close']) - np.log(price['pre_close'])
    ret = pd.pivot_table(price, index='time', columns='code', values='ret')

    Relative_strength = {}
    for i in track(range(len(ret) - 251), description='正在计算非滞后相对强度⋯⋯'):
        tmp = ret.iloc[i:i+252, :]
        tmp = tmp.loc[:, tmp.isnull().sum(axis=0) / 252 <= 0.1].fillna(0.)
        np.sum(W.reshape(-1, 1)*tmp.values, axis=0)
        Relative_strength[tmp.index[-1]] = pd.Series(
            np.sum(W.reshape(-1, 1)*tmp.values, axis=0), index=tmp.columns)
    Relative_strength = pd.DataFrame(Relative_strength).T
    Relative_strength.index.name = 'time'
    Relative_strength = Relative_strength.rolling(11).mean().dropna(how='all')
    Relative_strength = pd.melt(Relative_strength.reset_index(
    ), id_vars='time', value_name='Relative_strength').dropna().reset_index(drop=True)

    # Historical alpha 长期历史alpha
    half_life = 126
    window = 252
    hs300 = get_price(type='index', codes=['399300.SZ'], start_date=s3, end_date=end_date, fields=['pre_close', 'close'])
    price = pd.concat([price, hs300]).reset_index(drop=True)
    price['ret'] = price['close'] / price['pre_close'] - 1
    ret = pd.pivot_table(price, values='ret', index='time', columns='code')
    # hist_alpha（回归截距项）
    hist_alpha = []
    for i in track(range(len(ret)-window+1), description='正在计算Hist_alpha...'):
        tmp = ret.iloc[i:i+window, :].copy()
        W_full = np.diag(W)
        # 筛选出具有完整数据的股票
        Y_full = tmp.dropna(axis=1).drop(columns='399300.SZ')
        idx_full, Y_full = Y_full.columns, Y_full.values
        X_full = np.c_[np.ones((window, 1)), tmp.loc[:, '399300.SZ'].values]
        theta_full = np.linalg.pinv(
            X_full.T@W_full@X_full)@X_full.T@W_full@Y_full
        hist_alpha_full = pd.Series(theta_full[0], index=idx_full, name=tmp.index[-1])

        hist_alpha_lack = {}
        # 不具有完整数据的股票
        for c in set(tmp.columns) - set(idx_full) - set('399300.SZ'):
            tmp_ = tmp.loc[:, [c, '399300.SZ']].copy()
            tmp_.loc[:, 'W'] = W
            tmp_ = tmp_.dropna()
            W_lack = np.diag(tmp_['W'])
            if len(tmp_) < half_life:
                continue
            X_lack = np.c_[np.ones(len(tmp_)), tmp_['399300.SZ'].values]
            Y_lack = tmp_[c].values
            beta_tmp = np.linalg.pinv(
                X_lack.T@W_lack@X_lack)@X_lack.T@W_lack@Y_lack
            hist_alpha_lack[c] = beta_tmp[0]
        hist_alpha_lack = pd.Series(hist_alpha_lack, name=tmp.index[-1])
        hist_alpha.append(pd.concat([hist_alpha_full, hist_alpha_lack]).sort_index())
    hist_alpha = pd.DataFrame(hist_alpha)
    hist_alpha.drop(columns=['399300.SZ'], inplace=True)
    hist_alpha.index.name = 'time'
    hist_alpha = hist_alpha.rolling(11).mean().dropna(how='all')
    hist_alpha = pd.melt(hist_alpha.reset_index(
    ), id_vars='time', value_name='Hist_alpha').dropna().reset_index(drop=True)
    hist_alpha.columns = ['time', 'code', 'Hist_alpha']

    factor = factor.merge(Relative_strength).merge(hist_alpha)
    clear_output()
    print('完成计算Momentum因子\n')
    return factor


# Quality（一级因子）
# Leverage（二级因子）
# Market Leverage（三级因子）：市场杠杆
# Book Leverage（三级因子）：账面杠杆
# Debt to asset ratio（三级因子）：资产负债比
# Earning Variability（二级因子）
# Variation in Sales（三级因子）：营业收入波动率
# Variation in Earning（三级因子）：盈利波动率
# Variation in Cashflows（三级因子）：现金流波动率
# Standard deviation of Analyst Forecast Earnings-to-Price（三级因子）：分析师预测盈市率标准差
# Earnings Quality（二级因子）
# Accruals Balancesheet version（三级因子）：资产负债表应计项目
# Accruals Cashflow version（三级因子）：现金流量表应计项目
# 计算数据调整为半年度数据
# Leverage
def cal_Leverage(codes=None, start_date=None, end_date=None, count=250):
    print('正在计算Leverage')
    s, end_date = __start_end_date__(start_date=start_date, end_date=end_date, count=count)
    # 杠杆
    s1, _ = __start_end_date__(start_date=None, end_date=s, count=270)
    basic = get_basic(
        codes=codes, start_date=s1, end_date=end_date, 
        fields=[
        'total_non_current_liability', # 长期负债
        'total_assets', 'total_liability', # 总资产、总负债
        'oth_eqt_tools_p_shr' # 优先股
        ])
    tmp_col = ['total_non_current_liability', 'oth_eqt_tools_p_shr']

    # 只拿半年报和年报
    basic = basic[basic['statDate'].dt.month.isin([6, 12])]
    # 优先股单位调整
    basic['PE'] = (basic['oth_eqt_tools_p_shr'].fillna(0))/1e8  # 以元为单位
    # 长期负债单位调整
    basic['LD'] = (basic['total_non_current_liability'] / 1e8).fillna(0)
    # 计算必须披露的时间
    basic['discDate'] = basic['statDate'].apply(__discDate)
    basic = basic.query('pubDate<discDate').drop(columns='pubDate').rename(columns={'discDate':'pubDate'})  # 当发布日pubDate早于披露日discDate时，用披露日替代发布日

    # 交易日对齐
    basic = pubDate_align_tradedate(basic.drop(columns=tmp_col))
    val = get_valuation(codes=codes, start_date=s, end_date=end_date, fields=['market_cap', 'pb_ratio'])
    factor = pd.merge(basic, val).rename(columns={'market_cap': 'ME'})
    # pb值反推普通股账面价值
    factor['BE'] = factor['ME'] / factor['pb_ratio']
    # 因子计算
    factor['Market_Leverage'] = factor.eval('(ME+PE+LD)/ME')
    factor['Book_Leverage'] = factor.eval('(BE+PE+LD)/ME')
    factor['Debt_to_asset_ratio'] = factor.eval('total_liability/total_assets')
    factor = factor[['code', 'time', 'Market_Leverage', 'Book_Leverage', 'Debt_to_asset_ratio']]
    return factor.sort_values(['code', 'time'], ignore_index=True)


# Earnings Variability（二级因子）：盈利波动 
# Variation in Sales（三级因子）：营业收入波动率。过去五个财年的年营业收入标准差除以平均年营业收入。
# Variation in Earning（三级因子）：盈利波动率。过去五个财年的年净利润标准差除以平均年净利润。
# Variation in Cashflows（三级因子）：现金流波动率。过去五个财年的年现金及现金等价物净增加额标准差除以平均年现金及现金等价物净增加额。
# Standard deviation of Analyst Forecast Earnings-to-Price（三级因子）：分析师预测盈市率标准差。预测12月eps的标准差除以当前股价。

def cal_Earnings_Variability(codes=None, start_date=None, end_date=None, count=250):
    print('正在计算Earnings_Variability')
    s, end_date = __start_end_date__(start_date=start_date, end_date=end_date, count=count)
    # 盈利波动
    # 营业收入波动率、净利润波动率、现金流波动率
    basic = get_basic(
        codes=codes, 
        start_date=pd.to_datetime(s)-pd.Timedelta(days=365*6+1), end_date=end_date, 
        ttm_dict={'operating_revenue':'sum', 'net_profit':'sum', 'cash_equivalent_increase':'sum'})
    # 只拿半年报和年报
    basic = basic[basic['statDate'].dt.month.isin([6, 12])]
    # 计算必须披露的时间
    basic['discDate'] = basic['statDate'].apply(__discDate)
    basic = basic.query('pubDate<discDate').drop(columns='pubDate')\
        .rename(columns={'discDate':'pubDate'})\
        .sort_values(by=['code', 'statDate'])
    def variation(x, **kargs):
        return 4*np.nanstd(x) / np.nanmean(x)

    # 去极值和标准化（均值0 标准差1）处理
    def __modify(x):
        vars_ = ['Variation_in_Sales', 'Variation_in_Earning', 'Variation_in_Cashflow']
        for v in vars_:
            x[v] = MAD_winsorize(x[v].fillna(np.nanmedian(x[v])))
            x[v] -= x[v].mean()
            x[v] /= x[v].std()
        return x

    V_in_Sales = panel_rolling_apply(
        basic, time_col='pubDate', id_col='code', value_col='operating_revenue(TTM)', window=10, apply_func=variation, 
        ).rename(columns={'operating_revenue(TTM)': 'Variation_in_Sales'})
    V_in_Earning = panel_rolling_apply(
        basic, time_col='pubDate', id_col='code', value_col='net_profit(TTM)', window=10, apply_func=variation
        ).rename(columns={'net_profit(TTM)': 'Variation_in_Earning'})
    V_in_Cashflow = panel_rolling_apply(
        basic, time_col='pubDate', id_col='code', value_col='cash_equivalent_increase(TTM)', window=10, apply_func=variation
        ).rename(columns={'cash_equivalent_increase(TTM)': 'Variation_in_Cashflow'})
    factor_ = V_in_Sales.merge(V_in_Earning, how='outer').merge(V_in_Cashflow, how='outer')
    factor_ = factor_.groupby('pubDate').apply(__modify, include_groups=False).reset_index()
    factor = pubDate_align_tradedate(factor_)  # 公告日对齐到交易日，对非公告日用前向填充最新披露值
    factor = factor.loc[
        (factor['time']>=pd.to_datetime(s))&(factor['time']<=pd.to_datetime(end_date)), 
        ['code', 'time', 'Variation_in_Sales', 'Variation_in_Earning', 'Variation_in_Cashflow']]
    
    
    # 分析师预测的波动
    '''
    计算分析师EP波动率。
    该因子的计算方法为分析师预测当年12月eps的标准差除以当前股价。
    我认为这个数值可能不合理，因为股票分红、拆股等事件会影响股票股价，而分析师给出的eps预测只是基于当时的股票数量。
    所以，我将计算方法调整为分析师预测当年12月净利润的累计波动率，除以股票当天的市值。
    '''
    
    def __cumstd(x):
        f_ = []
        def __sub_cumstd(y, f_):
            f_ += y['np'].values.tolist()
            if len(f_)<5:
                return np.nan
            return np.nanstd(f_)
        np_std = x.groupby('time').apply(lambda z: __sub_cumstd(z, f_), include_groups=False)
        np_std.name = 'np_std'
        return np_std.dropna()

    forecast_EP_std = []
    for year in track(range(pd.to_datetime(s).year, pd.to_datetime(end_date).year+1), description='正在计算forecast_EP_std...'):
        forecast_np = get_report(codes=codes, end_date=pd.to_datetime(f'{year}-12-31', format='%Y-%m-%d'), count=365*3, year=year, fields=['np'])
        forecast_np['np'] /= 10000
        np_std = forecast_np.groupby('code').apply(__cumstd, include_groups=False).reset_index().rename(columns={'time': 'pubDate'})
        np_std = pubDate_align_tradedate(np_std, end_date=pd.to_datetime(f'{year}-12-31', format='%Y-%m-%d'))
        np_std = np_std[np_std['time'] >= pd.Timestamp(f'{year}-01-01')].reset_index(drop=True)
        val = get_valuation(codes=codes, start_date=pd.to_datetime(f'{year}-01-01', format='%Y-%m-%d'), end_date=pd.to_datetime(f'{year}-12-31', format='%Y-%m-%d'), fields=['market_cap'])
        f_EP_std = pd.merge(np_std, val)
        f_EP_std['forecast_EP_std'] = f_EP_std.eval('np_std/market_cap')
        forecast_EP_std.append(f_EP_std)
    forecast_EP_std = pd.concat(forecast_EP_std)
    forecast_EP_std = forecast_EP_std[(forecast_EP_std['time']<=pd.to_datetime(end_date))&(forecast_EP_std['time']>=pd.to_datetime(s))].reset_index(drop=True)
    factor = factor.merge(forecast_EP_std[['code', 'time', 'forecast_EP_std']], how='outer')
    return factor.reset_index(drop=True).sort_values(['code', 'time'], ignore_index=True)


# Earning Quality（二级因子）：盈利质量
# Accruals Balancesheet version（三级因子）：资产负债表应计项目。
# Accruals Cashflow version（三级因子）：现金流量表应计项目。

def cal_Earnings_Quality(codes=None, start_date=None, end_date=None, count=250):
    print('正在计算Earnings_Quality')
    s, end_date = __start_end_date__(start_date=start_date, end_date=end_date, count=count)
    # 资产负债表、现金流量表应计项目
    # 统一获取字段
    basic = get_basic(
        codes=codes, 
        start_date=pd.to_datetime(s)-pd.Timedelta(days=365*6+1), end_date=end_date, 
        fields=[
        'total_assets', 'total_liability', 'cash_and_equivalents_at_end', 
        'non_current_liability_in_one_year', 'total_non_current_liability', 'shortterm_loan',  # 总带息债务 
        'fixed_assets_depreciation', 'intangible_assets_amortization', 'defferred_expense_amortization' #折旧摊销
        ]).rename(columns={'total_assets': 'TA', 'total_liability': 'TL', 'cash_and_equivalents_at_end': 'Cash'})
    # 降频率至半年度
    basic['quarter'] = basic['statDate'].apply(lambda x: x.quarter)
    basic['year'] = basic['statDate'].apply(lambda x: x.year)
    basic = basic[basic['quarter'].isin([2, 4])]
    # 现金流量表和利润表字段
    cashflow = get_basic(
        codes=codes, 
        start_date=pd.to_datetime(s)-pd.Timedelta(days=365*6+1), end_date=end_date, 
        fields=['net_operate_cash_flow', 'net_invest_cash_flow']
        ).rename(columns={'net_operate_cash_flow':'CFO', 'net_invest_cash_flow':'CFI'})
    income = get_basic(
        codes=codes, 
        start_date=pd.to_datetime(s)-pd.Timedelta(days=365*6+1), end_date=end_date, 
        fields=['net_profit']
    ).rename(columns={'net_profit': 'NI'})
    # 合并表格
    basic = basic.merge(cashflow).merge(income)
    # 计算必须披露的时间
    basic['discDate'] = basic['statDate'].apply(__discDate)
    basic = basic.query('pubDate<discDate').drop(columns='pubDate')\
        .rename(columns={'discDate':'pubDate'})\
        .sort_values(by=['code', 'statDate'])
    # 累计数据做差分
    diff_col = [
        'fixed_assets_depreciation', 'intangible_assets_amortization', 'defferred_expense_amortization', 
        'CFO', 'CFI', 'NI'
    ]
    
    # 对财务累计项做差分处理（把累计值变成当期增量）
    def __diff(x):
        if len(x)==2:    
            x.loc[x.index[-1], diff_col] = x.loc[:, diff_col].diff().iloc[-1, :]  # 逐行差分，即“后一行 - 前一行”
        return x

    basic = basic.groupby(['code', 'year'], group_keys=False).apply(__diff).sort_values(by=['code', 'statDate'])
    # 计算中间变量
    basic['TD'] = basic.fillna(0).eval('non_current_liability_in_one_year + total_non_current_liability + shortterm_loan')
    basic['DA'] = basic.fillna(0).eval('fixed_assets_depreciation + intangible_assets_amortization + defferred_expense_amortization')
    basic['NOA'] = basic.fillna(0).eval('(TA-Cash)-(TL-TD)')
    #计算因子
    # ABS
    basic['delta_NOA'] = basic.groupby('code')['NOA'].diff()
    basic['ACCR_BS'] = basic.eval('delta_NOA-DA')
    basic['ABS'] = basic.eval('-ACCR_BS/TA')
    # ACF
    basic['ACCR_CF'] = basic.fillna(0).eval('NI-(CFO+CFI)+DA')
    basic['ACF'] = basic.fillna(0).eval('-ACCR_CF/TA')
    factor = basic[['code', 'pubDate', 'ABS', 'ACF']]
    factor = pubDate_align_tradedate(factor)
    factor = factor.loc[
        (factor['time']>=pd.to_datetime(s))&(factor['time']<=pd.to_datetime(end_date)), 
        ['code', 'time', 'ABS', 'ACF']]
    return factor.reset_index(drop=True).sort_values(['code', 'time'], ignore_index=True)


# Profitability（二级因子）：盈利能力
# 资产周转率Asset turnover（三级因子）
# 资产毛利率Gross profitability（三级因子）
# 销售毛利率Gross Profit Margin（三级因子）
# 总资产收益率 Return on assets（三级因子）

def cal_Profitability(codes=None, start_date=None, end_date=None, count=250):
    print('正在计算Profitability')
    s, end_date = __start_end_date__(start_date=start_date, end_date=end_date, count=count)
    basic = get_basic(
        codes=codes, 
        start_date=pd.to_datetime(s)-pd.Timedelta(days=365*6+1), end_date=end_date, 
        fields=['total_assets'], ttm_dict={
        'total_operating_revenue': 'sum', 'total_operating_cost': 'sum', 
        'np_parent_company_owners': 'sum'
        }).rename(columns={
        'total_assets': 'TA', 
        'total_operating_revenue(TTM)': 'Sales', 
        'total_operating_cost(TTM)': 'COGS', 
        'np_parent_company_owners(TTM)': 'Earnings'
        })
    # 计算必须披露的时间
    basic['discDate'] = basic['statDate'].apply(__discDate)
    basic = basic.query('pubDate<discDate').drop(columns='pubDate')\
        .rename(columns={'discDate':'pubDate'})\
        .sort_values(by=['code', 'statDate'])
    basic['ATO'] = basic.eval('Sales/TA')
    basic['GP'] = basic.eval('(Sales-COGS)/TA')
    basic['GPM'] = basic.eval('(Sales-COGS)/Sales')
    basic['ROA'] = basic.eval('Earnings/TA')
    factor = basic[['code', 'pubDate', 'ATO', 'GP', 'GPM', 'ROA']]
    factor = pubDate_align_tradedate(factor)
    factor = factor.groupby(['code', 'time'], as_index=False).mean()
    factor = factor.loc[
        (factor['time']>=pd.to_datetime(s))&(factor['time']<=pd.to_datetime(end_date)), 
        ['code', 'time', 'ATO', 'GP', 'GPM', 'ROA']]
    return factor.reset_index(drop=True).sort_values(['code', 'time'], ignore_index=True)


# Investment Quality（二级因子）：投资质量
# 总资产增长率Total Assets Growth Rate（三级因子）：最近5个财政年度的总资产对时间的回归的斜率值，除以平均总资产，最后取相反数
# 股票发行量增长率Issuance growth（三级因子）：最近5个财政年度的流通股本对时间的回归的斜率值，除以平均流通股本，最后取相反数
# 资本支出增长率Capital expenditure growth（三级因子）：将过去5个财政年度的资本支出对时间的回归的斜率值，除以平均资本支出，最后取相反数

def cal_Investment_Quality(codes=None, start_date=None, end_date=None, count=250):
    print('正在计算Investment_Quality')
    s, end_date = __start_end_date__(start_date=start_date, end_date=end_date, count=count)
    s, end_date = pd.to_datetime(s), pd.to_datetime(end_date)

    def __t_reg(x, field, min_period):
        """时间序列回归斜率除以平均值"""
        x = x[field].dropna()
        if len(x)<=min_period:
            return np.nan
        return talib.LINEARREG_SLOPE(x, timeperiod=len(x)).iloc[-1] / x.mean()

    def sub_Investment_Quality(semi_yr):
        """计算一个样本点，半年度频率"""
        basic = get_basic(codes=codes, count=21, end_date=semi_yr, fields=['total_assets', 'paidin_capital'])
        # 计算必须披露的时间
        basic['discDate'] = basic['statDate'].apply(__discDate)
        basic = basic.query('pubDate<discDate').drop(columns='pubDate')\
            .rename(columns={'discDate':'pubDate'})\
            .sort_values(by=['code', 'statDate'])
        Total_Assets_Growth_Rate = - tmp.groupby('code')\
            .apply(__t_reg, field='total_assets', min_period=6, include_groups=False).fillna(0.)
        Issuance_growth = - tmp.groupby('code')\
            .apply(__t_reg, field='paidin_capital', min_period=6, include_groups=False).fillna(0.)
        
        def __get_captial_expenditure(yr):
            # 改过
            tmp_all = get_basic(codes=codes, 
                fields=[
                'longterm_account_payable', # 长期应收款
                'specific_account_payable', # 专项应付款
                'fix_intan_other_asset_acqui_cash', # 购建固定资产、无形资产和其他长期资产支付的现金
                ])
            tmp_all = tmp_all[(tmp_all['statDate'].dt.year==yr) & (tmp_all['statDate'].dt.month==12)]
            tmp = tmp_all[['code', 'pubDate', 'statDate', 'longterm_account_payable', 'specific_account_payable']]
            tmp1 = tmp_all[['code', 'pubDate', 'statDate', 'fix_intan_other_asset_acqui_cash']]

            tmp_field = ['longterm_account_payable', 'specific_account_payable']
            tmp.loc[:, tmp_field] = tmp.loc[:, tmp_field].astype(float).fillna(0.)
            tmp['长期无息负债'] = tmp[tmp_field].sum(axis=1)
            delta_L = tmp.groupby('code')['长期无息负债']\
                .apply(lambda x: x.iloc[-1] - x.iloc[0]).rename('长期无息负债增量').reset_index()  # 计算年内增量 = 年末 - 年初
            tmp1 = tmp1.merge(delta_L)
            # 计算指定年度的 Capital_expenditure：用当年年末的“购建固定资产、无形资产和其他长期资产支付的现金”减去“长期无息负债的增量”（年内末 - 年内初）
            tmp1['Capital_expenditure'] = tmp1.eval('fix_intan_other_asset_acqui_cash - 长期无息负债增量')
            return tmp1[['code', 'statDate', 'pubDate', 'Capital_expenditure']]
        tmp = []
        for yr in range(semi_yr.year-5, semi_yr.year):
            tmp.append(__get_captial_expenditure(yr))
        tmp = pd.concat(tmp).reset_index(drop=True)
        Capital_expenditure_growth = - tmp.groupby('code')\
            .apply(__t_reg, field='Capital_expenditure', min_period=2, include_groups=False).fillna(0.)
        sub_factor = pd.concat([Total_Assets_Growth_Rate, Issuance_growth, Capital_expenditure_growth], axis=1)
        sub_factor.columns = ['Total_Assets_Growth_Rate', 'Issuance_growth', 'Capital_expenditure_growth']
        return sub_factor
    semi_yrs = [pd.Timestamp(s.year-1, 9, 1)]
    for y in range(s.year, end_date.year+1):
        q4 = pd.Timestamp(y, 5, 1)
        q2 = pd.Timestamp(y, 9, 1)
        if q4 > s:
            if q4 < end_date:
                semi_yrs.append(q4)
        else:
            semi_yrs.clear()
            semi_yrs.append(q4)
        if q2 < end_date:
            semi_yrs.append(q2)
    factor = []
    for semi_yr in semi_yrs:
        sub_factor = sub_Investment_Quality(semi_yr)
        sub_factor['pubDate'] = semi_yr
        factor.append(sub_factor.reset_index())
    factor = pubDate_align_tradedate(pd.concat(factor))
    factor = factor.loc[
        (factor['time']>=pd.to_datetime(s))&(factor['time']<=pd.to_datetime(end_date)), 
        ['code', 'time', 'Total_Assets_Growth_Rate', 'Issuance_growth', 'Capital_expenditure_growth']]
    return factor.reset_index(drop=True).sort_values(['code', 'time'], ignore_index=True)


def cal_Quality(codes=None, start_date=None, end_date=None, count=250):
    print('正在计算Quality因子')
    Leverage = cal_Leverage(codes=codes, start_date=start_date, end_date=end_date, count=count)
    Earnings_Variability = cal_Earnings_Variability(codes=codes, start_date=start_date, end_date=end_date, count=count)
    Earnings_Quality = cal_Earnings_Quality(codes=codes, start_date=start_date, end_date=end_date, count=count)
    Profitability = cal_Profitability(codes=codes, start_date=start_date, end_date=end_date, count=count)
    Investment_Quality = cal_Investment_Quality(codes=codes, start_date=start_date, end_date=end_date, count=count)
    factor = pd.concat([Leverage, Earnings_Variability, Earnings_Quality, Profitability, Investment_Quality], axis=1)
    factor = factor.loc[:, ~factor.columns.duplicated(keep='first')]
    print('完成计算Quality因子\n')
    return factor


# Value（一级因子）：估值
# BTOP（二级因子）：Book to price（三级因子）：将最近报告期的普通股账面价值除以当前市值
# Earnings Yield（二级因子）
# Trailing Earnings to price Ratio（三级因子）：过去12个月的盈利除以当前市值
# Analyst Predicted Earnings to Price（三级因子）：分析师预测EP比。预测12个月的盈利除以当前市值
# Cash earnings to price（三级因子）：过去12个月的现金盈利除以当前市值
# Enterprise multiple（三级因子）：上一财政年度的息税前利润（EBIT）除以当前企业价值（EV）
# Long Term reversal（二级因子）
# Long term relative strength（三级因子）：长期相对强度。滞后 273 个交易日，在 11 个交易日的时间窗口内取非滞后的长期相对强度值的等权平均值，最后取相反数。其中非滞后的长期相对强度为以时间窗口 1040 个交易日计算股票对数收益率加权之和，半衰期 260 个交易日。
# Long term historical alpha（三级因子）：长期历史Alpha。滞后 273 个交易日，在 11 个交易日的时间窗口内取非滞后长期历史 Alpha 值的等权平均值，最后取相反数。其中非滞后的长期历史Alpha 时间窗口为 1040 个交易日。

def cal_Value(codes=None, start_date=None, end_date=None, count=250):
    print('正在计算Value因子')
    s, end_date = __start_end_date__(start_date=start_date, end_date=end_date, count=count)
    s, end_date = pd.to_datetime(s), pd.to_datetime(end_date)

    val = get_valuation(
        codes=codes, start_date=start_date, end_date=end_date, count=count, 
        fields=['pb_ratio', 'pe_ratio', 'pcf_ratio'])
    val['Book_to_price'] = 1 / val['pb_ratio']
    val['Earning_to_price'] = 1 / val['pe_ratio']
    val['Cash_earning_to_price'] = 1 / val['pcf_ratio']
    factor = val[['code', 'time', 'Book_to_price', 'Earning_to_price', 'Cash_earning_to_price']]

    def __cummean(x):
        f_ = []
        def __sub_cummean(y, f_):
            f_ += y['np'].values.tolist()
            if len(f_)<5:
                return np.nan
            return np.nanmean(f_)
        np_std = x.groupby('time').apply(lambda z: __sub_cummean(z, f_), include_groups=False)
        np_std.name = 'np_mean'
        return np_std.dropna()

    forecast_EP_mean = []
    for year in track(range(pd.to_datetime(s).year, pd.to_datetime(end_date).year+1), description='正在计算forecast_EP_mean...'):
        forecast_np = get_report(end_date=pd.to_datetime(f'{year}-12-31'), count=365*3, year=year, fields=['np'])
        forecast_np['np'] /= 10000
        np_mean = forecast_np.groupby('code').apply(__cummean, include_groups=False).reset_index().rename(columns={'time': 'pubDate'})
        np_mean = pubDate_align_tradedate(np_mean, end_date=pd.to_datetime(f'{year}-12-31'))
        np_mean = np_mean[np_mean['time'] >= pd.Timestamp(f'{year}-01-01')].reset_index(drop=True)
        val = get_valuation(codes=codes, start_date=pd.to_datetime(f'{year}-01-01', format='%Y-%m-%d'), end_date=pd.to_datetime(f'{year}-12-31', format='%Y-%m-%d'), fields=['market_cap'])
        f_EP_mean = pd.merge(np_mean, val)
        f_EP_mean['forecast_EP_mean'] = f_EP_mean.eval('np_mean/market_cap')
        forecast_EP_mean.append(f_EP_mean)
    forecast_EP_mean = pd.concat(forecast_EP_mean)
    forecast_EP_mean = forecast_EP_mean[(forecast_EP_mean['time']<=pd.to_datetime(end_date))&(forecast_EP_mean['time']>=pd.to_datetime(s))].reset_index(drop=True)
    factor = factor.merge(forecast_EP_mean[['code', 'time', 'forecast_EP_mean']], how='left')

    basic = get_basic(
        codes=codes, start_date=s - pd.Timedelta(days=365), end_date=end_date, 
        ttm_dict={'operating_revenue': 'sum', 'operating_cost': 'sum'})
    basic['EBIT'] = basic['operating_revenue(TTM)'] - basic['operating_cost(TTM)']
    # 计算必须披露的时间
    basic['discDate'] = basic['statDate'].apply(__discDate)
    basic = basic.query('pubDate<discDate').drop(columns='pubDate')\
        .rename(columns={'discDate':'pubDate'})\
        .sort_values(by=['code', 'statDate'])
    basic = pubDate_align_tradedate(basic)
    tmp = get_valuation(codes=codes, start_date=s, end_date=end_date, count=count, fields=['market_cap'])
    tmp = basic.merge(tmp)
    tmp['Enterprise_multiple'] = tmp.eval('EBIT/market_cap').div(1e8)
    factor = factor.merge(tmp[['code', 'time', 'Enterprise_multiple']], how='outer')

    window = 750
    half_life = 260

    # 长期相对强度
    s3, _ = __start_end_date__(start_date=None, end_date=s, count=window+12)
    W = get_exponent_weight(window=window, half_life=260)
    if codes is None:
        codes = 'all-stock'
    price = get_price(type='stock', codes=codes, start_date=s3, end_date=end_date, fields=['pre_close', 'close'])
    price['ret'] = np.log(price['close']) - np.log(price['pre_close'])
    ret = pd.pivot_table(price, index='time', columns='code', values='ret')

    Relative_strength = {}
    for i in track(range(len(ret) - window-1), description='正在计算长期非滞后相对强度……'):
        tmp = ret.iloc[i:i+window, :]
        tmp = tmp.loc[:, tmp.isnull().sum(axis=0) / window <= 0.1].fillna(0.)
        np.sum(W.reshape(-1, 1)*tmp.values, axis=0)
        Relative_strength[tmp.index[-1]] = pd.Series(np.sum(W.reshape(-1, 1)*tmp.values, axis=0), index=tmp.columns)
    Relative_strength = pd.DataFrame(Relative_strength).T
    Relative_strength.index.name = 'time'
    Relative_strength = Relative_strength.rolling(11).mean().dropna(how='all').mul(-1)
    Relative_strength = pd.melt(Relative_strength.reset_index(), id_vars='time', value_name='Longterm_Relative_strength')\
        .dropna().reset_index(drop=True)

    # 计算长期Alpha
    price = get_price(type='stock', codes=codes, start_date=s3, end_date=end_date, count=count, fields=['pre_close', 'close'])
    hs300 = get_price(type='index', codes=['399300.SZ'], start_date=s3, end_date=end_date, fields=['pre_close', 'close'])
    price = pd.concat([price, hs300]).reset_index(drop=True)
    price['ret'] = price['close'] / price['pre_close'] - 1
    ret = pd.pivot_table(price, values='ret', index='time', columns='code')

    W = get_exponent_weight(window=window, half_life=half_life)

    def __cal_Alpha(tmp):
        W_full = np.diag(W)
        Y_full = tmp.dropna(axis=1).drop(columns='399300.SZ')
        idx_full, Y_full = Y_full.columns, Y_full.values
        X_full = np.c_[np.ones((window, 1)), tmp.loc[:, '399300.SZ'].values]
        beta_full = np.linalg.pinv(X_full.T@W_full@X_full)@X_full.T@W_full@Y_full
        
        alpha_full = pd.Series(beta_full[0], index=idx_full, name=tmp.index[-1])

        alpha_lack = {}
        # 不具有完整数据的股票
        for c in set(tmp.columns) - set(idx_full) - set('399300.SZ'):
            tmp_ = tmp.loc[:, [c, '399300.SZ']].copy()
            tmp_.loc[:, 'W'] = W
            tmp_ = tmp_.dropna()
            W_lack = np.diag(tmp_['W'])
            if len(tmp_) < half_life:
                continue
            X_lack = np.c_[np.ones(len(tmp_)), tmp_['399300.SZ'].values]
            Y_lack = tmp_[c].values
            beta_tmp = np.linalg.pinv(X_lack.T@W_lack@X_lack)@X_lack.T@W_lack@Y_lack
            alpha_lack[c] = beta_tmp[0]
        alpha_lack = pd.Series(alpha_lack, name=tmp.index[-1])
        return pd.concat([alpha_full, alpha_lack]).sort_index()

    Alpha = Parallel(8)(delayed(__cal_Alpha)(
        ret.iloc[i:i+window, :].copy()) for i in 
        track(range(len(ret)-window+1), description='正在计算alpha...'))

    Alpha = pd.concat(Alpha, axis=1).T
    Alpha['399300.SZ'] = Alpha['399300.SZ'].apply(lambda x: x[0])
    Alpha = Alpha.rolling(11).mean().dropna(how='all').mul(-1)
    Alpha = pd.melt(Alpha.reset_index(), id_vars='index').dropna()
    Alpha.columns = ['time', 'code', 'Longterm_Alpha']

    factor = factor.merge(Relative_strength, how='left').merge(Alpha, how='left')
    print('完成计算Value因子\n')
    return factor


# Growth（一级因子）：成长
# Predicted growth 3year（三级因子）：分析师预测长期盈利增长率。分析师预测的长期（3-5）年利润增长率。
# Historical earnings per share growth rate（三级因子）：近5个财年的每股收益平均增长率。过去5个财政年度的每股收益对时间回归的斜率除以平均每股年收益
# Historical sales per share growth rate（三级因子）：近5个财政年度的每股年营业收入平均增长率。过去5个财政年度的每股年营业收入对时间回归斜率除以平均每股年营业收入

def cal_Growth(codes=None, start_date=None, end_date=None, count=250):
    print('正在计算Growth因子')
    s, end_date = __start_end_date__(start_date=start_date, end_date=end_date, count=count)
    s, end_date = pd.to_datetime(s), pd.to_datetime(end_date)
    def __cummean(x):
        f_ = []
        def __sub_cummean(y, f_):
            f_ += y['roe'].values.tolist()
            if len(f_)<5:
                return np.nan
            return np.nanmean(f_)
        np_std = x.groupby('time').apply(lambda z: __sub_cummean(z, f_), include_groups=False)
        np_std.name = 'roe_mean'
        return np_std.dropna()

    forecast_roe_mean = []
    for year in track(range(pd.to_datetime(s).year, pd.to_datetime(end_date).year+1), description='正在计算forecast_roe_mean...'):
        forecast_roe = get_report(end_date=pd.to_datetime(f'{year}-12-31'), count=365*3, year=year+2, fields=['roe'])
        roe_mean = forecast_roe.groupby('code').apply(__cummean, include_groups=False).reset_index().rename(columns={'time': 'pubDate'})
        roe_mean = pubDate_align_tradedate(roe_mean, end_date=pd.to_datetime(f'{year}-12-31'))
        roe_mean = roe_mean[roe_mean['time'] >= pd.Timestamp(f'{year}-01-01')].reset_index(drop=True)
        forecast_roe_mean.append(roe_mean)
    forecast_roe_mean = pd.concat(forecast_roe_mean)
    forecast_roe_mean = forecast_roe_mean[(forecast_roe_mean['time']<=pd.to_datetime(end_date))&(forecast_roe_mean['time']>=pd.to_datetime(s))].reset_index(drop=True)
    
    def __t_reg(x, field, min_period):
        """时间序列回归斜率除以平均值"""
        x = x[field].dropna()
        if len(x)<=min_period:
            return np.nan
        return talib.LINEARREG_SLOPE(x, timeperiod=len(x)).iloc[-1] / x.mean()

    def sub_growth(semi_yr):
        basic = get_basic(
            count=21, end_date=semi_yr, 
            ttm_dict={'np_parent_company_owners': 'sum', 'operating_revenue': 'sum'})

        # 计算必须披露的时间
        basic['discDate'] = basic['statDate'].apply(__discDate)
        basic = basic.query('pubDate<discDate').drop(columns='pubDate')\
            .rename(columns={'discDate':'pubDate'})\
            .sort_values(by=['code', 'statDate'])
        Earning_Growth_Rate = - basic.groupby('code')\
            .apply(__t_reg, field='np_parent_company_owners(TTM)', min_period=8, include_groups=False)
        OP_Growth_Rate = - basic.groupby('code')\
            .apply(__t_reg, field='operating_revenue(TTM)', min_period=8, include_groups=False)
        sub_factor = pd.concat([Earning_Growth_Rate, OP_Growth_Rate], axis=1)
        sub_factor.columns = ['Earning_Growth_Rate', 'OP_Growth_Rate']
        return sub_factor

    semi_yrs = [pd.Timestamp(s.year-1, 9, 1)]
    for y in range(s.year, end_date.year+1):
        q4 = pd.Timestamp(y, 5, 1)
        q2 = pd.Timestamp(y, 9, 1)
        if q4 > s:
            if q4 < end_date:
                semi_yrs.append(q4)
        else:
            semi_yrs.clear()
            semi_yrs.append(q4)
        if q2 < end_date:
            semi_yrs.append(q2)
    factor = []
    for semi_yr in semi_yrs:
        sub_factor = sub_growth(semi_yr)
        sub_factor['pubDate'] = semi_yr
        factor.append(sub_factor.reset_index())
    factor = pubDate_align_tradedate(pd.concat(factor))
    factor = factor.loc[
        (factor['time']>=pd.to_datetime(s))&(factor['time']<=pd.to_datetime(end_date)), 
        ['code', 'time', 'Earning_Growth_Rate', 'OP_Growth_Rate']]
    factor = factor.merge(forecast_roe_mean[['code', 'time', 'roe_mean']], how='left')
    print('完成计算Growth因子\n')
    return factor


# Sentiment（一级因子）：情绪
# Revision ratio（三级因子）：调整比率。分析师调整比率的每月变动，定义为向上调整次数减去向下调整次数，除以总的调整次数。
# 对于第一个因子，我的数据库中缺乏【上调】、【下调】字段，因此在此不进行计算。
# Change in analyst-predicted earnings-to-price（三级因子）：分析师预测EP比变化
# Change in analyst-predicted earnings per share（三级因子）：分析师预测每股收益变化

def cal_Sentiment(codes=None, start_date=None, end_date=None, count=250):
    print('正在计算Sentiment因子')
    s, end_date = __start_end_date__(start_date=start_date, end_date=end_date, count=count)
    s, end_date = pd.to_datetime(s), pd.to_datetime(end_date)
    def __cummean(x):
        f_ = []
        def __sub_cummean(y, f_):
            f_ += y['np'].values.tolist()
            if len(f_)<5:
                return np.nan
            return np.nanmean(f_)
        np_std = x.groupby('time').apply(lambda z: __sub_cummean(z, f_), include_groups=False)
        np_std.name = 'np_mean'
        return np_std.dropna()

    forecast_EP_mean = []
    for year in track(range(pd.to_datetime(s).year-2, pd.to_datetime(end_date).year+1), description='正在计算Pred_EP_chg...'):
        forecast_np = get_report(end_date=pd.to_datetime(f'{year}-12-31'), count=365*3, year=year, fields=['np'])
        forecast_np['np'] /= 10000
        np_mean = forecast_np.groupby('code').apply(__cummean, include_groups=False).reset_index().rename(columns={'time': 'pubDate'})
        np_mean['pubDate'] = pd.to_datetime(np_mean['pubDate'])
        np_mean = pubDate_align_tradedate(np_mean, end_date=pd.to_datetime(f'{year}-12-31'))
        np_mean = np_mean[np_mean['time'] >= pd.Timestamp(f'{year}-01-01')].reset_index(drop=True)
        forecast_EP_mean.append(np_mean)
    forecast_EP_mean = pd.concat(forecast_EP_mean)
    f_ep_chg = forecast_EP_mean\
        .pivot_table(index='time', columns='code', values='forecast_EP_mean')\
        .pct_change(periods=63).dropna(how='all')
    f_ep_chg = sum([f_ep_chg.shift(i*63).div(i+1).fillna(0.) for i in range(4)]).dropna(how='all')
    f_ep_chg = pd.melt(f_ep_chg.reset_index(), id_vars='time', value_name='Pred_EP_chg')
    
    forecast_np_mean = []
    for year in track(range(pd.to_datetime(s).year-2, pd.to_datetime(end_date).year+1), description='正在计算Pred_EPS_chg...'):
        forecast_np = get_report(end_date=pd.to_datetime(f'{year}-12-31'), count=365*3, year=year, fields=['np'])
        forecast_np['np'] /= 10000
        np_mean = forecast_np.groupby('code').apply(__cummean, include_groups=False).reset_index().rename(columns={'time': 'pubDate'})
        np_mean['pubDate'] = pd.to_datetime(np_mean['pubDate'])
        np_mean = pubDate_align_tradedate(np_mean, end_date=pd.to_datetime(f'{year}-12-31'))
        np_mean = np_mean[np_mean['time'] >= pd.Timestamp(f'{year}-01-01')].reset_index(drop=True)
        forecast_np_mean.append(np_mean)
    forecast_np_mean = pd.concat(forecast_np_mean)
    f_np_chg = forecast_np_mean\
        .pivot_table(index='time', columns='code', values='np_mean')\
        .pct_change(periods=63).dropna(how='all')
    f_np_chg = sum([f_np_chg.shift(i*63).div(i+1).fillna(0.) for i in range(4)]).dropna(how='all')
    f_np_chg = pd.melt(f_np_chg.reset_index(), id_vars='time', value_name='Pred_EPS_chg')
    factor = pd.merge(f_ep_chg, f_np_chg, how='outer')
    cond1 = factor['time']>=pd.to_datetime(s)
    cond2 = factor['time']<=pd.to_datetime(end_date)
    cond3 = factor['code'].apply(lambda x: x.startswith(('0', '6', '3')))
    factor = factor.loc[cond1&cond2&cond3, :]
    print('完成计算Sentiment因子\n')
    return factor.reset_index(drop=True)


# Dividend Yield（一级因子）：分红
# Dividend to price ratio（三级因子）：近1年的每股股息与上月末股价的比值
# Analyst predicted dividend to price ratio（三级因子）：分析师预测1年的每股股息与当前价格的比值

def cal_Dividend(codes=None, start_date=None, end_date=None, count=250):
    print('正在计算Dividend因子')
    s, end_date = __start_end_date__(start_date=start_date, end_date=end_date, count=count)
    s, end_date = pd.to_datetime(s), pd.to_datetime(end_date)

    val = get_valuation(
        codes=codes, start_date=start_date, end_date=end_date, count=count, 
        fields=['dv_ratio'])
    factor = val[['code', 'time', 'dv_ratio']]

    def __cummean(x):
            f_ = []
            def __sub_cummean(y, f_):
                f_ += y['rd'].values.tolist()
                if len(f_)<5:
                    return np.nan
                return np.nanmean(f_)
            np_std = x.groupby('time').apply(lambda z: __sub_cummean(z, f_), include_groups=False)
            np_std.name = 'rd_mean'
            return np_std.dropna()
    DTOPF = []
    for year in track(range(pd.to_datetime(s).year, pd.to_datetime(end_date).year+1), description='正在计算Forecast_Dividend_to_Price...'):
        forecast_rd = get_report(end_date=pd.to_datetime(f'{year}-12-31'), count=365*3, year=year, fields=['rd'])
        rd_mean = forecast_rd.groupby('code').apply(__cummean, include_groups=False).reset_index().rename(columns={'time': 'pubDate'})
        rd_mean = pubDate_align_tradedate(rd_mean, end_date=pd.to_datetime(f'{year}-12-31'))
        rd_mean = rd_mean[rd_mean['time'] >= pd.Timestamp(f'{year}-01-01')].reset_index(drop=True)
        DTOPF.append(rd_mean)
    DTOPF = pd.concat(DTOPF)
    DTOPF = DTOPF[(DTOPF['time']<=pd.to_datetime(end_date))&(DTOPF['time']>=pd.to_datetime(s))].reset_index(drop=True)
    factor = factor.merge(DTOPF[['code', 'time', 'rd_mean']], how='left')
    factor = factor.rename(columns={'dv_ratio': 'Dividend_to_Price', 'rd_mean': 'Forecast_Dividend_to_Price'})
    print('完成计算Dividend因子\n')
    return factor


# 在全局交易日列表 all_dates 中，找到从给定 end_date 开始向后第 shift 个交易日并返回该日期。
def next_trade_date(end_date, shift):
    start_move = 0
    for d in all_dates:
        if d == end_date:
            start_move = 1
        if start_move>0:
            # 当计数等于 shift+1 时返回当前日期 d，即找到了向后移动 shift 步的交易日
            if start_move==shift+1:
                return d
            start_move+=1
    else:
        return end_date + pd.Timedelta(days=shift)
   
    
def get_forward_return(n=3, start_date=None, end_date=None, count=60, is_fill=True, cost_field='open', close_price='close'):
    """
    计算股票未来 n 天的算术收益
    n: 向后挪的天数，默认3天。
    start_date/end_date/count: 开始、结束、交易日数量.
    is_fill: 是否向后补齐缺失数据。默认补齐。
    cost_field: 成本价计算字段 ['open', 'close', 'pre_close']
    close_price: 平仓价格计算字段，默认收盘价平仓。
    """
    start_date, end_date = __start_end_date__(start_date=start_date, end_date=end_date, count=count)
    if is_fill:
        end_date = next_trade_date(end_date, shift=n)  # 延长结束日（结束冗余）以保证有足够的后向价格
    end_date = end_date if pd.to_datetime(end_date)<=pd.to_datetime(global_end_date) else global_end_date
    data = get_price(type='stock', start_date=start_date, end_date=end_date, fields=['open', 'close', 'pre_close'])
    data['forward_open'] = data.groupby('code', as_index=False)[cost_field].shift(-1)
    data['forward_close'] = data.groupby('code', as_index=False)[close_price].shift(-n)
    data['forward_return'] = data.eval('forward_close/forward_open - 1')
    return data[['code', 'time', 'forward_return']].dropna()

def get_factor_ic(factor, n=5, ic_type='spearman', end_date=None, forward_return_kargs={}):
    """
    - factor: pd.DataFrame
        - index: [code, time]
        - columns: [factor_name]
        - values: 因子暴露值
    - n: int
        - 向后n天的ic值
    - ic_type: str
        - 可选范围：[pearson, kendall, spearman]
    """
    def __ic(x):
        with np.errstate(divide='ignore'):
            r = x['forward_return']
            f = x.drop(columns='forward_return')
            return f.corrwith(r, method=ic_type)
    factor = factor.copy()
    tmp = factor.reset_index()
    start_date, end_date_ = tmp['time'].min(), tmp['time'].max()
    if end_date is None:
        end_date = end_date_
    ret = get_forward_return(n=n, start_date=start_date, end_date=end_date, **forward_return_kargs)
    # 改过
    '''
    原先注释掉的 df = pd.concat([factor, ret], axis=1) 将按列拼接因子和收益，但需要两者对齐索引
    现在改为 df = pd.merge(factor, ret, on=['code','time'], how='inner')，更明确地在 code,time 上做内连接，保证只有同时存在因子值与收益的行被保留，避免索引错位或 NaN 列。
    '''
    # df = pd.concat([factor, ret], axis=1)
    df = pd.merge(factor, ret, on=['code', 'time'], how='inner')
    # 改过
    df.drop(columns=['code'], inplace=True)
    res = pd.melt(df.groupby('time').apply(__ic, include_groups=False).reset_index(), id_vars='time').dropna()
    res.columns = ['time', 'factor_name', 'ic']
    factor_names = res['factor_name'].unique()
    res = res.pivot_table(index='time', columns='factor_name', values='ic').reindex(columns=factor_names) # 时间为行，各因子IC为列
    return res.sort_values(by=['time'])

def calculate_icir(ic):
    """
    计算每个因子的ICIR值
    """
    # 计算每个因子的统计指标
    ic_stats = ic.agg([
        'mean',       # mean(IC): 多个时间截面上ic的平均值
        'std',        # std(ic): IC的波动性(标准差)
        'count',      # IC值的数量
        'min',        # IC最小值
        'max'         # IC最大值
        ]).T
    # 计算ICIR = mean(IC) / std(IC)
    ic_stats['icir'] = ic_stats['mean'] / ic_stats['std']
    # 处理可能出现的除零错误（当std=0时）
    ic_stats['icir'] = ic_stats['icir'].replace([np.inf, -np.inf], np.nan)
    # 按ICIR绝对值排序（通常我们关注绝对值大的因子）
    ic_stats = ic_stats.sort_values('icir', key=abs, ascending=False)
    ic_stats = ic_stats[['icir', 'mean(IC)']]
    return ic_stats


# 纯因子收益率求解
def Pure_Factor_Returns(t, Style):
    """
    t: pd.Timestamp, time
    Style: pd.DataFrame, Style factor explosure
    """
    # 风格因子暴露，已经做了标准化
    Style = Style.droplevel(1).sort_index()
    # 行业哑变量矩阵，采用申万一级行业分类
    ind = get_industry(date=t)
    # ind['cons'] = 1
    ind = ind.assign(cons=1)
    Industry = ind.pivot_table(index='code', columns='L1', values='cons').fillna(0.)
    Industry, Style = Industry.align(Style, join='inner', axis=0)
    # 国家因子暴露，全为1
    Country = pd.Series(1., index=Style.index, name='Country')
    # 合并
    X = pd.concat([Country, Industry, Style], axis=1)
    # 异方差调整，WLS权重计算
    val = get_valuation(end_date=t, count=1, fields=['circulating_market_cap'])\
        .set_index('code')['circulating_market_cap']
    V = val.align(X, axis=0, join='right')[0].fillna(0.)
    V = pd.DataFrame(np.diag(np.sqrt(V) / np.sum(np.sqrt(V))), index=V.index, columns=V.index)
    # ind['Size'] = ind['code'].map(val)
    ind = ind.assign(Size=ind['code'].map(val))
    industry_weights = ind.groupby('L1')['Size'].sum() / ind['Size'].sum()
    industry_weights = industry_weights[Industry.columns]
    # 约束矩阵R计算
    k = len(X.columns)
    diag_R = np.diag(np.ones(k))
    location = len(industry_weights)
    R = np.delete(diag_R, location, axis=1)
    adj_industry_weights = -industry_weights.div(industry_weights.iloc[-1]).iloc[:-1]
    R[location, 1:location] = adj_industry_weights.values
    # 因子权重计算
    # print(f'{type(R)=}') #numpy.ndarray
    # print(f'{type(X)=}') #pandas.core.frame.DataFrame
    # print(f'{type(V)=}') #pandas.core.frame.DataFrame
    W = R@np.linalg.pinv(R.T@X.T@V@X@R)@R.T@X.T@V
    W.index = X.columns
    # 纯因子收益率计算
    price = get_price(type='stock', start_date=t+pd.Timedelta(days=1), count=1, fields=['pre_close', 'close'])
    price['r'] = price.eval('close/pre_close - 1')
    r = price.set_index('code')['r']
    r = r.align(X, join='right')[0].fillna(0.)
    factor_return = W.dot(r).to_frame(name=t).T
    # 纯因子对数收益率计算
    price['log_r'] = np.log(price['close'] / price['pre_close'])  # ln
    log_r = price.set_index('code')['log_r']
    log_r = log_r.align(X, join='right')[0].fillna(0.)
    log_factor_return = W.dot(log_r).to_frame(name=t).T
    return factor_return, log_factor_return

def cal_all_factors(
    codes,
    start_date,
    end_date,
    count
    ):
    if count==None:
        count = len(get_trade_date(start_date=start_date, end_date=end_date))

    if not os.path.exists('result/Size.csv'):
        Size_df = cal_Size(
            codes=codes,
            start_date=start_date,
            end_date=end_date,
            count=count
        )
        Size_df.drop_duplicates(subset=['code', 'time'], keep="last", inplace=True)
        Size_df_clean = clean_BARRA(Size_df)
        save_df(Size_df_clean, 'result/Size.csv')
    Size_df_clean = read_df('result/Size.csv')

    if not os.path.exists('result/Volatility.csv'):
        Volatility_df = cal_Volatility(
            codes=codes,
            start_date=start_date,
            end_date=end_date,
            count=count,
            window=252,
            half_life=63,
        )
        Volatility_df.drop_duplicates(subset=['code', 'time'], keep="last", inplace=True)
        Volatility_df_clean = clean_BARRA(Volatility_df)
        save_df(Volatility_df_clean, 'result/Volatility.csv')
    Volatility_df_clean = read_df('result/Volatility.csv')

    if not os.path.exists('result/Liquidity.csv'):
        Liquidity_df = cal_Liquidity(
            codes=codes,
            start_date=start_date,
            end_date=end_date,
            count=count
        )
        Liquidity_df.drop_duplicates(subset=['code', 'time'], keep="last", inplace=True)
        Liquidity_df_clean = clean_BARRA(Liquidity_df)
        save_df(Liquidity_df_clean, 'result/Liquidity.csv')
    Liquidity_df_clean = read_df('result/Liquidity.csv')

    if not os.path.exists('result/Momentum.csv'):
        Momentum_df = cal_Momentum(
            codes=codes,
            start_date=start_date,
            end_date=end_date,
            count=count
        )
        Momentum_df.drop_duplicates(subset=['code', 'time'], keep="last", inplace=True)
        Momentum_df_clean = clean_BARRA(Momentum_df)
        save_df(Momentum_df_clean, 'result/Momentum.csv')
    Momentum_df_clean = read_df('result/Momentum.csv')

    if not os.path.exists('result/Quality.csv'):
        Quality_df = cal_Quality(
            codes=codes,
            start_date=start_date,
            end_date=end_date,
            count=count
        )
        Quality_df.drop_duplicates(subset=['code', 'time'], keep="last", inplace=True)
        Quality_df_clean = clean_BARRA(Quality_df)
        save_df(Quality_df_clean, 'result/Quality.csv')
    Quality_df_clean = read_df('result/Quality.csv')

    if not os.path.exists('result/Value.csv'):
        Value_df = cal_Value(
            codes=codes,
            start_date=start_date,
            end_date=end_date,
            count=count
        )
        Value_df.drop_duplicates(subset=['code', 'time'], keep="last", inplace=True)
        Value_df_clean = clean_BARRA(Value_df)
        save_df(Value_df_clean, 'result/Value.csv')
    Value_df_clean = read_df('result/Value.csv')

    if not os.path.exists('result/Growth.csv'):
        Growth_df = cal_Growth(
            codes=codes,
            start_date=start_date,
            end_date=end_date,
            count=count
        )
        Growth_df.drop_duplicates(subset=['code', 'time'], keep="last", inplace=True)
        Growth_df_clean = clean_BARRA(Growth_df)
        save_df(Growth_df_clean, 'result/Growth.csv')
    Growth_df_clean = read_df('result/Growth.csv')

    if not os.path.exists('result/Sentiment.csv'):
        Sentiment_df = cal_Sentiment(
            codes=codes,
            start_date=start_date,
            end_date=end_date,
            count=count
        )
        Sentiment_df.drop_duplicates(subset=['code', 'time'], keep="last", inplace=True)
        Sentiment_df_clean = clean_BARRA(Sentiment_df)
        save_df(Sentiment_df_clean, 'result/Sentiment.csv')
    Sentiment_df_clean = read_df('result/Sentiment.csv')

    if not os.path.exists('result/Dividend.csv'):
        Dividend_df = cal_Dividend(
            codes=codes,
            start_date=start_date,
            end_date=end_date,
            count=count
        )
        Dividend_df.drop_duplicates(subset=['code', 'time'], keep="last", inplace=True)
        Dividend_df_clean = clean_BARRA(Dividend_df)
        save_df(Dividend_df_clean, 'result/Dividend.csv')
    Dividend_df_clean = read_df('result/Dividend.csv')

    all_third_factors = pd.concat([Size_df_clean, Volatility_df_clean,
                                    Liquidity_df_clean, Momentum_df_clean, Quality_df_clean, Value_df_clean, Growth_df_clean, Sentiment_df_clean, Dividend_df_clean
                                   ], axis=1)
    all_third_factors = all_third_factors.loc[:, ~all_third_factors.columns.duplicated(keep='first')]
    all_third_factors['time'] = pd.to_datetime(all_third_factors['time'])
    save_df(all_third_factors, 'result/全部三级因子.csv')

    # 因子合成
    # 最终选择等权合成，注释掉的部分是IC加权合成
    print('正在合成一级因子')
    first_cne6_names = ['Size', 'Volatility', 'Liquidity', 'Momentum', 'Quality', 'Value', 'Growth', 'Sentiment', 'Dividend']
    barra_list = []
    for index, f in enumerate([Size_df_clean, Volatility_df_clean,
                                Liquidity_df_clean, Momentum_df_clean, Quality_df_clean, Value_df_clean, Growth_df_clean, Sentiment_df_clean, Dividend_df_clean
                                ]):
        # 下面注释掉的4行是IC加权合成
        # ic = get_factor_ic(f, n=1)
        # w = ic.pivot_table(index='time', columns='factor_name', values='ic')\
        #     .shift(1).ewm(halflife=250, min_periods=120).mean()
        # f = (f * w).mean(axis=1).dropna()
        f = f.set_index(['code', 'time'])
        # 三级因子的平均值为一级因子值
        f = f.mean(axis=1)
        f.name = first_cne6_names[index]
        barra_list.append(f)

    barra = pd.concat(barra_list, axis=1)
    barra = barra.groupby('time', group_keys=False).apply(lambda x: (x - x.mean()) / (x.std()+1e-6))
    barra_save = barra.reset_index()
    barra_save['time'] = pd.to_datetime(barra_save['time'])
    save_df(barra_save, 'result/一级因子.csv')
    barra = barra_save.set_index(['code', 'time'])
    print('完成合成一级因子\n')
    return all_third_factors, barra


if __name__ == '__main__':
    
    os.makedirs('result', exist_ok=True)

    # 所有可用股票
    all_ts_codes_df = pd.read_csv('已下载股票.csv')
    all_ts_codes = all_ts_codes_df['ts_code'].to_list()
    print(f'时间范围：{start_date} 到 {end_date}')

    import time
    startt = time.perf_counter()

    # 选择部分股票
    # selected_ts_codes = [i.rstrip('.csv') for i in os.listdir('daily')]
    selected_ts_codes = all_ts_codes
    print(f'共{len(selected_ts_codes)}只股票')

    # 读取daily_basic
    # get_valuation()
    print('正在读取valuation')
    daily_basic_path = 'daily_basic.csv'
    daily_basic_df_all = pd.read_csv(daily_basic_path)
    daily_basic_df_s = daily_basic_df_all[daily_basic_df_all['ts_code'].isin(selected_ts_codes)]
    daily_basic_df_s.rename(columns=name_exchange_anti, inplace=True)
    daily_basic_df_s['time'] = pd.to_datetime(daily_basic_df_s['time'])

    # 读取daily, 包括沪深300指数
    # get_price()
    print('正在读取price')
    daily_path = 'daily.csv'
    daily_df_all = pd.read_csv(daily_path)
    daily_df_s = daily_df_all[daily_df_all['ts_code'].isin(selected_ts_codes)]
    daily_df_s.rename(columns=name_exchange_anti, inplace=True)
    daily_df_s['time'] = pd.to_datetime(daily_df_s['time'].astype(str))
    SHSZ300_df = read_df('index_daily_399300.SZ.csv')
    SHSZ300_df.rename(columns=name_exchange_anti, inplace=True)
    SHSZ300_df['time'] = pd.to_datetime(SHSZ300_df['time'].astype(str))
    daily_df_s, SHSZ300_df

    # 读取industry
    # get_industry()
    print('正在读取industry')
    industry_path = r'all_industry.csv'
    industry_df_all = pd.read_csv(industry_path)
    industry_df_s = industry_df_all[industry_df_all['ts_code'].isin(selected_ts_codes)]
    industry_df_s.dropna(subset=['trade_date'], inplace=True)
    industry_df_s.rename(columns=name_exchange_anti, inplace=True)
    industry_df_s['time'] = pd.to_datetime(industry_df_s['time'])

    # 读取财务数据季报
    print('正在读取basic')
    combined_df_all = read_df('finance.csv')
    combined_df = combined_df_all[combined_df_all['ts_code'].isin(selected_ts_codes)]
    combined_df.rename(columns=name_exchange_anti, inplace=True)
    combined_df['statDate'] = pd.to_datetime(combined_df['statDate'])
    combined_df['pubDate'] = pd.to_datetime(combined_df['pubDate'])
    # 首先确保数据按ts_code和end_date排序
    combined_df.sort_values(['code', 'statDate'], inplace=True)

    # 读取report
    # get_report()
    print('正在读取report')
    report_rc_df_all_path = 'report_rc.csv'
    report_rc_df_all = pd.read_csv(report_rc_df_all_path)
    report_rc_df_s = report_rc_df_all[report_rc_df_all['ts_code'].isin(selected_ts_codes)]
    report_rc_df_s.rename(columns=name_exchange_anti, inplace=True)
    report_rc_df_s.rename(columns={'report_date': 'time'}, inplace=True)
    report_rc_df_s['time'] = pd.to_datetime(report_rc_df_s['time'].astype(str))

    all_third_factors, barra = cal_all_factors(
        codes=None,
        start_date=start_date,
        end_date=end_date,
        count=None
    )

    print('正在计算ic, icir')
    ic = get_factor_ic(
        factor=all_third_factors,
        n=5,
        ic_type='spearman',
        end_date=None,
        forward_return_kargs={},
        )
    save_df(ic.reset_index(), 'result/ic.csv')
    icir = calculate_icir(ic)
    save_df(icir.reset_index(), 'result/icir.csv')
    
    print('正在计算纯因子收益率')
    res = Parallel(8)(
        delayed(Pure_Factor_Returns)(t, Style)
        for t, Style in track(barra.groupby('time'), description='正在计算纯因子收益率')
    )

    res_1 = pd.concat([i[0] for i in res])
    res_1.index.name = "time"
    res_1 = res_1.reset_index()
    save_df(res_1, 'result/纯因子收益率.csv')
    log_res = pd.concat([i[1] for i in res])
    log_res.index.name = "time"
    log_res = log_res.reset_index()
    save_df(log_res, 'result/纯因子对数收益率.csv')
    
    print('完成计算纯因子收益率\n')
    endt = time.perf_counter()
    print(f'共耗时{time.strftime("%H时%M分%S秒", time.gmtime(endt - startt))}')
    
    factor_names = [
    'Size', 'Volatility', 'Liquidity', 
    'Momentum', 'Quality', 'Value', 
    'Growth', 'Sentiment', 'Dividend'
    ]
    # 绘制天数
    count = 250
    fig, axs = plt.subplots(3, 3, figsize=(24, 9))
    axs = axs.ravel()
    for i, ax in enumerate(axs):
        s = res_1[factor_names[i]].iloc[-count-1:]
        s.iloc[0] = 0.
        s = s.add(1).cumprod()
        ax.plot(s, color='k')
        ax.set_title(factor_names[i])
        ax.grid()
    fig.tight_layout()
    fig.savefig('photo/Pure_Factor_Returns.png')
    
    fig, axs = plt.subplots(3, 3, figsize=(24, 9))
    axs = axs.ravel()
    for i, ax in enumerate(axs):
        s = log_res[factor_names[i]].iloc[-count-1:]
        s.iloc[0] = 0.
        s = s.add(1).cumprod()
        ax.plot(s, color='k')
        ax.set_title(factor_names[i])
        ax.grid()
    fig.tight_layout()
    fig.savefig('photo/Pure_Factor_Returns_log.png')