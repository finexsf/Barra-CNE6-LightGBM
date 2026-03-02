import pandas as pd

# 名称对照表 jqdata:tushare
name_exchange = {
    'code': 'ts_code',
    'time': 'trade_date',
    'circulating_market_cap': 'circ_mv',
    'turnover_ratio': 'turnover_rate',
    'market_cap': 'total_mv',
    'pb_ratio': 'pb',
    'pe_ratio': 'pe',
    'total_non_current_liability': 'total_ncl',
    'total_liability': 'total_liab',
    'operating_revenue': 'revenue',
    'cash_equivalent_increase': 'n_incr_cash_cash_equ',
    'cash_and_equivalents_at_end': 'c_cash_equ_end_period',
    'non_current_liability_in_one_year': 'non_cur_liab_due_1y',
    'shortterm_loan': 'st_borr',
    'fixed_assets_depreciation': 'depr_fa_coga_dpba',
    'intangible_assets_amortization': 'amort_intang_assets',
    'defferred_expense_amortization': 'lt_amort_deferred_exp',
    'net_operate_cash_flow': 'n_cashflow_act',
    'net_invest_cash_flow': 'n_cashflow_inv_act',
    'total_operating_revenue': 'total_revenue',
    'total_operating_cost': 'total_cogs', #'total_opcost'
    'longterm_account_payable': 'lt_payable',
    'specific_account_payable': 'specific_payables',
    'fix_intan_other_asset_acqui_cash': 'c_pay_acq_const_fiolta',
    'operating_cost': 'oper_cost',
    'paidin_capital': 'total_share', #?
    'pubDate': 'ann_date',
    'statDate': 'end_date',
}
name_exchange_anti = {value: key for key, value in name_exchange.items()}
# 所有日期
all_dates = sorted(pd.to_datetime(pd.read_csv('df\可用交易日_SZSE.csv')['cal_date'], format='%Y%m%d').to_list())
# 中括号里面是tushare自带的行业，大类是申万一级行业官方名称，由deepseek归类
industry_d = {
    "农林牧渔": ["农业综合", "饲料", "农业加工", "林业", "种植业", "畜牧业", "渔业"],
    "基础化工": ["农药化肥", "化纤", "涤纶", "化肥", "粘胶纤维", "化工原料", "合成树脂", "日用化工", "农药", "钛白粉", "染料涂料", "锦纶", "无机盐", "氯碱", "塑料", "塑料薄膜", "橡胶", "其他塑料", "信息化学", "民爆用品", "氨纶", "炭黑", "磷化工", "改性塑料", "氟化工", "合成革", "化学试剂", "塑料零件", "其他化学", "有机化学", "其他原料", "纯碱"],
    "钢铁": ["普钢", "特种钢", "钢加工", "铁矿"],
    "有色金属": ["铅锌", "铝", "小金属", "稀有金属", "铜", "黄金", "金属新材"],
    "电子": ["元器件", "显示器件", "无源器件", "芯片设计", "PCB", "电子元件", "光学元件", "半导体", "LED", "芯片封测", "芯片材料", "电子化学", "元件材料", "芯片制造", "磁性材料", "安防设备", "芯片设备"],
    "汽车": ["运输设备", "汽车服务", "汽车配件", "汽车整车", "摩托车"],
    "家用电器": ["家用电器", "视听器材", "白色家电", "厨卫电器", "照明灯具", "家电材料"],
    "食品饮料": ["软饮料", "白酒", "啤酒", "红黄酒", "葡萄酒", "其他酒类", "食品", "食品综合", "肉制品", "调味品", "乳制品", "红黄药酒", "黄酒"],
    "纺织服饰": ["服饰", "纺织", "家纺"],
    "轻工制造": ["自行车", "造纸", "包装印刷", "家居用品", "家具", "家用品", "乐器", "休闲玩具", "文具用品", "珠宝首饰", "广告包装"],
    "医药生物": ["生物制药", "医药商业", "化学制药", "中成药", "保健品", "医疗服务", "医疗保健", "原料药", "医疗器械", "药用辅料", "临床研究"],
    "公用事业": ["火力发电", "水务", "供气供热", "燃气", "水力发电", "热电", "公用事业", "新型电力"],
    "交通运输": ["轨道交通", "港口", "机场", "空运", "水运", "铁路", "航空", "仓储物流", "公路", "公共交通", "汽运", "船舶", "船舶制造", "路桥"],
    "房地产": ["全国地产", "区域地产", "房产服务", "园区开发"],
    "商贸零售": ["其他商业", "商品批发", "其他连锁", "商品城", "批发业", "商贸代理", "贸易", "百货", "超市连锁", "超市", "电器连锁"],
    "社会服务": ["酒店餐饮", "文教休闲", "旅游景点", "旅游服务", "教育服务", "体育", "广告营销", "商务服务", "文化娱乐"],
    "银行": ["银行"],
    "非银金融": ["证券", "多元金融", "保险"],
    "综合": ["综合类", "其他"],
    "建筑材料": ["玻璃", "其他建材", "其它建材", "水泥", "矿物制品", "耐火材料", "陶瓷", "管材", "建材"],
    "建筑装饰": ["建筑工程", "园林工程", "建筑施工", "装修装饰", "基础建设", "房屋建筑", "建筑设计", "家装", "建筑安装"],
    "电力设备": ["电气设备", "电池", "输变电", "电自动化", "电机制造", "发电设备", "电线电缆", "电源设备", "输配电", "电控设备", "电器仪表"],
    "机械设备": ["轻工机械", "专用机械", "工程机械", "机械基件", "机床制造", "农用机械", "通用仪器", "激光设备", "纺织机械", "化工机械", "电梯", "通用机械", "金融设备", "光学仪器", "专用仪器", "航空制造"],
    "国防军工": [],
    "计算机": ["软件服务", "行业软件", "互联网", "IT设备", "电脑设备", "系统软件", "互联服务", "科技服务", "综合软件", "IT外设", "电子商务"],
    "传媒": ["影视音像", "广电传输", "出版业", "广告营销", "影视制作", "游戏", "发行院线", "文化娱乐", "文化创意", "互联媒体", "数字营销"],
    "通信": ["通信应用", "通信设备", "通信产品", "通信传输", "通信终端", "电信运营", "通信配件"],
    "煤炭": ["煤炭开采", "焦炭加工"],
    "石油石化": ["石油加工", "石油贸易", "石油开采"],
    "环保": ["环境保护", "环保工程", "环保设备", "生态保护", "环境监测"],
    "美容护理": ["个护健康"]
}

def save_df(df, path):
    if path.lower().endswith('.csv'):
        df.to_csv(path, index=False)
    else:
        with pd.ExcelWriter(path, engine='xlsxwriter') as writer:
            workbook = writer.book
            worksheet = workbook.add_worksheet('Sheet1')
            worksheet.freeze_panes(row=1, col=0)
            df.to_excel(writer, sheet_name='Sheet1',index=False)
        
def read_df(path):
    if path.lower().endswith('.csv'):
        df = pd.read_csv(path)
    else:
        df = pd.read_excel(path, sheet_name='Sheet1')
    return df

def gen_periods(start_date, end_date):
    '''
    该函数的作用是，给定两个日期 start_date 和 end_date ，返回这两个日期之间（含首尾）所有季度末的日期
    '''
    start_date = pd.to_datetime(start_date, format='%Y%m%d')
    end_date = pd.to_datetime(end_date, format='%Y%m%d')
    # 获取开始日期所在季度
    start_year = start_date.year
    start_quarter = (start_date.month - 1) // 3 + 1
    start_quarter_end = pd.to_datetime(f"{start_year}{start_quarter * 3:02d}01", format='%Y%m%d') + pd.offsets.QuarterEnd() # 不管你给的是几月几号，pd.offsets.QuarterEnd() 会自动把这个日期“推”到本季度的最后一天
    # 获取结束日期所在季度
    end_year = end_date.year
    end_quarter = (end_date.month - 1) // 3 + 1
    end_quarter_end = pd.to_datetime(f"{end_year}{end_quarter * 3:02d}01", format='%Y%m%d') + pd.offsets.QuarterEnd()
    # 生成季度列表
    quarters = []
    current_quarter = start_quarter_end
    while current_quarter <= end_quarter_end:
        quarters.append(current_quarter.strftime('%Y%m%d'))
        current_quarter += pd.offsets.QuarterEnd()
    return quarters