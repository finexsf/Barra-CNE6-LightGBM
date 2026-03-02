from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# ====== 新增：统一字体和字号 ======
plt.rcParams['font.size'] = 15
plt.rcParams['axes.titlesize'] = 25
plt.rcParams['axes.labelsize'] = 15
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 13

# 累计收益率走势图
print('正在绘制纯因子累计收益率走势图')
Path('photo').mkdir(parents=True, exist_ok=True)
# 纯因子顺序与表格一致
pure_factor_order = [
    'Size', 'Volatility', 'Liquidity',
    'Momentum', 'Quality', 'Value',
    'Growth', 'Sentiment', 'Dividend'
]
res = pd.read_csv('/home/xusifan/LL/cal_cne6_902/tushare-20120101_20250101/result/纯因子收益率.csv')
res['time'] = pd.to_datetime(res['time'])
count = 250
fig, axs = plt.subplots(3, 3, figsize=(18, 14))
axs = axs.ravel()
for i, factor in enumerate(pure_factor_order):
    if factor not in res.columns:
        continue
    s = res[factor].iloc[-count-1:].reset_index(drop=True)
    s.iloc[0] = 0.
    s = s.add(1).cumprod()
    axs[i].plot(res['time'].iloc[-count-1:], s, color='k')
    axs[i].set_title(factor)
    axs[i].grid()
# 删除多余子图（如果有）
for j in range(len(pure_factor_order), len(axs)):
    fig.delaxes(axs[j])
fig.tight_layout()
fig.savefig('/home/xusifan/LL/cal_cne6_902/photo/风格纯因子收益走势图.png', dpi=300)
print('累计收益率走势图已保存到 /home/xusifan/LL/cal_cne6_902/photo/风格纯因子收益走势图.png')

# ==============================================

# 读取IC数据
ic_df = pd.read_csv('/home/xusifan/LL/cal_cne6_902/tushare-20120101_20250101/result/ic.csv')

# 风格与三级因子映射（请根据实际数据调整）
style_factors = {
    'Size': ['LNCAP', 'MIDCAP'],
    'Volatility': ['BETA', 'Hist_sigma', 'Daily_std', 'Cumulative_range'],
    'Liquidity': ['Monthly_share_turnover', 'Quarterly_share_turnover', 'Annual_share_turnover', 'Annualized_traded_value_ratio'],
    'Momentum': ['Short_Term_reversal', 'Seasonality', 'Industry_Momentum', 'Relative_strength', 'Hist_alpha'],
    'Quality': ['Market_Leverage','Book_Leverage','Debt_to_asset_ratio',
                'Variation_in_Sales','Variation_in_Earning','Variation_in_Cashflow','forecast_EP_std',
                'ABS','ACF',
                'ATO','GP','GPM','ROA',
                'Total_Assets_Growth_Rate','Issuance_growth','Capital_expenditure_growth'],
    'Value': ['Book_to_price',
              'Earning_to_price','Cash_earning_to_price','forecast_EP_mean','Enterprise_multiple',
              'Longterm_Relative_strength','Longterm_Alpha'],
    'Growth': ['Earning_Growth_Rate', 'OP_Growth_Rate', 'roe_mean'],
    'Sentiment': ['Pred_EP_chg', 'Pred_EPS_chg'],
    'Dividend Yield': ['Dividend_to_Price', 'Forecast_Dividend_to_Price']

}

# 每个风格的子图布局（行数, 列数）
style_layouts = {
    'Size': (1, 2),
    'Volatility': (2, 2),
    'Liquidity': (1, 4),
    'Momentum': (2, 3),
    'Quality': (4, 4),
    'Value': (2, 4),
    'Growth': (1, 3),
    'Sentiment': (1, 2),
    'Dividend Yield': (1, 2)
}

count = 250  # 保持与累计收益率走势图一致
for style, factors in tqdm(style_factors.items(), desc='风格IC序列'):
    n = len(factors)
    n_row, n_col = style_layouts.get(style, ((n + 3) // 4, 4))
    fig, axs = plt.subplots(n_row, n_col, figsize=(7*n_col, 5*n_row))
    axs = np.array(axs).reshape(-1, n_col)
    axs_flat = axs.flatten()
    for i, factor in enumerate(factors):
        if factor not in ic_df.columns:
            continue
        ax = axs_flat[i]
        ic_series = ic_df[factor].iloc[-count-1:].reset_index(drop=True)
        ic_cum = ic_series.cumsum()
        # 柱状图：红色为正，蓝色为负
        colors = np.where(ic_series >= 0, '#b45a5a', '#5a6ea6')
        ax.bar(np.arange(len(ic_series)), ic_series, color=colors, alpha=0.5, width=1.0)
        # 累计IC黑色实线
        ax2 = ax.twinx()
        ax2.plot(np.arange(len(ic_cum)), ic_cum, color='k', linewidth=2)
        # 计算IC均值和IR
        ic_mean = ic_series.mean()
        ic_ir = ic_series.mean() / ic_series.std() if ic_series.std() != 0 else 0
        # 将标题放在x轴下方，并加(a)、(b)等
        subtitle = f"({chr(97+i)}) {factor}, IC = {ic_mean:.4f}, IR = {ic_ir:.4f}"
        ax.set_xlabel(subtitle)
        ax.axhline(0, color='gray', linestyle='--', linewidth=1)
        ax.grid()
        # y轴设置
        ax2.set_ylabel('Cumulative IC')
        ax.set_ylabel('IC')
        ax2.axhline(0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
        ax2.set_ylim(ic_cum.min()*1.2, ic_cum.max()*1.2)
    # 删除多余子图
    for j in range(n, n_row * n_col):
        fig.delaxes(axs_flat[j])
    fig.tight_layout(rect=[0, 0, 1, 1])
    fig.savefig(f'/home/xusifan/LL/cal_cne6_902/photo/{style}.png', dpi=300)
    plt.close(fig)
    print(f'{style} 风格IC图已保存到 /home/xusifan/LL/cal_cne6_902/photo/{style}.png')


# ====== 新增：一级因子IC累计序列图 ======
print('正在绘制一级因子IC累计序列图...')
first_factor_path = '/home/xusifan/LL/cal_cne6_902/tushare-20120101_20250101/result/一级因子.csv'
first_factor_df = pd.read_csv(first_factor_path)
if 'time' in first_factor_df.columns:
    first_factor_df['time'] = pd.to_datetime(first_factor_df['time'])
    time_index = first_factor_df['time']
    factor_names = [col for col in first_factor_df.columns if col != 'time']
else:
    time_index = np.arange(len(first_factor_df))
    factor_names = list(first_factor_df.columns)

# 按表格顺序排列
pure_factor_order = [
    'Size', 'Volatility', 'Liquidity',
    'Momentum', 'Quality', 'Value',
    'Growth', 'Sentiment', 'Dividend'
]

count = 250
fig, axs = plt.subplots(3, 3, figsize=(14, 14))
axs = axs.ravel()
for i, factor in enumerate(pure_factor_order):
    if factor not in first_factor_df.columns:
        continue
    ic_series = first_factor_df[factor].iloc[-count-1:].reset_index(drop=True)
    ic_cum = ic_series.cumsum()
    ax = axs[i]
    # 柱状图：红色为正，蓝色为负
    colors = np.where(ic_series >= 0, '#b45a5a', '#5a6ea6')
    ax.bar(np.arange(len(ic_series)), ic_series, color=colors, alpha=0.5, width=1.0)
    # 累计IC黑色实线
    ax2 = ax.twinx()
    ax2.plot(np.arange(len(ic_cum)), ic_cum, color='k', linewidth=2)
    # 计算IC均值和IR
    ic_mean = ic_series.mean()
    ic_ir = ic_series.mean() / ic_series.std() if ic_series.std() != 0 else 0
    # 标题放在x轴下方，并加(a)、(b)等
    subtitle = f"({chr(97+i)}) {factor}, IC = {ic_mean:.4f}, IR = {ic_ir:.4f}"
    ax.set_xlabel(subtitle)
    ax.axhline(0, color='gray', linestyle='--', linewidth=1)
    ax.grid()
    ax2.set_ylabel('Cumulative IC')
    ax.set_ylabel('IC')
    ax2.axhline(0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax2.set_ylim(ic_cum.min()*1.2, ic_cum.max()*1.2)
# 删除多余子图（如果有）
for j in range(len(pure_factor_order), len(axs)):
    fig.delaxes(axs[j])
fig.tight_layout()
fig.subplots_adjust(top=0.90)
fig.savefig('/home/xusifan/LL/cal_cne6_902/photo/一级因子IC累计序列.png', dpi=300)
plt.close(fig)
print('一级因子IC累计序列图已保存到 /home/xusifan/LL/cal_cne6_902/photo/一级因子IC累计序列.png')

# ====== 新增：风格纯因子收益率走势图 ======
print('正在绘制风格纯因子累计收益率走势图...')
style_factor_names = [
    'Growth', 'Liquidity', 
    'Momentum', 'Quality', 
    'Sentiment', 'Size', 
    'Value', 'Volatility', 
    'Dividend'
]
# 读取纯因子收益率数据
style_res_path = '/home/xusifan/LL/cal_cne6_902/tushare-20120101_20250101/result/纯因子收益率.csv'
style_res = pd.read_csv(style_res_path)
if 'time' in style_res.columns:
    style_res['time'] = pd.to_datetime(style_res['time'])
    time_index = style_res['time'].iloc[-251:]
else:
    time_index = np.arange(251)

# ====== 新增：风格纯因子收益率单图 ======
print('正在绘制风格纯因子累计收益率单图...')
single_plot_factors = [
    'Value', 'Size', 'Volatility', 'Liquidity', 'Momentum', 
    'Quality', 'Growth', 'Sentiment', 'Dividend'
]
# 读取纯因子收益率数据
single_res_path = '/home/xusifan/LL/cal_cne6_902/tushare-20120101_20250101/result/纯因子收益率.csv'
single_res = pd.read_csv(single_res_path)
if 'time' in single_res.columns:
    single_res['time'] = pd.to_datetime(single_res['time'])
    time_index = single_res['time'].iloc[-251:]
else:
    time_index = np.arange(251)

count = 250
plt.figure(figsize=(12, 7))
color_list = [
    'tab:blue', 'tab:orange', 'tab:green', 'tab:purple', 'tab:brown',
    'tab:gray', 'tab:pink', 'tab:red', 'tab:olive'
]
for i, factor in tqdm(enumerate(single_plot_factors), desc='风格纯因子收益率单图', total=len(single_plot_factors)):
    if factor not in single_res.columns:
        continue
    s = single_res[factor].iloc[-count-1:].reset_index(drop=True)
    s.iloc[0] = 0.
    s = s.add(1).cumprod()
    # 图注英文单词仅首字母大写
    plt.plot(time_index, s, label=factor.capitalize(), color=color_list[i % len(color_list)])
plt.legend(title='FACTOR')
plt.grid()
plt.tight_layout()
plt.savefig('/home/xusifan/LL/cal_cne6_902/photo/风格纯因子收益率_单图.png', dpi=300)
plt.close()
print('风格纯因子累计收益率单图已保存到 /home/xusifan/LL/cal_cne6_902/photo/风格纯因子收益率_单图.png')

