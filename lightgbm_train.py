import os
import json
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, root_mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

from utils import read_df, save_df


def read_data(delay):
    print('正在读取数据')
    print(f"使用因子数据: {os.path.abspath('result/全部三级因子.csv')}")
    barra_df = read_df('result/全部三级因子.csv')
    barra_df['time'] = pd.to_datetime(barra_df['time'])

    industry_path = 'all_industry.csv'
    industry_df_all = pd.read_csv(industry_path)
    industry_df_s = industry_df_all
    industry_df_s.dropna(subset=['trade_date'], inplace=True)  # 删除 trade_date 列中有缺失值的行
    industry_df_s.rename(columns={'ts_code': 'code', 'trade_date': 'time'}, inplace=True)
    industry_df_s['time'] = pd.to_datetime(industry_df_s['time'])
    # 筛选行业数据，使其时间范围与因子数据一致
    ind = industry_df_s[industry_df_s['time'].between(barra_df['time'].min(), barra_df['time'].max(), inclusive='both')]
    ind['cons'] = 1
    # 以 code 和 time 为索引，L1 为列名，生成行业暴露矩阵，缺失值填 0。
    Industry = ind.pivot_table(index=['code', 'time'], columns='L1', values='cons').fillna(0.)
    Industry.reset_index(inplace=True)
    # 合并
    barra_df = barra_df.merge(Industry, how='left', on=['code', 'time'])
    # 国家因于对任何股票的暴露都是1
    barra_df['Country'] = 1

    barra_df = barra_df.set_index(['code', 'time'])
    # 标准化：对每个时间点的数据进行标准化（均值为 0，方差为 1）
    barra_df = barra_df.groupby('time', group_keys=False).apply(lambda x: (x - x.mean()) / (x.std()+1e-6))
    barra_df = barra_df.reset_index()

    # 获取股票实际收益
    daily_path = 'daily.csv'
    daily_df_all = read_df(daily_path)
    daily_df_s = daily_df_all
    daily_df_s.rename(columns={'ts_code': 'code', 'trade_date': 'time'}, inplace=True)
    daily_df_s['time'] = pd.to_datetime(daily_df_s['time'], format='%Y%m%d')
    all_dates = sorted(pd.to_datetime(read_df('可用交易日_SZSE.csv')['cal_date'], format='%Y%m%d').to_list())
    def __start_end_date__(start_date, end_date, count, all_dates=all_dates):
        '''
        start_date, end_date: pandas._libs.tslibs.timestamps.Timestamp
        count: 交易日数量。
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
    
    def get_price(type, codes=None, start_date=None, end_date=None, count=250, fields=None):
        """
        type: stock: 股票; index: 指数(比如沪深300)
        codes: list,或者,'all-stock'取所有股票,'all-index'取所有指数。
        start_date, end_date:str  %Y%m%d
        count: 交易日数量。
        fields:可取['open', 'high', 'low', 'close', 'volume', 'money', 'pre_close']
        """
        if type=='stock':
            df = daily_df_s
        elif type=='index':
            df = SHSZ300_df
        all__dates = df['time'].unique()
        all__dates = all__dates.tolist()
        all__dates.sort()
        start_date, end_date = __start_end_date__(start_date, end_date, count, all_dates=all__dates)
        
        df = df[df['time'].between(start_date, end_date, inclusive='both')]
        if codes=='all-stock' or codes=='all-index' or codes==None:
            pass
        else:
            df = df[df['code'].isin(codes)]
        if fields == None:
            pass
        else:
            fields = ['time', 'code'] + fields
            df = df[fields]
        return df

    codes = 'all-stock'
    start_date = barra_df['time'].min()
    # 获取因子数据的最早日期，作为起始日期
    end_date = barra_df['time'].max() + pd.Timedelta(days=delay + 30)
    # 获取因子数据的最晚日期，并向后延长 delay+30 天，作为结束日期
    price_df = get_price(type='stock', codes=codes, start_date=start_date, end_date=end_date, fields=['close'])
    price_df.sort_values(['code', 'time'], inplace=True)
    # 对每只股票，计算未来第 delay 天的收盘价（向前移动 delay 行）
    price_df['close_future'] = price_df.groupby('code')['close'].shift(-delay)
    # 计算未来收益率：未来收盘价与当前收盘价的差值除以当前收盘价
    price_df['return_ratio'] = (price_df['close_future'] - price_df['close']) / price_df['close']
    # 将收益率大于 0 的标记为 1，否则为 0，作为分类标签
    price_df['class_label'] = (price_df['return_ratio'] > 0).astype(int)

    # 将因子数据与价格数据按股票代码和时间左连接合并，得到训练数据集
    train_df = pd.merge(
        barra_df, 
        price_df, 
        on=['code', 'time'], 
        how='left',  # 左连接（保留 df1 的所有行）
    )
    return train_df

def once_train_reg(train_df: pd.DataFrame, test_data_X):
    '''单次训练回归模型'''
    # 计算 所有日期排序后 80% 分位数的值，作为训练/验证集的分割点
    val_start = train_df['time'].quantile(0.8)
    train_df.set_index(['code', 'time'], inplace=True)
    # 去掉收盘价（close）、未来收盘价（close_future）、收益率（return_ratio）、分类标签（class_label）后
    # X 是因子特征（包含行业因子和国家因子）和行业暴露值
    X = train_df.drop(columns=['close', 'close_future', 'return_ratio', 'class_label'])
    # y 是收益率
    y = train_df['return_ratio']
    
    # 按时间划分训练集和测试集
    train_idx = X.index.get_level_values('time') < val_start
    val_idx = X.index.get_level_values('time') >= val_start

    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    # 1. 在训练集上拟合scaler（计算均值和标准差）
    scaler = StandardScaler()
    scaler.fit(X_train) # 只使用训练集计算参数
    # 2. 用训练集的标准化参数对训练集和验证集进行标准化
    X_train_scaled = scaler.transform(X_train)
    X_val_scaled = scaler.transform(X_val) # 测试集用训练集的参数转换
    X_train = X_train_scaled
    X_val = X_val_scaled

    # 训练回归模型
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
    # 设置 LightGBM 回归模型的参数
    params = {
        'boosting_type': 'gbdt',
        'objective': 'regression',#回归：预测连续值
        'metric': 'rmse',#均方根误差
        'n_estimators': 1000,#最大迭代次数（最大树数量）
        'num_leaves': 31, #单棵树的最大叶子数
        'learning_rate': 0.01,
        'feature_fraction': 0.9, #特征采样比例（列采样）：每轮迭代随机选择90%的特征，类似随机森林的列采样
        'bagging_fraction': 0.8, #数据采样比例（行采样）：每轮迭代随机选择80%的样本训练
        'bagging_freq': 5, #每5次迭代执行一次bagging（数据采样）
        'verbosity': 1, #日志输出级别：-1：不输出 0：错误信息 1：警告+基本信息
        'seed': 42# 随机数种子（重要！保证结果可复现，如果不设置，每次训练结果可能略有差异）
    }

    print("训练LightGBM回归模型...")
    model = lgb.train(
        params,
        train_data,
        num_boost_round=1000, #最大迭代次数（实际可能因早停而提前终止）
        valid_sets=[train_data, val_data],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=True), #- 若验证集指标连续50轮未提升，则停止训练
            lgb.log_evaluation(period=10) #每50轮输出一次评估结果
        ]
    )

    # 计算并排序特征重要性
    imp = dict(sorted(zip(X.columns, model.feature_importance().tolist()), key= lambda x: x[1], reverse=True))
    # 预测：用训练好的模型对测试数据进行预测
    predictions = model.predict(test_data_X)
    return imp, predictions  # 返回特征重要性和预测结果


def once_train_clf(train_df: pd.DataFrame, test_data_X):
    '''单次训练分类模型'''
    # 计算 所有日期排序后 80% 分位数的值，作为训练/验证集的分割点
    val_start = train_df['time'].quantile(0.8)
    train_df.set_index(['code', 'time'], inplace=True)
    X = train_df.drop(columns=['close', 'close_future', 'return_ratio', 'class_label'])
    # cls_y 为分类标签（上涨为1，未涨为0）
    cls_y = train_df['class_label']
    # 按时间划分训练集和测试集
    train_idx = X.index.get_level_values('time') < val_start
    val_idx = X.index.get_level_values('time') >= val_start

    # 分别获取训练和验证集的特征和标签
    X_train, X_val = X[train_idx], X[val_idx]
    y_train_clf, y_val_clf = cls_y[train_idx], cls_y[val_idx]

    # 训练分类模型
    params_clf = {
        'objective': 'binary',           # 二分类任务
        '''
        binary_logloss：二分类对数损失（越小越好），对概率预测的置信惩罚严格
        auc：ROC 曲线下面积（越接近 1 越好），对类别不平衡比较稳健
        average_precision：平均精度，关注精确率/召回率
        '''
        'metric': ['binary_logloss', 'auc', 'average_precision'],  # 评估指标
        'boosting_type': 'gbdt',         # 梯度提升决策树
        'num_leaves': 31,                # 树的最大叶子数
        'learning_rate': 0.01,           # 学习率
        'feature_fraction': 0.8,         # 每次迭代使用的特征比例（列采样）
        'bagging_fraction': 0.8,         # 每次迭代使用的数据比例（行采样）
        'bagging_freq': 5,               # 每5次迭代执行一次bagging（行采样）
        'verbose': -1,                   # 不输出训练信息
        'random_state': 42,              # 随机种子
        'n_estimators': 1000,            # 树的数量
        'reg_alpha': 0.1,                # L1正则化（对叶子权重系数施加 L1 惩罚）
        'reg_lambda': 0.1,               # L2正则化（对叶子权重系数施加 L2 惩罚）
        'min_child_samples': 20,         # 叶子节点最少样本数
        'class_weight': 'balanced'       # 处理类别不平衡，按类别频率自动设置权重
        }
    print("训练LightGBM分类模型...")
    model = lgb.LGBMClassifier(**params_clf)
    model.fit(
        X_train, y_train_clf,
        eval_set=[(X_val, y_val_clf)],      # 验证集用于早停
        eval_metric=['binary_logloss', 'auc'],  # 评估指标
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=True), # 若验证集指标连续50轮未提升，则停止训练
            lgb.log_evaluation(period=10) #每50轮输出一次评估结果
        ]
        )

    # 计算并排序特征重要性
    imp = dict(sorted(zip(X.columns, model.feature_importances_.tolist()), key= lambda x: x[1], reverse=True))
    # 预测：用训练好的模型对测试数据进行类别预测（0/1）
    predictions = model.predict(test_data_X)
    # 预测测试数据为正类（上涨）的概率
    prob = model.predict_proba(test_data_X)[:, 1]
    return imp, predictions, prob  # 返回特征重要性、预测类别和预测概率
    

if __name__ == "__main__":
    '''可设置'''
    数据文件夹 = 'tushare-20120101_20250101'
    # 训练窗口天数数, 一年大概250个交易日
    train_window = 1390
    # 预测未来第n个交易日, 并每第n个交易日滚动训练
    delay = 15
    # 模型类型: reg: 回归模型, clf: 分类模型
    model_type = 'clf'

    os.chdir(数据文件夹)
    train_df = read_data(delay)
    result_dir = f'lightgbm_{train_window}_{delay}_{model_type}'
    os.makedirs(result_dir, exist_ok=True)

    # 统计并打印数据的时间范围和交易日数量
    all_dates = sorted(train_df['time'].unique().tolist())
    print(f"总因子数据时间窗口: {train_df['time'].min().strftime('%Y-%m-%d')} - {train_df['time'].max().strftime('%Y-%m-%d')}")
    print(f"总因子数据交易日数量: {len(all_dates)}")
    # 如果训练窗口加预测天数超过数据长度，则报错
    if train_window + delay > len(all_dates):
        raise ValueError(f'训练窗口天数{train_window} 加上 预测未来天数{delay} 超出已有数据范围（{all_dates[0]} - {all_dates[-1]}）')
    
    # 进行滚动回测
    print("开始滚动回测...")
    # 初始化滚动窗口的起止索引
    start_index = 0
    end_index = start_index + train_window
    test_index = end_index + 1
    
    # 每次循环：确定训练窗口和测试日期，准备训练和测试数据
    while end_index < len(all_dates)-1:
        # 训练开始日期
        start_date = all_dates[start_index]
        # 训练结束日期
        end_date = all_dates[end_index]
        print(f"处理窗口: {start_date.strftime('%Y-%m-%d')} 到 {end_date.strftime('%Y-%m-%d')}")
        print(f"窗口内天数: {end_index - start_index}")
        # print(len(all_dates), train_window, start_date, end_date)
        # raise SystemExit
        result_path = os.path.join(result_dir, f'{start_date.strftime("%Y_%m_%d")}--{end_date.strftime("%Y_%m_%d")}.json')
        # 测试日期
        test_date = all_dates[test_index]
        once_train_df = train_df[train_df['time'].between(start_date, end_date, inclusive='both')]
        test_data = train_df[train_df['time']==test_date]
        test_data.set_index(['code', 'time'], inplace=True)
        test_data_X = test_data.drop(columns=['close', 'close_future', 'return_ratio', 'class_label'])
        test_data_y_reg = test_data['return_ratio']
        test_data_y_clf = test_data['class_label']

        # 单次训练,分别返回回归模型和分类模型
        if model_type == 'reg':
            imp, pred = once_train_reg(once_train_df, test_data_X)
            # 回归模型评估
            rmse = root_mean_squared_error(test_data_y_reg, pred)
            r2 = r2_score(test_data_y_reg, pred)
            # 存储结果
            with open(result_path, 'w', encoding='utf8') as f:
                json.dump({
                    'rmse': rmse,
                    'r2': r2,
                    '特征重要性': imp,
                }, f, ensure_ascii=False, indent=2)
            test_data['预测收益率'] = pred
        else:
            imp, pred, prob = once_train_clf(once_train_df, test_data_X)
            # 分类模型评估
            accuracy = accuracy_score(test_data_y_clf, pred)
            f1 = f1_score(test_data_y_clf, pred)
            auc = roc_auc_score(test_data_y_clf, prob)
            # 存储结果
            with open(result_path, 'w', encoding='utf8') as f:
                json.dump({
                    'accuracy': accuracy,
                    'f1': f1,
                    'auc': auc,
                    '特征重要性': imp,
                }, f, ensure_ascii=False, indent=2)
            test_data['预测上涨概率'] = prob

        # 窗口向后滚动 delay 天，准备下一次回测
        start_index += delay
        end_index += delay
        test_index = end_index + 1

    # 创建投资组合：根据模型类型生成投资组合（按预测值排序）
    print("创建投资组合...")
    test_data.reset_index(inplace=True)
    if model_type == 'reg':
        portfolio = test_data[['code', 'close','close_future','return_ratio','class_label','预测收益率']]
        # 基于回归预测的组合（选择预测收益率最高的股票）
        portfolio = portfolio.sort_values('预测收益率', ascending=False)
    else:
        portfolio = test_data[['code', 'close','close_future','return_ratio','class_label','预测上涨概率']]
        # 基于分类预测的组合（选择上涨概率最高的股票）
        portfolio = portfolio.sort_values('预测上涨概率', ascending=False)
    portfolio_result_path = os.path.join(result_dir, f'投资组合.csv')
    save_df(portfolio, portfolio_result_path)
    print(f"\n结果已保存到{os.path.abspath(result_dir)}")