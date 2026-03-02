# 基于 LightGBM 的非线性多因子选股与收益预测系统

## 📌 项目简介

本项目基于 **LightGBM 梯度提升树算法**，构建非线性多因子选股与收益预测模型，对 **Barra CNE6 风险模型因子**进行组合建模与预测分析。

本项目对应论文：

> *A LightGBM-Based Nonlinear Multi-Factor Model Using Barra CNE6 Factors*
> Accepted at SSDAM 2026 (CPCI indexed)

软件著作权：

> 《基于机器学习的多因子选股与收益预测分析软件 V1.0》（申请中）

------

## 🚀 项目特点

- ✅ 基于 LightGBM 构建非线性多因子模型
- ✅ 支持高维 Barra CNE6 因子输入
- ✅ 自动特征选择与因子重要性排序
- ✅ 支持滚动训练与时间序列回测
- ✅ 输出收益预测结果与选股组合
- ✅ 可扩展至 XGBoost / CatBoost 等模型对比实验

------

## 🏗 系统架构

```
数据预处理
    ↓
因子标准化 / 去极值
    ↓
训练集 / 测试集划分（时间序列）
    ↓
LightGBM 模型训练
    ↓
收益预测
    ↓
因子重要性分析
    ↓
组合构建与回测
```

------

## 🔧 环境配置

`Python 3.11`

```
pip install -r requirements.txt
```

------

## ▶️ 使用方式

```
cd cal_cne6_902  # cal_cne6_902 文件夹内是最终版代码
python tushare_data.py  # 从tushare获取数据
python cne6.py  # 计算barra cne6因子
python lightgbm_train.py  # 训练lightgbm：分类（涨/跌）/回归（涨跌概率），包含barra cne6因子、国家因子、行业因子
python lightgbm_train no_industry&country.py  # 只包含barra cne6因子，可更直观观察barra cne6因子的影响
python draw.py  # 绘图代码
```

注：

1、调用tushare api获取数据需要积分，获取字段不同积分要求不同详见tushare文档，可以从咸鱼买有积分的账号使用，积分数量不同价格不同

2、utils.py是数据预处理的工具，包括tushare与聚宽jqdata的映射关系、三级行业因子与一级行业因子的关系（使用Deepseek归纳）等

3、所有数据都保存在al_cne6_902/tushare-20120101_20250101`

------

## 📜 免责声明

本项目仅用于学术研究与技术交流，不构成任何投资建议。

投资有风险，入市需谨慎。