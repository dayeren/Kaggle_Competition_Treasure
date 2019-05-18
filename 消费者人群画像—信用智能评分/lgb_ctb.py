# -*- coding: utf-8 -*-
# @Time    : 2019/1/22 1:24 PM
# @Author  : Inf.Turing
# @Site    :
# @File    : lgb_baseline.py
# @Software: PyCharm

# 不要浪费太多时间在自己熟悉的地方，要学会适当的绕过一些
# 良好的阶段性收获是坚持的重要动力之一
# 用心做事情，一定会有回报
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import ElasticNet, BayesianRidge
from sklearn.model_selection import KFold, StratifiedKFold, RepeatedKFold
from sklearn.metrics import mean_squared_error as mse, mean_absolute_error as mae, mean_absolute_error
import xgboost as xgb
import matplotlib.pyplot as plt
import catboost as ctb

path = '/Users/inf/PycharmProject/kaggle/dcic_china_move/data'

train = pd.read_csv(path + '/train_dataset.csv')
train['type'] = 1
test = pd.read_csv(path + '/test_dataset.csv')
test['type'] = 0
data = pd.concat([train, test], ignore_index=True)

# 关键特征被fillna 0 了 这里还原回来
data.loc[data['用户年龄'] == 0, '用户年龄'] = None
data.loc[data['用户年龄'] > 100, '用户年龄'] = None
data.loc[data['用户话费敏感度'] == 0, '用户话费敏感度'] = None
data.loc[data['用户近6个月平均消费值（元）'] == 0, '用户近6个月平均消费值（元）'] = None

data.rename(columns={'用户编码': 'id', '信用分': 'score'}, inplace=True)

origin_bool_feature = ['当月是否体育场馆消费', '当月是否景点游览', '当月是否看电影', '当月是否到过福州山姆会员店', '当月是否逛过福州仓山万达',
                       '缴费用户当前是否欠费缴费', '是否经常逛商场的人', '是否大学生客户', '是否4G不健康客户', '是否黑名单客户',
                       '用户最近一次缴费距今时长（月）', '用户实名制是否通过核实']

origin_num_feature = ['用户话费敏感度', '用户年龄', '近三个月月均商场出现次数', '当月火车类应用使用次数', '当月飞机类应用使用次数',
                      '当月物流快递类应用使用次数', '用户当月账户余额（元）', '用户网龄（月）', '缴费用户最近一次缴费金额（元）',
                      '当月通话交往圈人数', '当月旅游资讯类应用使用次数', '当月金融理财类应用使用总次数', '当月网购类应用使用次数',
                      '当月视频播放类应用使用次数', '用户账单当月总费用（元）', '用户近6个月平均消费值（元）']

count_feature_list = []

for i in ['用户近6个月平均消费值（元）', '用户账单当月总费用（元）', '缴费用户最近一次缴费金额（元）']:
    count_feature_list.append('count_' + i)
    data['count_' + i] = data[i].map(data[i].value_counts())

# 业务特征
data['five_all'] = data['用户近6个月平均消费值（元）'] * data['用户网龄（月）'].apply(lambda x: min(x, 6)) - data['用户账单当月总费用（元）']
data['fee_del_mean'] = data['用户账单当月总费用（元）'] - data['用户近6个月平均消费值（元）']
data['fee_remain_now'] = data['缴费用户最近一次缴费金额（元）'] / data['用户账单当月总费用（元）']

data['次数'] = data[['当月网购类应用使用次数', '当月物流快递类应用使用次数', '当月金融理财类应用使用总次数',
                   '当月视频播放类应用使用次数', '当月飞机类应用使用次数', '当月火车类应用使用次数', '当月旅游资讯类应用使用次数']].sum(axis=1)

for col in ['当月金融理财类应用使用总次数', '当月旅游资讯类应用使用次数']:  # 这两个比较积极向上一点
    data[col + '_百分比'] = data[col] / data['次数']
data['regist_month'] = data['用户网龄（月）'] % 12



lgb_model = lgb.LGBMRegressor(
    num_leaves=32, reg_alpha=0., reg_lambda=0.01, objective='mse', metric='mae',
    max_depth=-1, learning_rate=0.01, min_child_samples=50,
    n_estimators=15000, subsample=0.7, colsample_bytree=0.45, subsample_freq=5,
)

# ab_id 是去除的200个样本的id。是由5折预测出来的结果的loss最大的200个构成。也可以将这200个数据权重设置0.01（线下好像更好了。没机会试了）
ab_id = [
]

# 设置样本权重
data['temp_label'] = data['score']
# 这里设置为None 而不是删除该数据，因为删除的话，线下一定是提升的，对于线上而言，异常数据依旧存在，所以应该关注在训练集无异常，而测试集有异常下的处理效果
data['sample_weight'] = data['temp_label'] + 200
data['sample_weight'] = data['sample_weight'] / data['sample_weight'].mean()
# 方案1 ，不训练
data.loc[data.id.isin(ab_id), 'temp_label'] = None
# 方案2，样本权重设置低一点
data.loc[data.id.isin(ab_id), 'sample_weight'] = 0.01

# 感谢大佬分享的参数
ctb_params = {
    'n_estimators': 10000,
    'learning_rate': 0.02,
    'random_seed': 4590,
    'reg_lambda': 0.08,
    'subsample': 0.7,
    'bootstrap_type': 'Bernoulli',
    'boosting_type': 'Plain',
    'one_hot_max_size': 10,
    'rsm': 0.5,
    'leaf_estimation_iterations': 5,
    'use_best_model': True,
    'max_depth': 6,
    'verbose': -1,
    'thread_count': 4
}
ctb_model = ctb.CatBoostRegressor(**ctb_params)