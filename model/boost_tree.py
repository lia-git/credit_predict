#! /usr/bin/env python
# -*- coding:utf-8 -*- 

import logging
import sys
import pandas as pd
from sklearn import metrics
from sklearn.cross_validation import train_test_split
import xgboost as xgb
import time

t = time.time()
logger = logging.getLogger(__name__)

formatter = logging.Formatter('%(asctime)s - %(filename)s - [line:%(lineno)d] - %(levelname)s - %(message)s')
file_handler = logging.FileHandler("../log/run_xgb.log")
file_handler.setFormatter(formatter)
console_handler = logging.StreamHandler(sys.stdout)
console_handler.formatter = formatter
logger.addHandler(file_handler)
logger.addHandler(console_handler)
logger.setLevel(logging.INFO)

params = {
    'booster': 'gbtree',
    'objective':'binary:logistic',
    # 'num_class': 10,  # 类别数，与 multisoftmax 并用
    'gamma': 0.1,  # 用于控制是否后剪枝的参数,越大越保守，一般0.1、0.2这样子。
    'max_depth': 10,  # 构建树的深度，越大越容易过拟合
    'lambda': 0.8,  # 控制模型复杂度的权重值的L2正则化项参数，参数越大，模型越不容易过拟合。
    'subsample': 0.7,  # 随机采样训练样本
    'colsample_bytree': 0.7,  # 生成树时进行的列采样
    'min_child_weight': 5,
    # 这个参数默认是 1，是每个叶子里面 h 的和至少是多少，对正负样本不均衡时的 0-1 分类而言
    # ，假设 h 在 0.01 附近，min_child_weight 为 1 意味着叶子节点中最少需要包含 100 个样本。
    # 这个参数非常影响结果，控制叶子节点中二阶导的和的最小值，该参数值越小，越容易 overfitting。
    'silent': 0,  # 设置成1则没有运行信息输出，最好是设置为0.
    'eta': 0.07,  # 如同学习率
    'seed': 1000,
    # 'nthread': 7,  # cpu 线程数
    # 'eval_metric': 'auc'
    "tree_method":"gpu_hist"
}


def load_train(file_name="../data/application_train_forced.csv"):
    data_frame = pd.read_csv(file_name, header=0)
    return data_frame


def init_xgb(data):

    from sklearn import preprocessing
    for f in data.columns:
        if data[f].dtype=='object':
            lbl = preprocessing.LabelEncoder()
            lbl.fit(list(data[f].values))
            data[f] = lbl.transform(list(data[f].values))

    data_train_origin ,data_test = train_test_split(data, test_size = 0.15,random_state=1)
    data_train ,data_val = train_test_split(data_train_origin, test_size = 0.15,random_state=1)

    X = data_train.drop(['TARGET'],axis=1)
    y = data_train.TARGET
    dtrain = xgb.DMatrix(X, label=y)
    dval = xgb.DMatrix(data_val.drop(['TARGET'],axis=1), label=data_val.TARGET)
    dtest = xgb.DMatrix(data_test.drop(['TARGET'],axis=1), label=data_test.TARGET)
    model = xgb.train(params, dtrain,num_boost_round=2000,evals = [(dtrain,"train"),( dval,'val')],early_stopping_rounds=500,
                      evals_result = {'eval_metric': 'logloss'})
    model.save_model('../persist_model/xgb_{}.model'.format(t)) # 用于存储训练出的模型
    print ("best best_ntree_limit",model.best_ntree_limit)
    preds = model.predict(dtest,ntree_limit=model.best_ntree_limit)
    print ("\nModel Report")
    # print ("Accuracy : %.4g" % metrics.accuracy_score(dtrain[target].values, dtrain_predictions))
    logger.info("id = {0} AUC Score (Test): {1:.5f}" .format(t,metrics.roc_auc_score(data_test.TARGET, preds)))


if __name__ == '__main__':
    init_xgb(load_train())
