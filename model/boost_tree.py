#! /usr/bin/env python
# -*- coding:utf-8 -*- 

import logging
import sys
import pandas as pd
import pymysql
from sklearn import metrics
from sklearn.cross_validation import train_test_split
import xgboost as xgb
import time
import numpy as np

logger = logging.getLogger(__name__)

formatter = logging.Formatter('%(asctime)s - %(filename)s - [line:%(lineno)d] - %(levelname)s - %(message)s')
file_handler = logging.FileHandler("../log/run_mysql_xgb.log")
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
    'min_child_weight': 1,
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

number_boost_round = 500
early_stopping_rounds = 80
def load_train(file_name="../data/forced_all.csv"):
    data_frame = pd.read_csv(file_name, header=0)
    return data_frame

def load_mysql(sql):
    conn = pymysql.connect(host='192.168.1.97', port=3306, user='root', passwd='hemei@ai', db='question_simple')
    df = pd.read_sql(sql,conn,index_col=None)
    new_cols = []
    for col in df.columns.tolist():
        if '_ID_' not in col:
            new_cols.append(col)


    print(",".join(new_cols))
    print(df[new_cols].dtypes)
    print(df.dtypes)


    return df[new_cols]


def init_xgb(data,params):
    t = time.time()

    h_list = []
    from sklearn import preprocessing
    for f in data.columns:
        x= data[f].iloc[0]
        if f not in h_list:
            h_list.append(f)
        if isinstance(x,str):
            lbl = preprocessing.LabelEncoder()
            lbl.fit(list(data[f].values))
            data[f] = lbl.transform(list(data[f].values))

    data = data[h_list]
    data_train_origin ,data_test = train_test_split(data, test_size = 0.15,random_state=1)
    data_train ,data_val = train_test_split(data_train_origin, test_size = 0.15,random_state=1)

    X = data_train.drop(['TARGET'],axis=1)
    y = data_train.TARGET
    dtrain = xgb.DMatrix(X, label=y)
    dval = xgb.DMatrix(data_val.drop(['TARGET'],axis=1), label=data_val.TARGET)
    dtest = xgb.DMatrix(data_test.drop(['TARGET'],axis=1), label=data_test.TARGET)

    params['number_boost_round'] = number_boost_round
    params['early_stopping_rounds'] = early_stopping_rounds
    model = xgb.train(params, dtrain,num_boost_round=number_boost_round,evals = [(dtrain,"train"),( dval,'val')],
                      early_stopping_rounds=early_stopping_rounds,
                      evals_result = {'eval_metric': 'auc'})
    print ("best best_ntree_limit",model.best_ntree_limit)
    preds = model.predict(dtest,ntree_limit=model.best_ntree_limit)
    print ("\nModel Report")
    # print ("Accuracy : %.4g" % metrics.accuracy_score(dtrain[target].values, dtrain_predictions))
    logger.info("params {}".format(params))
    auc = "{0:.9f}".format( metrics.roc_auc_score(data_test.TARGET, preds))
    logger.info("id = {0} AUC Score (Test): {1}" .format(t,auc))
    model.save_model('../persist_model/xgb_mysql_{}_{}.model'.format(auc,t)) # 用于存储训练出的模型

    return metrics.roc_auc_score(data_test.TARGET, preds),params,t
#
# def main():
#     init_xgb(data,params)





if __name__ == '__main__':
    sql = 'select * from application_train a join bureau_dis b on a.SK_ID_CURR = b.SK_ID_CURR'
    # data = load_train()
    data = load_mysql(sql)
    best = 0
    p_m = None
    t_ = 0
    gamma_s = [i/100.0 for i in range(1,50,2)]
    max_depth_s = [i for i in range(2,15,1)]
    lambda_t_s = [i/10.0 for i in range(5,100,5)]
    subsample_s = [i/10.0 for i in range(4,10,1)]
    colsample_bytree_s = [ i/10.0 for i in range(3,10,1)]
    min_child_weight_s = [i for i in range(3,12,1)]
    eta_s =  [i/100.0 for i in range(1,100,1)]
    for gamma  in gamma_s:
        params["gamma"] = gamma

        for lambda_t in lambda_t_s:
            params["lambda"] = lambda_t
            for subsample in subsample_s:
                params['subsample'] = subsample
                for max_depth in max_depth_s:
                    params["max_depth"] = max_depth
                # for min_child_weight in min_child_weight_s:
                #     params['min_child_weight'] = min_child_weight
                    for colsample_bytree  in colsample_bytree_s:
                            params['colsample_bytree'] = colsample_bytree
                            for eta in eta_s:
                                params['eta'] = eta
                                auc,p,t = init_xgb(data,params)
                                if auc > best:
                                    best = auc
                                    p_m = p
                                    t_ = t



    logger.info("best result is  {0:.5f} ,params {1},key {}".format(best,best,t_))