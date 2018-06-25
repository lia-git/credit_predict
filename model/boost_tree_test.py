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
file_handler = logging.FileHandler("../log/run_mysql_xgb_test.log")
file_handler.setFormatter(formatter)
console_handler = logging.StreamHandler(sys.stdout)
console_handler.formatter = formatter
logger.addHandler(file_handler)
logger.addHandler(console_handler)
logger.setLevel(logging.INFO)

# params = {
#     'booster': 'gbtree',
#     'objective': 'binary:logistic',
#     # 'num_class': 10,  # 类别数，与 multisoftmax 并用
#     'gamma': 0.1,  # 用于控制是否后剪枝的参数,越大越保守，一般0.1、0.2这样子。
#     'max_depth': 10,  # 构建树的深度，越大越容易过拟合
#     'lambda': 0.8,  # 控制模型复杂度的权重值的L2正则化项参数，参数越大，模型越不容易过拟合。
#     'subsample': 0.7,  # 随机采样训练样本
#     'colsample_bytree': 0.7,  # 生成树时进行的列采样
#     'min_child_weight': 1,
#     # 这个参数默认是 1，是每个叶子里面 h 的和至少是多少，对正负样本不均衡时的 0-1 分类而言
#     # ，假设 h 在 0.01 附近，min_child_weight 为 1 意味着叶子节点中最少需要包含 100 个样本。
#     # 这个参数非常影响结果，控制叶子节点中二阶导的和的最小值，该参数值越小，越容易 overfitting。
#     'silent': 0,  # 设置成1则没有运行信息输出，最好是设置为0.
#     'eta': 0.07,  # 如同学习率
#     'seed': 1000,
#     # 'nthread': 7,  # cpu 线程数
#     # 'eval_metric': 'auc'
#     "tree_method": "gpu_hist"
# }

params =  {'booster': 'gbtree', 'objective': 'binary:logistic', 'gamma': 0.01, 'max_depth': 3, 'lambda': 0.5, 'subsample': 0.4, 'colsample_bytree': 0.7, 'min_child_weight': 1, 'silent': 0, 'eta': 0.41, 'seed': 1000, 'tree_method': 'gpu_hist'}
number_boost_round = 500
early_stopping_rounds = 80


def load_train(file_name="../data/forced_all.csv"):
    data_frame = pd.read_csv(file_name, header=0)
    return data_frame


def load_mysql(sql):
    conn = pymysql.connect(host='192.168.1.97', port=3306, user='root', passwd='hemei@ai', db='question_simple')
    df = pd.read_sql(sql, conn, index_col=None)
    new_cols = []
    for col in df.columns.tolist():
        if '_ID_' not in col:
            new_cols.append(col)

    print(",".join(new_cols))
    print(df[new_cols].dtypes)
    print(df.dtypes)

    return df[new_cols]

def load_extra_ids(sql):
    conn = pymysql.connect(host='192.168.1.97', port=3306, user='root', passwd='hemei@ai', db='question_simple')
    df = pd.read_sql(sql, conn, index_col=None)
    new_cols = []
    for col in df.columns.tolist():


        return df[col]


def load_mysql_test(sql):
    conn = pymysql.connect(host='192.168.1.97', port=3306, user='root', passwd='hemei@ai', db='question_simple')
    df = pd.read_sql(sql, conn, index_col=None)
    new_cols = []
    for col in df.columns.tolist():
        if '_ID_' not in col:
            new_cols.append(col)

    print(",".join(new_cols))
    print(df[new_cols].dtypes)
    print(df.dtypes)

    return df["SK_ID_CURR"],df[new_cols]


# def wash_data(data):

def init_xgb(data, params,data_un = None):
    t = time.time()

    h_list = []
    h_test = []
    from sklearn import preprocessing
    for f in data.columns:
        x = data[f].iloc[0]
        if f not in h_list:
            h_list.append(f)
            if f != "TARGET" :
                h_test.append(f)
        if isinstance(x, str):
            lbl = preprocessing.LabelEncoder()
            lbl.fit(list(data[f].values))
            data[f] = lbl.transform(list(data[f].values))
            if data_un is not None:
                lbl.fit(list(data_un[f].values))
                data_un[f] = lbl.transform(list(data_un[f].values))




    data = data[h_list]
    if data_un is not None:
        data_un = data_un[h_test]
    data_train_origin, data_test = train_test_split(data, test_size=0.15, random_state=1)
    data_train, data_val = train_test_split(data_train_origin, test_size=0.15, random_state=1)

    X = data_train.drop(['TARGET'], axis=1)
    y = data_train.TARGET
    dtrain = xgb.DMatrix(X, label=y)
    dval = xgb.DMatrix(data_val.drop(['TARGET'], axis=1), label=data_val.TARGET)
    dtest = xgb.DMatrix(data_test.drop(['TARGET'], axis=1), label=data_test.TARGET)

    params['number_boost_round'] = number_boost_round
    params['early_stopping_rounds'] = early_stopping_rounds
    # model = xgb.train(params, dtrain, num_boost_round=number_boost_round, evals=[(dtrain, "train"), (dval, 'val')],
    #                   early_stopping_rounds=early_stopping_rounds,
    #                   evals_result={'eval_metric': 'auc'})

    model = xgb.Booster(model_file='../persist_model/xgb_mysql_{}.model'.format(model_file))
    # dtest2 = xgb.DMatrix('dtest.buffer')
    # preds2 = bst2.predict(dtest2)
    print("best best_ntree_limit", model.best_ntree_limit)
    preds = model.predict(dtest, ntree_limit=model.best_ntree_limit)
    print("\nModel Report")
    params["model_name"] = 'xgb_mysql_{}.model'.format(model_file)
    # print ("Accuracy : %.4g" % metrics.accuracy_score(dtrain[target].values, dtrain_predictions))
    logger.info("params {}".format(params))
    auc = "{0:.9f}".format(metrics.roc_auc_score(data_test.TARGET, preds))
    # logger.info("id = {0} AUC Score (Test): {1}".format(t, auc))
    # model.save_model('../persist_model/xgb_mysql_{}_{}.model'.format(auc, t))  # 用于存储训练出的模型
    if data_un is not None:
        test = xgb.DMatrix(data_un, label=None)
        preds_test = model.predict(test, ntree_limit=model.best_ntree_limit)
        p = pd.DataFrame(preds_test,columns=["TARGET"])
        res = pd.concat([Ids,p],axis=1)

        res = res.groupby('SK_ID_CURR').agg({'TARGET':'mean'})
        res = res.reindex_axis(['TARGET'], axis=1)
        res = res.rename_axis(None).reset_index()
    # res.rename_axis
        print(res.columns.tolist())
        res = res.rename(columns={'index': 'SK_ID_CURR'})
        print(res.head(5))

        exsist_ids = Ids.tolist()
        extra_ids = list(set(all_ids) - set(exsist_ids))
        import random
        random.seed(10)
        v_l = []
        for i in range(len(extra_ids)):
            v_l.append(random.uniform(0.1,1.0))
        extra_pd = pd.DataFrame({"SK_ID_CURR":extra_ids,"TARGET":v_l})
        res = pd.concat([res,extra_pd])
        print("missing id : {}".format(len(extra_ids)))

        print(res.head(5))

    #reset index

        save_file(res,"../log/submission.csv")

        # res.reset_index()





    return metrics.roc_auc_score(data_test.TARGET, preds), params, t



def save_file(df,file_name):
    df.to_csv(file_name,index=False,header=True)




#
# def main():
#     init_xgb(data,params)


if __name__ == '__main__':
    model_file = sys.argv[1]
    sql = 'select * from application_train a join bureau_dis b on a.SK_ID_CURR = b.SK_ID_CURR '
    # data = load_train()
    data = load_mysql(sql)
    sql_test = '''
                select a.SK_ID_CURR,NAME_CONTRACT_TYPE,CODE_GENDER,FLAG_OWN_CAR,FLAG_OWN_REALTY,CNT_CHILDREN,AMT_INCOME_TOTAL,AMT_CREDIT,NAME_INCOME_TYPE,NAME_EDUCATION_TYPE,NAME_FAMILY_STATUS,NAME_HOUSING_TYPE,REGION_POPULATION_RELATIVE,DAYS_BIRTH,DAYS_EMPLOYED,DAYS_REGISTRATION,FLAG_MOBIL,FLAG_EMP_PHONE,FLAG_WORK_PHONE,FLAG_CONT_MOBILE,FLAG_PHONE,FLAG_EMAIL,REGION_RATING_CLIENT,REGION_RATING_CLIENT_W_CITY,WEEKDAY_APPR_PROCESS_START,HOUR_APPR_PROCESS_START,REG_REGION_NOT_LIVE_REGION,REG_REGION_NOT_WORK_REGION,LIVE_REGION_NOT_WORK_REGION,REG_CITY_NOT_LIVE_CITY,REG_CITY_NOT_WORK_CITY,LIVE_CITY_NOT_WORK_CITY,ORGANIZATION_TYPE,FLAG_DOCUMENT_2,FLAG_DOCUMENT_3,FLAG_DOCUMENT_4,FLAG_DOCUMENT_5,FLAG_DOCUMENT_6,FLAG_DOCUMENT_7,FLAG_DOCUMENT_8,FLAG_DOCUMENT_9,FLAG_DOCUMENT_10,FLAG_DOCUMENT_11,FLAG_DOCUMENT_12,FLAG_DOCUMENT_13,FLAG_DOCUMENT_14,FLAG_DOCUMENT_15,FLAG_DOCUMENT_16,FLAG_DOCUMENT_17,FLAG_DOCUMENT_18,FLAG_DOCUMENT_19,FLAG_DOCUMENT_20,FLAG_DOCUMENT_21,CREDIT_ACTIVE,CREDIT_CURRENCY,DAYS_CREDIT,CREDIT_DAY_OVERDUE,CNT_CREDIT_PROLONG,AMT_CREDIT_SUM_OVERDUE,CREDIT_TYPE,DAYS_CREDIT_UPDATE,MONTHS_BALANCE,STATUS
               from  
               application_test a join bureau_dis b on a.SK_ID_CURR = b.SK_ID_CURR 
                '''
    # data = load_train()
    Ids,data_un = load_mysql_test(sql_test)
    all_ids = load_extra_ids("select distinct SK_ID_CURR from application_test").tolist()

    best = 0
    p_m = None
    t_ = 0
    # gamma_s = [i/100.0 for i in range(1,50,2)]
    # max_depth_s = [i for i in range(2,15,1)]
    # lambda_t_s = [i/10.0 for i in range(5,100,1)]
    # subsample_s = [i/10.0 for i in range(4,10,1)]
    # colsample_bytree_s = [ i/10.0 for i in range(3,10,1)]
    # min_child_weight_s = [i for i in range(3,12,1)]
    # eta_s =  [i/100.0 for i in range(1,100,1)]
    # for gamma  in gamma_s:
    #     params["gamma"] = gamma
    #     for max_depth in max_depth_s:
    #         params["max_depth"] = max_depth
    #         for lambda_t in lambda_t_s:
    #             params["lambda"] = lambda_t
    #             for subsample in subsample_s:
    #                 params['subsample'] = subsample
    #                 # for min_child_weight in min_child_weight_s:
    #                 #     params['min_child_weight'] = min_child_weight
    #                 for colsample_bytree  in colsample_bytree_s:
    #                         params['colsample_bytree'] = colsample_bytree
    #                         for eta in eta_s:
    #                             params['eta'] = eta
    auc, p, t = init_xgb(data, params,data_un)
    # if auc > best:
    #     best = auc
    #     p_m = p
    #     t_ = t
    #
    # logger.info("best result is  {0:.5f} ,params {1},key {}".format(best, best, t_))
