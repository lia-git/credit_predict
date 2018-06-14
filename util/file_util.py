#! /usr/bin/env python
# -*- coding:utf-8 -*- 

import logging
import sys

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

formatter = logging.Formatter('%(asctime)s - %(filename)s - [line:%(lineno)d] - %(levelname)s - %(message)s')
file_handler = logging.FileHandler("../log/run.log")
file_handler.setFormatter(formatter)
console_handler = logging.StreamHandler(sys.stdout)
console_handler.formatter = formatter
logger.addHandler(file_handler)
logger.addHandler(console_handler)
logger.setLevel(logging.INFO)

def load_raw_csv(file_name,target=False):
    data_frame = pd.read_csv(file_name,header = 0)
    logger.info(" {} features :{}".format(len(data_frame.columns),data_frame.columns.tolist()))
    delete_cols = []
    left_cols = []
    float_value_cols = []
    cat_cols = []
    for col in data_frame.columns:
        attr_info = {"name":col}
        pd_col = data_frame[col]
        uni_value = pd_col.unique().tolist()
        if  np.object == pd_col.dtypes:
            if len(uni_value) == 1:
                delete_cols.append(col)
            else:
                cat_cols.append(col)
                type = "category"
        if  np.int64 == pd_col.dtypes \
                or np.int32 == pd_col.dtypes \
                or np.float == pd_col.dtypes \
                or np.float64 == pd_col.dtypes:
            if len(uni_value) == 1:
                delete_cols.append(col)
            else:
                if len(uni_value) <4:
                    cat_cols.append(col)
                    type = "category"

                else:
                    float_value_cols.append(col)
                    type= "continues"
        cx= pd_col.dtypes
        if pd_col.isnull().sum()/len(pd_col) ==0 and col != "TARGET":
            left_cols.append(col)
        if pd_col.isnull().sum() >0:
            logger.warning("Raw,name:{}-type:{}-uni_count:{}-nan_count:{}-ratos:{}".format(col,type,len(uni_value),
                                                                                           pd_col.isnull().sum(),
                                                                                           1.0 *pd_col.isnull().sum()/len(pd_col)))
        logger.warning("Unique,name:{} , type: {}, count:{} , value:{}".format(col,cx,len(uni_value),uni_value[:6]))
    logger.warning("size = {},delete_cols {}".format(len(delete_cols),delete_cols))
    logger.warning("size = {},cat_cols :{}".format(len(cat_cols),cat_cols))
    logger.warning("size = {},float_cols :{}".format(len(float_value_cols),float_value_cols))
    if target:
        left_cols.append("TARGET")
    return data_frame[left_cols]
    # logger.info(data_frame.head(5))

def save_file(df,file_name):
    df.to_csv(file_name,index=False)


def merge(df1,df2,left_on,right_on, del_=None,how ="inner"):
    if left_on == right_on:
        merged = pd.merge(df1,df2,on= left_on,how = how)
    else:
        merged = pd.merge(df1,df2,left_on= left_on,right_on=right_on,how = how)
    if del_ is not None:
        del merged[del_]
    del merged[right_on]
    logger.info(merged.head(5))
    logger.info(merged.columns)
    return merged


if __name__ == '__main__':
    force_data = load_raw_csv("../data/application_train.csv",target=True)
    force_bureau_one = load_raw_csv("../data/bureau.csv")
    force_bureau_balance = load_raw_csv("../data/bureau_balance.csv")
    force_credit_balance = load_raw_csv("../data/credit_card_balance.csv")
    force_installments_patments = load_raw_csv("../data/installments_payments.csv")
    force_POS_CASH = load_raw_csv("../data/POS_CASH_balance.csv")
    force_previous = load_raw_csv("../data/previous_application.csv")
    force_bureau = merge(force_bureau_one,force_bureau_balance,"SK_ID_BUREAU","SK_ID_BUREAU")
    force_p1 = merge(force_data,force_bureau,"SK_ID_CURR","SK_ID_CURR","SK_ID_BUREAU")
    force_p2 = merge(force_p1,force_credit_balance,"SK_ID_CURR","SK_ID_CURR","SK_ID_PREV")
    force_p3 = merge(force_p2,force_installments_patments,"SK_ID_CURR","SK_ID_CURR","SK_ID_PREV")
    force_p4 = merge(force_p3,force_POS_CASH,"SK_ID_CURR","SK_ID_CURR","SK_ID_PREV")
    force_all = merge(force_p4,force_previous,"SK_ID_CURR","SK_ID_CURR","SK_ID_PREV")

    logger.info(force_all.head(5))
    save_file(force_data,"../data/forced_all.csv")
    # save_file(force_data,"../data/application_train_forced.csv")


