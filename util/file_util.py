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

def load_raw_csv(file_name):
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
        if pd_col.isnull().sum()/len(pd_col) ==1:
            left_cols.append(col)
        if pd_col.isnull().sum() >0:
            logger.warning("Raw,name:{}-type:{}-uni_count:{}-nan_count:{}-ratos:{}".format(col,type,len(uni_value),
                                                                                           pd_col.isnull().sum(),
                                                                                           1.0 *pd_col.isnull().sum()/len(pd_col)))
        logger.warning("Unique,name:{} , type: {}, count:{} , value:{}".format(col,cx,len(uni_value),uni_value[:6]))
    logger.warning("size = {},delete_cols {}".format(len(delete_cols),delete_cols))
    logger.warning("size = {},cat_cols :{}".format(len(cat_cols),cat_cols))
    logger.warning("size = {},float_cols :{}".format(len(float_value_cols),float_value_cols))
    return data_frame[left_cols]
    # logger.info(data_frame.head(5))

def save_file(df,file_name):
    df.to_csv(file_name,index=False)





if __name__ == '__main__':
    force_data = load_raw_csv("../data/application_train.csv")
    logger.info(force_data.head(5))
    save_file(force_data,"../data/application_train_forced.csv")


