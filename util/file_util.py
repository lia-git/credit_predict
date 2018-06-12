#! /usr/bin/env python
# -*- coding:utf-8 -*- 

import logging
import pandas as pd
import numpy as np
from numpy.core.multiarray import dtype

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_csv(file_name):
    data_frame = pd.read_csv(file_name,header = 0)
    logger.info(" {} features :{}".format(len(data_frame.columns),data_frame.columns.tolist()))
    delete_cols = []
    float_value_cols = []
    cat_cols = []
    for col in data_frame.columns:
        pd_col = data_frame[col]
        uni_value = pd_col.unique().tolist()
        if  np.object == pd_col.dtypes:
            if len(uni_value) == 1:
                delete_cols.append(col)
            else:
                cat_cols.append(col)
        if  np.int64 == pd_col.dtypes\
                or np.int32 == pd_col.dtypes \
                or np.float == pd_col.dtypes \
                or np.float64 == pd_col.dtypes:
            if len(uni_value) == 1:
                delete_cols.append(col)
            else:
                if len(uni_value) <4:
                    cat_cols.append(col)
                else:
                    float_value_cols.append(col)

            cx= pd_col.dtypes
            logger.warning("name:{} , type: {}, count:{} , value:{}".format(col,cx,len(uni_value),uni_value[:6]))
    logger.warning("size = {},delete_cols {}".format(len(delete_cols),delete_cols))
    logger.warning("size = {},cat_cols :{}".format(len(cat_cols),cat_cols))
    logger.warning("size = {},float_cols :{}".format(len(float_value_cols),float_value_cols))
    # logger.info(data_frame.head(5))




if __name__ == '__main__':
    load_csv("../data/application_train.csv")


