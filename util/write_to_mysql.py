#! /usr/bin/env python
# -*- coding:utf-8 -*- 

import logging
import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine

logger = logging.getLogger(__name__)

formatter = logging.Formatter('%(asctime)s - %(filename)s - [line:%(lineno)d] - %(levelname)s - %(message)s')
file_handler = logging.FileHandler("../log/run.log")
file_handler.setFormatter(formatter)
console_handler = logging.StreamHandler(sys.stdout)
console_handler.formatter = formatter
logger.addHandler(file_handler)
logger.addHandler(console_handler)
logger.setLevel(logging.INFO)



def generate_create_sql(file_name,name,key):
    sql_str = "CREATE TABLE {} ( \n".format(name)
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
                add_str = "{} VARCHAR(100) NOT NULL ,\n".format(str(col))
                type = "category"
        if  np.int64 == pd_col.dtypes \
                or np.int32 == pd_col.dtypes \
                or np.float == pd_col.dtypes \
                or np.float64 == pd_col.dtypes:
            if np.int64 == pd_col.dtypes or np.int32 == pd_col.dtypes:
                add_str = "{} INT(10)  NOT NULL ,\n".format(str(col))
            if np.float == pd_col.dtypes or np.float64 == pd_col.dtypes:
                add_str = "{} double(12,6) NOT NULL ,\n".format(str(col))

            if len(uni_value) == 1:
                delete_cols.append(col)
            else:
                if len(uni_value) <7:
                    cat_cols.append(col)
                    type = "category"
                    add_str = "{} VARCHAR(100) NOT NULL ,\n".format(str(col))


                else:
                    float_value_cols.append(col)
                    type= "continues"
        # if type == "continues":
        sql_str +=  add_str
    end_str = "PRIMARY KEY ( {}) \n) ENGINE=InnoDB DEFAULT CHARSET=utf8;".format(key)
    sql_str += end_str

    print(sql_str)
    print()
        #
        # cx= pd_col.dtypes
        # if pd_col.isnull().sum()/len(pd_col) ==0 and col != "TARGET":


def insert_mysql(file_name,name):
    data_frame = pd.read_csv(file_name,header = 0,index_col=None)
    print(data_frame.head(5))
    # data_frame.reset_index(drop = True, inplace = True)
    print(data_frame.head(5))

    data_frame.to_sql(name=name,con=engine,if_exists='append',index=False)


if __name__ == '__main__':
    engine = create_engine("mysql+pymysql://root:hemei@ai@192.168.1.97/question_s")
    # generate_create_sql("../data/force_previous_application.csv","previous_application","")
    insert_mysql("../data/force_previous_application.csv","previous_application")