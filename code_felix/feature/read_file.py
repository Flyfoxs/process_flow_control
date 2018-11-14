from code_felix.feature.config import *
from code_felix.utils_.util_log import logger
import pandas as pd
from code_felix.utils_.util_cache_file import *
import os
from code_felix.utils_.util_pandas import *
import numpy as np


from functools import lru_cache
@lru_cache()
def get_file_order():
    from code_felix.feature.config import file_order
    file_sn = dict(zip(file_order, range(0, len(file_order))))
    file_sn_list = sorted(file_sn.items(), key=lambda val: val[1])
    logger.debug(file_sn_list)
    return file_sn


@file_cache()
def get_raw_input():
    rootdir = ['./input/fix_interval/生产参数记录表（固定时间间隔）-2018年4月' ,
               './input/fix_interval/生产参数记录表（固定时间间隔）-2018年5月' ,
               ]

    df_month_list = []
    drop_time = False
    for cur_dir, month in zip(rootdir, [4, 5]):
        list = os.listdir(cur_dir)
        list = [f'{cur_dir}/{item}' for item in list if 'csv' in item]
        #logger.debug(f'=====File in one of the folder#{month}:{list}')
        #logger.debug(month)
        df_list = []
        for file in list:
           # logger.debug(file)
            f = open(file)
            file_name = os.path.basename(file)
            logger.debug(f'Get {file_name} base on {file}')
            file_sn_dict = get_file_order()
            file_sn = file_sn_dict[file_name]
            df = pd.read_csv(f, header=None, )
            df.columns = ['time_sn', 'time', f'val#{str(file_sn).rjust(2,"0")}']
            #df['month'] = month
            if month==5:
                df.time_sn = df.time_sn+518400
            df.set_index('time_sn',inplace=True)

            if drop_time:
                df.drop(['time'], axis=1, inplace=True)
            drop_time = True

            df_list.append(df)
            #logger.debug(df.shape)
        one_month = pd.concat(df_list, axis=1)

        df_month_list.append(one_month)
    all = pd.concat(df_month_list)
    all.sort_index(axis=1, inplace=True)
    return all

def get_report():
    report1 = pd.read_csv('./input/2.产品检验报告/产品检验报告2018-4-1.csv')
    report2 = pd.read_csv('./input/2.产品检验报告/产品检验报告2018-5-1-sample.csv')
    report1['month'] = 4
    report2['month'] = 5
    report = pd.concat([report1,report2])
    report['report_time_begin'] = report.apply(lambda row: get_report_time(row)[0], axis=1)
    report['report_time_end'] = report.apply(lambda row: get_report_time(row)[1], axis=1)
    report['time_sn'] = report.report_time_end.apply(lambda val: get_time_sn(val))
    report.set_index('time_sn', inplace=True)
    return report

@timed()
def get_feature(hours=4):
    raw = get_raw_input()
    if 'time' in raw:
        del raw['time']

    logger.debug(raw.shape)
    report = get_report()
    logger.debug(f"The shape of the report is {report.shape}")

    from code_felix.feature.config import time_interval
    gap = 3600*hours//time_interval

    columns = raw.columns
    final_columns = []
    for i in range(0, gap):
        final_columns.extend([f'{item}#{i}' for item in columns])

    feature = np.zeros(( len(report), len(final_columns)))


    for i, end in enumerate(report.index):
        begin = end - gap + 1
        #logger.debug(f'{begin}:{end}')
        #logger.debug(raw.loc[begin:end, :].values.shape)
        feature[i] = np.round(raw.loc[begin:end, :].values.flatten(),6)


    return pd.DataFrame(feature, columns=final_columns,index=report.index)


def get_report_time(row):
    year = 2018
    month_day = row.product_batch.split(' ')[0].replace('.', '-')
    begin_time = row.product_batch.split(' ')[-3]
    time_end = int(row.product_batch.split(' ')[-1].split(':')[0]) or 24
    time_begin = int(row.product_batch.split(' ')[-3].split(':')[0])

    gap = time_end  - time_begin

    return pd.to_datetime(f'{year}-{month_day} {begin_time}') , \
           pd.to_datetime(f'{year}-{month_day} {begin_time}') + np.timedelta64(gap, 'h'), gap


def get_train(hours_gap):
    return get_feature(hours_gap)[:144]

def get_test(hours_gap):
    return get_feature(hours_gap)[144:]


def get_label(type='phosphorus_content'):
    """
    phosphorus_content	nitrogen_content	total_nutrient	water_content	particle_size
    :param type:
    :return:
    """
    report = get_report()
    return report[pd.notnull(report[type])][type]



def get_time_sn(time):
    time_sn = (pd.to_datetime(time) - pd.to_datetime('2018-04-01 00:00:00') ) / np.timedelta64(5, 's') + 1
    return int(time_sn)

def get_all_file(path):
    logger.debug(f'Try to read file from"{path}')
    file_list = os.listdir(path)
    file_list = [file for file in file_list if '.h5' in file]
    return file_list




if __name__ == '__main__':
    #518400
    print(get_time_sn('2018-04-30 23:59:55'))