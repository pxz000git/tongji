#!/usr/bin/python
# -*- coding: utf-8 -*-
# author:pxz

host = '175.102.18.112'
port = 27018

# LDA相关数据库名
LDA_DATABASES_NAME = {
    "DB_NAME": "tongji_zjj"  # 数据库
}
# 数据表名
TABLE_NAME = {
    'WEB_DATA': 'lda_sum_data',  # 数据源 网页抓取的数据，共15584条
    'ALL_WEB_DATA': 'all_web',  # 数据源综合表，共15767条
    'ABS_DATA': 'abs_data',  # 摘要数据表
    'WEB1_VEC': 'web1_vec15',  # 集合1主题分布向量
    'WEB2_VEC': 'web2_vec15',  # 集合2主题分布向量
}
# 数据库表
# LDA_TABLE_NAME = {
#     "FUJIAN": "fujian_new", # 抓取的网页数据
#     "SHANXI": "shanxi_new", # 抓取的网页数据
#     "ZHEJIANG": "zhejiang", # 抓取的网页数据
#     "INFO": "info_web", # 收集的网页数据
#     "ABS": "abs_data", # 抓取的网页数据,的摘要
#     "LAD_SUM": "lda_sum_data", # 抓取的网页数据综合表，共15584条
#     "ALL_WEB": "all_web", # 集合1(抓取的网页数据),集合2(收集的网页数据)数据源综合表，共15767条
# }
