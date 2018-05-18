#!/usr/bin/python
# -*- coding: utf-8 -*-
# author:pxz
import os
import sys

base_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(base_path)
import csv
import codecs
import scipy.stats
import numpy as np
from pymongo import MongoClient


def transferData():
    '''
    转换本地csv文件到mongodbh中
    :return:
    '''
    host = '175.102.18.112'
    port = 27018
    db_name = 'tongji_zjj'
    WEB_DATA = 't2'  # web1_vec,web2_vec
    client = MongoClient(host=host, port=port)
    db = client[db_name]
    collection = db[WEB_DATA]
    f = codecs.open("../data/t2_vec.csv", 'r', encoding='utf-8')
    reader = csv.DictReader(f)
    for row in reader:
        collection.save(
            {"id": row["id"].split("'")[1],
             "topicDistribution": row["topicDistribution"].replace("[", "").replace("]", ""),
             "topic_num": row["topic_num"]})
    # {"id": row["id"],


def KLD(row1, row2):
    '''
    求KL散度，其值越小，表示连个分布越接近
    :param row1: 文档向量分布1
    :param row2: 文档向量分布2
    :return: KL散度值
    '''
    r1 = [np.float(i) for i in row1.split(',')]
    r2 = [np.float(j) for j in row2.split(',')]
    KL = scipy.stats.entropy(r1, r2)
    return KL


def distOJLD(row1, row2):
    '''
    求欧几里德距离，其值越小，表示两文档越相似
    :param row1: 文档向量分布1
    :param row2: 文档向量分布2
    :return: 距离值
    '''
    dist = list()
    r1 = row1.split(',')
    r2 = row2.split(',')
    len1 = len(r1)
    len2 = len(r2)
    if len1 <= len2:
        for i in range(len1):
            e = (float(r1[i]) - float(r2[i])) ** 2
            dist.append(e)
    s = sum(dist)
    return 1 / (1 + s ** .5)


def sortId(res):
    '''
    按照值的大小，取前5个
    :param res: 带有文档id的字符串
    :return: 排序后的文档
    '''
    res = res.sort(key=lambda x: float(x.split("@")[0]))
    return res


def getDocSimilarity(host, port, db_name, WEB_DATA1, WEB_DATA2):
    '''
    获取在集合2的所有文档中与集合1某一文档最相近的前5篇文档
    :param host: IP值
    :param port:端口
    :param db_name: 数据库
    :param WEB_DATA1: 集合1，文档分布向量
    :param WEB_DATA2: 集合2，文档分布向量
    :return:
    '''
    client = MongoClient(host=host, port=port)
    db = client[db_name]
    collection2 = db[WEB_DATA1]  # 连接集合1
    collection1 = db[WEB_DATA2]  # 连接集合2
    for item1 in collection1.find().limit(5):
        res = list()
        for item2 in collection2.find().limit(30):
            row1 = item1.get("topicDistribution")
            row2 = item2.get("topicDistribution")
            id1 = item1.get("id")
            id2 = item2.get("id")
            # 欧几里德距离
            # s = distOJLD(row1, row2)
            # s = str(s) + "@" + id1 + "&" + id2
            # res.append(s)
            # KL散度
            KLD(row1, row2)
            kl = KLD(row1, row2)
            kl = str(kl) + "@" + id1 + "&" + id2
            res.append(kl)
        res = sortId(res)
        print(res[:5])


if __name__ == "__main__":
    pass
    # transferData()
    host = '175.102.18.112'
    port = 27018
    db_name = 'tongji_zjj'
    WEB_DATA1 = 'web1_vec15'  # web1_vec30, test_web1
    WEB_DATA2 = 'web2_vec15'
    getDocSimilarity(host, port, db_name, WEB_DATA1, WEB_DATA2)
