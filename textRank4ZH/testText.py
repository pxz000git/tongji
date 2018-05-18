#!/usr/bin/python
# -*- coding: utf-8 -*-
# author:pxz
import os
import sys

base_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(base_path)
import codecs
from pymongo import MongoClient
from textRank4ZH import TextRank4Sentence


def load_data(host, port, db_name, SOUR_DATA):
    '''
    从数据库获取指定数据列
    :param host: IP
    :param port: 端口
    :param db_name: 数据库
    :param SOUR_DATA: 集合
    :return: 返回获取的指定数据列的列表
    '''
    # host = '175.102.18.112'
    # port = 27018
    # db_name = "tongji_zjj"
    # SOUR_DATA = "lda_sum_data"  # lda_sum_data, info_web
    client = MongoClient(host=host, port=port)
    db = client[db_name]  # 链接db_name数据库
    collect = db[SOUR_DATA]  # 使用SOUR_DATA集合
    projectionFields = {'_id': False, 'html': True}
    doc = list()
    for i in collect.find({}, projection=projectionFields).limit(5):
        if len(i['html']) == 0:
            continue
        doc.append(i['html'])
    return doc


# text = codecs.open('../data/doc/02.txt', 'r', 'utf-8').read()


def sentence_key(host, port, db_name, SOUR_DATA, WEB_DATA):
    '''
    把数据经过取摘要后以列表形式存入数据库
    :param host: IP
    :param port: 端口
    :param db_name: 数据库
    :param SOUR_DATA: 源数据集合 # lda_sum_data, info_web
    :param WEB_DATA: 结果数据集合
    :return: NUll
    '''
    data = load_data(host, port, db_name, SOUR_DATA)
    client = MongoClient(host=host, port=port)
    db = client[db_name]  # 链接db_name数据库
    collect = db[WEB_DATA]  # 使用ALL_WEB_DATA集合
    for text in data:
        sen = list()
        print('**************' + '摘要：' + '**************')
        tr4s = TextRank4Sentence.TextRank4Sentence()
        tr4s.analyze(text=text, lower=True, source='all_filters')
        for item in tr4s.get_key_sentences(num=3):
            print(item.index, item.weight, item.sentence)
            sen.append(item.sentence)
        collect.save({'html': sen})  # 保存到数据库tongji_zjj.abs_data


if __name__ == "__main__":
    host = '175.102.18.112'
    port = 27018
    db_name = "tongji_zjj"
    SOUR_DATA = "lda_sum_data"  # lda_sum_data, info_web
    WEB_DATA = "abs_data"  # abs_data
    sentence_key(host, port, db_name, SOUR_DATA, WEB_DATA)
