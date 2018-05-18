#!/usr/bin/python
# -*- coding: utf-8 -*-
# author:pxz
import os

os.environ["PYSPARK_PYTHON"] = "/usr/bin/python3.5"
from conf import settings
from modules.dataOperator import DataOperator
from modules import sentence
from textRank4ZH import testText
from modules import sentence
from docDistribution import docLda, docSimilarity

host = settings.host
port = settings.port

# 设置数据库
DB_NAME = settings.LDA_DATABASES_NAME['DB_NAME']
# 设置数据表
WEB_DATA = settings.TABLE_NAME['WEB_DATA']
WEB_DATA1 = settings.TABLE_NAME['WEB_DATA1']
WEB_DATA2 = settings.TABLE_NAME['WEB_DATA2']
ALL_WEB_DATA = settings.TABLE_NAME['ALL_WEB_DATA']
ABS_DATA = settings.TABLE_NAME['ABS_DATA']
WEB1_VEC = settings.TABLE_NAME['WEB1_VEC']
WEB2_VEC = settings.TABLE_NAME['WEB2_VEC']


def run():
    pass
    # db = DataOperator(host, port, DB_NAME, WEB_DATA, ALL_WEB_DATA)
    # db.dataTransfer(WEB_DATA, ALL_WEB_DATA)  # 数据转换
    # db.exec_lda(host, port, DB_NAME, ALL_WEB_DATA)  # 执行lda算法
    # sentence.analyzeSent(host, port, DB_NAME, ALL_WEB_DATA)  # 执行句子拆分分析
    # testText.sentence_key(host, port, DB_NAME, ALL_WEB_DATA, ABS_DATA)  # 执行文档取摘要
    # docLda.sumLDA(host, port, DB_NAME, WEB_DATA, WEB_DATA1, WEB_DATA2) # 求文档主题分布
    docSimilarity.getDocSimilarity(host, port, DB_NAME, WEB1_VEC, WEB2_VEC)  # 求文档相似度


if __name__ == "__main__":
    run()
