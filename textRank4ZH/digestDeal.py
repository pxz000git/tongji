#!/usr/bin/python
# -*- coding: utf-8 -*-
# author:pxz
import os
import sys

base_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(base_path)
import re
import jieba
from pyspark import SparkConf, SparkContext
from pyspark.ml.clustering import KMeans
from pyspark.ml.feature import Word2Vec
from pyspark.sql import SparkSession


def get_default_file():
    #  获取根目录
    d = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    return os.path.join(d, 'data/zhaiyao.txt')


def stop_words():
    '''
    :return: 无序不重复的停用词表
    '''
    stopwords = os.path.join(base_path, 'data/stopwords.txt')
    with open(stopwords, 'r') as f:
        words = f.read().split('\n')
    return list(set(words))


def digest():
    '''
    对摘要进行分词/取词向量聚类分析
    :return:
    '''
    # 获取spark程序入口
    conf = SparkConf().setMaster('local').setAppName("sentence")
    sc = SparkContext(conf=conf)
    spark = SparkSession(sc)
    df = spark.read.text('file://' + get_default_file())

    def tokenizer(row):
        '''
        文档分词,只匹配中文
        :param row:
        :return:
        '''
        row_s = row.content.strip('[]').split(',')
        return [row_s]
        # stops = stop_words()
        # result = list()
        # regex = r'^[\u4e00-\u9fa5_a-zA-Z]+$'
        # row_sp = ''.join(row.content.split())
        # row_cuts = jieba.cut(row_sp)
        # for row_cut in row_cuts:
        #     if len(row_cut) < 2:  # 过滤长度小于2的词
        #         continue
        #     if row_cut not in stops and row_cut != '\r' and '\n' and re.match(regex, row_cut):
        #         result.append(row_cut)
        # return [list(set(result))]

    df = df.selectExpr('value as content')
    df = df.na.drop(how='any', subset='content')
    rdd = df.rdd.map(tokenizer)
    df = spark.createDataFrame(rdd, ['content']).cache()
    # df.show(truncate=False)
    word2Vec = Word2Vec(vectorSize=100, minCount=1, inputCol="content", outputCol="features")
    model_wv = word2Vec.fit(df)
    wv_df = model_wv.transform(df)
    wv_df.show(truncate=False)
    km = KMeans(featuresCol="features", k=10)
    model_km = km.fit(wv_df)
    km_df = model_km.transform(wv_df).select("content", "prediction")
    km_df.show()
    km_df.show(truncate=False)


if __name__ == "__main__":
    digest()
    pass
