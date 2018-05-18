#!/usr/bin/python
# -*- coding: utf-8 -*-
# author:pxz
import os
import sys

base_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(base_path)
os.environ["PYSPARK_PYTHON"] = "/usr/bin/python"
from util import util
from sparkUtil.sparkEntrance import spark
from pymongo import MongoClient
from pyspark.ml.clustering import KMeans
from pyspark.ml.feature import Word2Vec


class SentenceSegmentation(object):
    """ 分句 """

    def __init__(self, delimiters=util.sentence_delimiters):
        """
        Keyword arguments:
        delimiters -- 可迭代对象，用来拆分句子
        """
        self.delimiters = set([util.as_text(item) for item in delimiters])

    def segment(self, text):
        res = [util.as_text(text)]

        util.debug(res)
        util.debug(self.delimiters)

        for sep in self.delimiters:
            text, res = res, []
            for seq in text:
                res += seq.split(sep)
        res = [s.strip() for s in res if len(s.strip()) > 0]
        return res


class Segmentation(object):
    def __init__(self, delimiters=util.sentence_delimiters):
        """
        Keyword arguments:
        stop_words_file -- 停止词文件
        delimiters      -- 用来拆分句子的符号集合
        """
        self.ss = SentenceSegmentation(delimiters=delimiters)

    def segment(self, text, lower=False):
        '''
        :param text:
        :param lower:
        :return:
        '''
        text = util.as_text(text)
        sentences = self.ss.segment(text)
        return util.AttrDict(sentences=sentences, )


def load_data(host, port, db_name, ALL_WEB_DATA):
    client = MongoClient(host=host, port=port)
    db = client[db_name]  # 链接db_name数据库
    collect = db[ALL_WEB_DATA]  # 使用lda_sum_data集合
    # text = list()
    projectionFields = {'_id': False, 'fileContent': True}
    doc = list()
    for i in collect.find({}, projection=projectionFields).limit(200):
        if len(i) == 0:
            continue
        doc.append(i['fileContent'])
    return doc


def senSplit(host, port, db_name, ALL_WEB_DATA):
    '''
    拆分句子
    :param host: IP
    :param port: 端口
    :param db_name: 数据库
    :param ALL_WEB_DATA: 数据源
    :return: 返回拆分句子后的列表
    '''
    # text = "视频里，我们的杰宝热情地用英文和全场观众打招呼并清唱了一段《Heal The World》。我们的世界充满了未知数。"
    data = load_data(host, port, db_name, ALL_WEB_DATA)
    s = ''
    for i in data:
        s += i
    seg = Segmentation()
    result = seg.segment(text=s, lower=True)
    sen_list = list()
    for s in result['sentences']:
        sen_list.append([s])
    return sen_list


def analyzeSent(host, port, db_name, ALL_WEB_DATA):
    '''
    对拆分的句子进行聚类分析
    :return:
    '''
    sen = senSplit(host, port, db_name, ALL_WEB_DATA)
    df = spark.createDataFrame(sen)

    def tokenizer(row):
        '''
        文档映射处理
        :param row:
        :return:
        '''
        result = list()
        row_sp = ''.join(row.sent.split())
        result.append(row_sp)
        return [result]

    rdd = df.selectExpr('_1 as sent').rdd.map(tokenizer)
    df = spark.createDataFrame(rdd, ['sent'])
    wv_df = Word2Vec(vectorSize=5, minCount=0, inputCol="sent", outputCol="features")
    model_wv = wv_df.fit(df)
    wv_df = model_wv.transform(df)
    # model_wv.getVectors().show(truncate=False)
    km = KMeans(featuresCol="features", k=5)
    model_km = km.fit(wv_df)
    df_km = model_km.transform(wv_df)
    df_km.select('sent', 'prediction').show()
    df_km.select('sent', 'prediction').show(truncate=False)
    # model_km.transform(wv_df).show(truncate=False)


if __name__ == "__main__":
    host = '175.102.18.112'
    port = 27018
    db_name = "tongji_zjj"
    ALL_WEB_DATA = "lda_sum_data"
    analyzeSent(host, port, db_name, ALL_WEB_DATA)
