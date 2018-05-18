#!/usr/bin/python
# -*- coding: utf-8 -*-
# author:pxz

from modules import cluster
from sparkUtil import sparkEntrance
import pymongo as pm


class DataOperator:
    def __init__(self, host, port, db_name, WEB_DATA, ALL_WEB_DATA):
        '''
        设置mongodb的地址，端口以及默认访问的集合，后续访问中如果不指定collection，则访问这个默认的
        :param host: 地址
        :param port: 端口
        :param db_name: 数据库
        :param WEB_DATA: 某网站爬虫数据表
        :param ALL_WEB_DATA: 所有网站爬虫数据表
        '''
        self.host = host
        self.port = port
        self.WEB_DATA = WEB_DATA
        self.ALL_WEB_DATA = ALL_WEB_DATA
        # 建立数据库连接
        self.client = pm.MongoClient(host=host, port=port)
        # 选择相应的数据库名称
        self.db_name = self.client.get_database(db_name)

    def dataTransfer(self, WEB_DATA, ALL_WEB_DATA):
        '''
        把WEB_DATA,WEB_DATA2集合中的数据转换到ALL_WEB_DATA集合中
        :param WEB_DATA:
        :param ALL_WEB_DATA:
        :return:
        '''
        collection1 = self.db_name.get_collection(WEB_DATA)
        collection2 = self.db_name.get_collection(ALL_WEB_DATA)
        for item in collection1.find():
            print(item.get('title'))
            collection2.save(
                {'title': item.get('title'), 'time': item.get('publicTime'), 'html': item.get('html'),
                 'org': item.get('puborg')})

    def load_data(self, host, port, db_name, ALL_WEB_DATA):
        '''
        从mongodb中加载数据
        :param host: ip
        :param port: 端口
        :param db_name: 数据库
        :param ALL_WEB_DATA: 数据表
        :return: dataframe
        '''
        df = sparkEntrance.spark.read.format("com.mongodb.spark.sql.DefaultSource") \
            .option("spark.mongodb.input.uri",
                    "mongodb://" + host + ":" + str(port) + "/" + db_name + '.' + ALL_WEB_DATA) \
            .load()
        df = cluster.df_operator(df)
        return df

    def exec_lda(self, host, port, db_name, ALL_WEB_DATA):
        '''
        执行算法
        :param host:
        :param port:
        :param db_name:
        :param ALL_WEB_DATA:
        :return:
        '''
        pass
        # 执行 CountVectorizer, LDA
        # cluster.cv_lda(self.load_data(host, port, db_name, ALL_WEB_DATA))
        # 执行 TFIDF+LDA
        # cluster.tfidf_lda(self.load_data(host, port, db_name, ALL_WEB_DATA))
        # 执行 CountVectorizer, IDF, LDA
        cluster.cv_idf_lda(self.load_data(host, port, db_name, ALL_WEB_DATA))


if __name__ == "__main__":
    pass
