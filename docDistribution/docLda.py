#!/usr/bin/python
# -*- coding: utf-8 -*-
# author:pxz
import os
import sys

base_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(base_path)
import re
import jieba
import pandas
from sparkUtil import sparkEntrance
from pyspark.ml.feature import CountVectorizer, IDF
from pyspark.ml.clustering import LDA


def get_default_file():
    #  获取根目录
    d = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    return os.path.join(d, 'data/stopwords.txt')


def stop_words():
    '''
    :return: 无序不重复的停用词表
    '''
    stopwords = get_default_file()  # 获取停用词
    with open(stopwords, 'r') as f:
        words = f.read().split('\n')
    return list(set(words))


stops = stop_words()


def tokenizer(row):
    '''
    文档分词
    :param row:
    :return:
    '''
    result = list()
    regex = r'^[\u4e00-\u9fa5_a-zA-Z]+$'
    # row_sp = ''.join(row.content.split())
    row_sp = ''.join(row.content.split())

    # 匹配文件名，例如：《城市轨道交通工程设计概预算编制办法》
    # 作为一个整体的名词加入词袋
    p = "《.*?》"
    ms = re.findall(p, row_sp)
    for m in ms:
        row_sp = row_sp.replace(m, "")
        tmp = m.replace("《", "").replace("》", "")
        if tmp not in result:
            result.append(tmp)

    # 匹配文件号，例如：（建标[2006]279号）
    # 作为无效词丢弃
    p = '（.*号）'
    ms = re.findall(p, row_sp)
    for m in ms:
        row_sp = row_sp.replace(m, "")

    row_cuts = jieba.cut(row_sp)
    # row_cuts = jieba.cut(row_sp)
    for row_cut in row_cuts:
        if len(row_cut) < 2:  # 过滤长度小于2的词
            continue
        if row_cut not in stops and row_cut != '\r' and '\n' and re.match(regex, row_cut):
            result.append(row_cut)
    return [result, row.id]


def loadData1(host, port, db_name, WEB_DATA1):
    '''
    加载集合1数据
    :param host:
    :param port:
    :param db_name: 数据库
    :param WEB_DATA1: 集合1，lda_sum_data，共15584条数据
    :return: dataframe
    '''
    df1 = sparkEntrance.spark.read.format("com.mongodb.spark.sql.DefaultSource") \
        .option("spark.mongodb.input.uri",
                "mongodb://" + host + ":" + str(port) + "/" + db_name + '.' + WEB_DATA1) \
        .load()
    df1 = df1.selectExpr('html as content', '_id as id').distinct()
    # 删除空值
    # If 'any', drop a row if it contains any nulls.
    # If 'all', drop a row only if all its values are null.
    df1 = df1.na.drop(how='any', subset='content')
    rdd1 = df1.rdd.map(tokenizer)
    # 由content , id创建dataframe
    df1 = sparkEntrance.spark.createDataFrame(rdd1, ['content', 'id']).cache()
    # CountVectorizer
    cv1 = CountVectorizer(inputCol="content", outputCol="features")
    model_cv1 = cv1.fit(df1)
    # 获取词汇值
    # vocabulary1 = model_cv1.vocabulary
    df_cv1 = model_cv1.transform(df1).cache()
    # IDF
    idf1 = IDF(inputCol="features", outputCol="cv")
    model_idf1 = idf1.fit(df_cv1)
    df_idf1 = model_idf1.transform(df_cv1).cache()
    return df_idf1


def loadData2(host, port, db_name, WEB_DATA2):
    '''
    加载集合1数据
    :param host:
    :param port:
    :param db_name: 数据库
    :param WEB_DATA2: 集合2，info_web，共184条数据
    :return: dataframe
    '''
    df2 = sparkEntrance.spark.read.format("com.mongodb.spark.sql.DefaultSource") \
        .option("spark.mongodb.input.uri",
                "mongodb://" + host + ":" + str(port) + "/" + db_name + '.' + WEB_DATA2) \
        .load()
    df2 = df2.selectExpr('html as content', '_id as id').distinct()
    # 删除空值
    # If 'any', drop a row if it contains any nulls.
    # If 'all', drop a row only if all its values are null.
    df2 = df2.na.drop(how='any', subset='content')
    rdd2 = df2.rdd.map(tokenizer)
    # 由content , id创建dataframe
    df2 = sparkEntrance.spark.createDataFrame(rdd2, ['content', 'id']).cache()
    # CountVectorizer
    cv2 = CountVectorizer(inputCol="content", outputCol="features")
    model_cv2 = cv2.fit(df2)
    # 获取词汇值
    # vocabulary2 = model_cv2.vocabulary
    df_cv2 = model_cv2.transform(df2).cache()
    # IDF
    idf2 = IDF(inputCol="features", outputCol="cv")
    model_idf2 = idf2.fit(df_cv2)
    df_idf2 = model_idf2.transform(df_cv2).cache()
    return df_idf2


def sumLDA(host, port, db_name, WEB_DATA, WEB_DATA1, WEB_DATA2):
    '''
    根据all_web数据，由CountVectorizer，IDF，LDA训练模型，然后分别加载lda_sum_data，info_web数据，生成主题分布向量
    :param host:
    :param port:
    :param db_name:
    :param WEB_DATA: all_web
    :param WEB_DATA1: lda_sum_data
    :param WEB_DATA2: info_web
    :return:
    '''
    df = sparkEntrance.spark.read.format("com.mongodb.spark.sql.DefaultSource") \
        .option("spark.mongodb.input.uri",
                "mongodb://" + host + ":" + str(port) + "/" + db_name + '.' + WEB_DATA) \
        .load()
    df = df.selectExpr('html as content', '_id as id').limit(30).distinct()
    df = df.na.drop(how='any', subset='content')
    rdd = df.rdd.map(tokenizer)
    df = sparkEntrance.spark.createDataFrame(rdd, ['content', 'id']).cache()
    # CountVectorizer
    cv = CountVectorizer(inputCol="content", outputCol="features")
    model_cv = cv.fit(df)
    vocabulary = model_cv.vocabulary
    df_cv = model_cv.transform(df)
    df_cv.cache()
    # IDF
    idf = IDF(inputCol="features", outputCol="cv")
    model_idf = idf.fit(df_cv)
    df_idf = model_idf.transform(df_cv)
    df_idf.cache()

    def getwords(row):
        '''
        根据下标，映射得到对应词
        :param row:
        :return:
        '''
        words = list()
        for index in row[3]:
            words.append(vocabulary[index])
        return [row[0], row[1], row[2], words]

    # LDA
    # 训练模型
    lda = LDA(k=10, seed=1, optimizer="em")
    model_lda = lda.fit(df_idf)

    data1_df_idf = loadData1(host, port, db_name, WEB_DATA1)
    data2_df_idf = loadData2(host, port, db_name, WEB_DATA2)
    # 主题数maxTermsPerTopic
    df_des = model_lda.describeTopics(maxTermsPerTopic=10)

    df_lda1 = model_lda.transform(data1_df_idf)
    df_lda2 = model_lda.transform(data2_df_idf)
    # 集合all_web，主题提取
    rdd_des = df_des.select("topic", "termIndices", "termWeights", df_des.termIndices).rdd.map(getwords)
    df_des = sparkEntrance.spark.createDataFrame(rdd_des, ['topic', 'termIndices', 'termWeights', 'words'])
    df_des.select('topic', 'words', 'termWeights').show(truncate=False)

    def tok(row):
        m = row["topicDistribution"].tolist()
        index = m.index(max(m))
        return [row['id'], index, row["topicDistribution"]]

    rdd_topic1 = df_lda1.rdd.map(tok)
    rdd_topic2 = df_lda2.rdd.map(tok)
    # lda_sum_data集合主题分布
    print("*************topicDistribution***1************")
    df_topic1 = sparkEntrance.spark.createDataFrame(rdd_topic1, ["id", "topic_num", "topicDistribution"])
    df_topic1.select("id", "topic_num", "topicDistribution").show(truncate=False)
    # info_web集合主题分布
    print("*************topicDistribution***2************")
    df_topic2 = sparkEntrance.spark.createDataFrame(rdd_topic2, ["id", "topic_num", "topicDistribution"])
    df_topic2.select("id", "topic_num", "topicDistribution").show(truncate=False)
    # 保存主题分布向量
    df_topic1.select("id", "topic_num", "topicDistribution").toPandas().to_csv("./t1_vec.csv", encoding="utf-8",
                                                                               sep=",", header=True)
    df_topic2.select("id", "topic_num", "topicDistribution").toPandas().to_csv("./t2_vec.csv", encoding="utf-8",
                                                                               sep=",", header=True)


if __name__ == "__main__":
    pass
    host = '175.102.18.112'
    port = 27018
    db_name = 'tongji_zjj'
    WEB_DATA = 'all_web'
    WEB_DATA1 = 'lda_sum_data'
    WEB_DATA2 = 'info_web'
    sumLDA(host, port, db_name, WEB_DATA, WEB_DATA1, WEB_DATA2)
