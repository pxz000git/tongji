#!/usr/bin/python
# -*- coding: utf-8 -*-
# author:pxz
import os
import sys

base_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(base_path)
os.environ["PYSPARK_PYTHON"] = "/usr/bin/python3.5"
import re
from sparkUtil import sparkEntrance
from pyspark.ml.clustering import LDA, KMeans, LocalLDAModel, DistributedLDAModel
from pyspark.ml.feature import CountVectorizer, HashingTF, IDF, Word2Vec

try:
    import jieba
except Exception as e1:
    os.system('pip install jieba')
    import jieba


def get_default_file():
    #  获取根目录
    d = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    return os.path.join(d, 'data/stopwords.txt')


def stop_words():
    '''
    :return: 无序不重复的停用词表
    '''
    stopwords = get_default_file()
    with open(stopwords, 'r') as f:
        words = f.read().split('\n')
    return list(set(words))


stops = stop_words()


def tokenizer(row):
    '''
    文档分词,只匹配中文
    :param row:
    :return:
    '''
    result = list()
    regex = r'^[\u4e00-\u9fa5_a-zA-Z]+$'
    row_sp = ''.join(row.content.split())
    row_cuts = jieba.cut(row_sp)
    for row_cut in row_cuts:
        if len(row_cut) < 2:  # 过滤长度小于2的词
            continue
        if row_cut not in stops and row_cut != '\r' and '\n' and re.match(regex, row_cut):
            result.append(row_cut)
    return [result]


def df_operator(df):
    '''
    对dataframe字段进行操作
    :param df:
    :return:
    '''
    df = df.selectExpr('html as content').limit(20).distinct()
    df = df.na.drop(how='any', subset='content')
    rdd = df.rdd.map(tokenizer)
    df = sparkEntrance.spark.createDataFrame(rdd, ['content']).cache()
    return df


def tfidf_lda(df):
    '''
    TFIDF+LDA
    :param df:
    :return: model
    '''
    # hashingTF
    hashingTF = HashingTF(inputCol="content", outputCol="features")
    df_TF = hashingTF.transform(df)
    print('df_TF')
    df_TF.show(truncate=False)
    # IDF
    idf = IDF(inputCol="features", outputCol="idf")
    model_idf = idf.fit(df_TF)
    df_idf = model_idf.transform(df_TF)
    print('df_idf')
    df_idf.cache()
    df_idf.show(truncate=False)
    # LDA
    lda = LDA(k=20, seed=1, optimizer="em")
    model_lda = lda.fit(df_idf)
    model_lda.describeTopics(maxTermsPerTopic=20)
    df_lda = model_lda.transform(df_idf)
    df_lda.select("content", "topicDistribution").show(truncate=False)
    sparkEntrance.spark.createDataFrame(df_lda.rdd, ['content', 'topicDistribution'])


def cv_lda(df):
    '''
    CountVectorizer, LDA
    :param df:
    :return:
    '''
    # CountVectorizer
    cv = CountVectorizer(inputCol="content", outputCol="features")
    model_cv = cv.fit(df)
    vocabulary = model_cv.vocabulary
    df_cv = model_cv.transform(df)
    df_cv.cache()
    df_cv.show(truncate=False)

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
    lda = LDA(k=20, seed=1, optimizer="em")
    model_lda = lda.fit(df_cv)
    # 设置主题关键字数maxTermsPerTopic
    df_des = model_lda.describeTopics(maxTermsPerTopic=20)
    rdd_des = df_des.select("topic", "termIndices", "termWeights", df_des.termIndices).rdd.map(getwords)
    df_des = sparkEntrance.spark.createDataFrame(rdd_des, ['topic', 'termIndices', 'termWeights', 'words'])
    # return df_des
    df_des.select('topic', 'words', 'termWeights').show(truncate=False)
    dd = df_des.select('topic', 'words', 'termWeights')
    # 默认保存parquet格式文件
    dd.write.save("file:///home/pxz/topic2/html_data")


def cv_idf_lda(df):
    '''
    CountVectorizer, IDF, LDA
    :param df:
    :return:
    '''
    pass
    # CountVectorizer
    cv = CountVectorizer(inputCol="content", outputCol="features")
    model_cv = cv.fit(df)
    vocabulary = model_cv.vocabulary
    df_cv = model_cv.transform(df)
    df_cv.cache()
    # # IDF
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
    lda = LDA(k=10, seed=1, optimizer="em")
    model_lda = lda.fit(df_idf)
    print(model_lda.describeTopics(maxTermsPerTopic=10))
    # model_lda.save('file:///home/pxz/model_lda/lda15')
    # print(lda[df_idf])
    # print(type(lda[df_idf]))
    # # 主题数maxTermsPerTopic
    df_des = model_lda.describeTopics(maxTermsPerTopic=15)
    rdd_des = df_des.select("topic", "termIndices", "termWeights", df_des.termIndices).rdd.map(getwords)
    df_des = sparkEntrance.spark.createDataFrame(rdd_des, ['topic', 'termIndices', 'termWeights', 'words'])
    # return df_des
    df_des.select('topic', 'words', 'termWeights').show(truncate=False)
    # dd = df_des.select('topic', 'words', 'termWeights')
    # 默认保存parquet格式文件
    # dd.write.save("file:///home/pxz/cv_idf_lda/info_web_data")


# 聚合操作
def createCombiner(value):
    return value


def mergeValue(acc, value):
    return acc + value


def mergeCombiners(acc1, acc2):
    return acc1 + acc2


# def tokenizer_l(row):
#     result = list()
#     for i in row.detail.split(','):
#         result.append(i)
#     return [row.cla, result]


# def wv_kmeans():
#     '''
#     对主题词进行聚类
#     :return:
#     '''
#     # 加载文档数据
#     df = sparkEntrance.spark.read.format('parquet').load('file:///home/pxz/cv_idf_lda/info_web_data/*.parquet')
#     # df.select('topic', 'words', 'termWeights').show(truncate=False)
#     # 加载列表清单数据
#     # df_l = sparkEntrance.spark.read.csv('file:///root/Documents/rule_list.csv', header=True)
#     # df_l.selectExpr('model_detail as detail').show(truncate=False)
#     # rdd_l = df_l.selectExpr('model_cla as cla', 'model_detail as detail').rdd.map(tokenizer_l).combineByKey(
#     #     createCombiner, mergeValue, mergeCombiners)
#     # df_l = sparkEntrance.spark.createDataFrame(rdd_l, ['cla', 'detail'])
#     #
#     # word2Vec_l = Word2Vec(minCount=1, vectorSize=5, inputCol='detail', outputCol="features")
#     # model_wv_l = word2Vec_l.fit(df_l)
#     # wv_l_df = model_wv_l.transform(df_l)
#     # model_wv_l.getVectors().show(truncate=False)
#     rdd = df.select('topic', 'words').rdd.combineByKey(createCombiner, mergeValue, mergeCombiners)
#     df = sparkEntry.spark.createDataFrame(rdd, ['topic', 'words'])
#     # 向量维数为5，一般要设置到几百维，然后使用降维操作
#     word2Vec = Word2Vec(vectorSize=10, inputCol="words", outputCol="features")
#     model_wv = word2Vec.fit(df)
#     wv_df = model_wv.transform(df)
#     # wv_df.show(truncate=False)
#     km = KMeans(featuresCol="features", k=5)
#     model_km = km.fit(wv_df)
#     km_df = model_km.transform(wv_df).select("words", "prediction")
#     km_df.show(truncate=False)


if __name__ == "__main__":
    pass
    # wv_kmeans()
