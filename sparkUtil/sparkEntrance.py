#!/usr/bin/python
# -*- coding: utf-8 -*-
# author:pxz
from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession

# 获取spark执行入口
conf = SparkConf().setMaster("local[2]").setAppName("Similarity")
sc = SparkContext(conf=conf)
spark = SparkSession(sc)