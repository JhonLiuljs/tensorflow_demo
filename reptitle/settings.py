# -*- coding: utf-8 -*-

# Scrapy settings for reptitle project
#
# For simplicity, this file contains only settings considered important or
# commonly used. You can find more settings consulting the documentation:
#
#     https://doc.scrapy.org/en/latest/topics/settings.html
#     https://doc.scrapy.org/en/latest/topics/downloader-middleware.html
#     https://doc.scrapy.org/en/latest/topics/spider-middleware.html

BOT_NAME = 'reptitle'

SPIDER_MODULES = ['reptitle.spiders']
NEWSPIDER_MODULE = 'reptitle.spiders'

#配置MongoDB数据库的连接信息
MONGO_URL = 'localhost'
MONGO_PORT = 27017
MONGO_DB = 'stock'

#参数等于False，就等于告诉你这个网站你想取什么就取什么，不会读取每个网站的根目录下的禁止爬取列表(例如：www.baidu.com/robots.txt）
ROBOTSTXT_OBEY = False

# Configure item pipelines
# See https://doc.scrapy.org/en/latest/topics/item-pipeline.html
ITEM_PIPELINES = {
    'reptitle.pipelines.MongoDBPipeline': 300
}
