# -*- coding: utf-8 -*-

# Define here the models for your scraped items
#
# See documentation in:
# https://doc.scrapy.org/en/latest/topics/items.html
# 数据项

import scrapy


# 地区
class AreaItem(scrapy.Item):
    # define the fields for your item here like:
    # name = scrapy.Field()
    code = scrapy.Field()
    name = scrapy.Field()
    pass


# 股票信息
class StockItem(scrapy.Item):
    code = scrapy.Field()           # 编码
    name = scrapy.Field()           # 名称
    type = scrapy.Field()           # 类别
    east_money_detail_url = scrapy.Field()     # 东方财富详情页地址


# 股票日行情
class StockDateItem(scrapy.Item):
    date = scrapy.Field()           # 日期
    code = scrapy.Field()           # 编码
    name = scrapy.Field()           # 名称
    tc_lose = scrapy.Field()        # 收盘价
    high = scrapy.Field()           # 最高价
    low = scrapy.Field()            # 最低价
    t_open = scrapy.Field()         # 开盘价
    lc_lose = scrapy.Field()        # 前收盘
    chg = scrapy.Field()            # 涨跌额
    p_chg = scrapy.Field()           # 涨跌幅
    vo_turnover = scrapy.Field()     # 成交量
    pass

