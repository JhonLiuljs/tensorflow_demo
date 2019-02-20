import scrapy
import random
import re
from reptitle.items import StockItem


class StockSpider(scrapy.Spider):
    name = "stock"
    allowed_domains = ["quote.eastmoney.com", "quotes.money.163.com"]
    start_urls = [
        "http://quote.eastmoney.com/stocklist.html"
    ]
    user_agent_list = [
        "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.1 (KHTML, like Gecko) Chrome/22.0.1207.1 Safari/537.1",
        "Mozilla/5.0 (X11; CrOS i686 2268.111.0) AppleWebKit/536.11 (KHTML, like Gecko) Chrome/20.0.1132.57 Safari/536.11",
        "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/536.6 (KHTML, like Gecko) Chrome/20.0.1092.0 Safari/536.6",
        "Mozilla/5.0 (Windows NT 6.2) AppleWebKit/536.6 (KHTML, like Gecko) Chrome/20.0.1090.0 Safari/536.6",
        "Mozilla/5.0 (Windows NT 6.2; WOW64) AppleWebKit/537.1 (KHTML, like Gecko) Chrome/19.77.34.5 Safari/537.1",
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/536.5 (KHTML, like Gecko) Chrome/19.0.1084.9 Safari/536.5",
        "Mozilla/5.0 (Windows NT 6.0) AppleWebKit/536.5 (KHTML, like Gecko) Chrome/19.0.1084.36 Safari/536.5",
        "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/536.3 (KHTML, like Gecko) Chrome/19.0.1063.0 Safari/536.3",
        "Mozilla/5.0 (Windows NT 5.1) AppleWebKit/536.3 (KHTML, like Gecko) Chrome/19.0.1063.0 Safari/536.3",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_8_0) AppleWebKit/536.3 (KHTML, like Gecko) Chrome/19.0.1063.0 Safari/536.3",
        "Mozilla/5.0 (Windows NT 6.2) AppleWebKit/536.3 (KHTML, like Gecko) Chrome/19.0.1062.0 Safari/536.3",
        "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/536.3 (KHTML, like Gecko) Chrome/19.0.1062.0 Safari/536.3",
        "Mozilla/5.0 (Windows NT 6.2) AppleWebKit/536.3 (KHTML, like Gecko) Chrome/19.0.1061.1 Safari/536.3"]

    def parse(self, response):
        ua = random.choice(self.user_agent_list)
        headers = {
            'Accept-Encoding': 'gzip, deflate, sdch, br',
            'Accept-Language': 'zh-CN,zh;q=0.8',
            'Connection': 'keep-alive',
            'Referer': 'https://gupiao.baidu.com/',
            'User-Agent': ua
        }  # 构造请求头
        for sel in response.selector.xpath('//div[contains(@class, "quotebody")]//div//ul//li'):
            if len(sel.xpath('a/@href').extract()) != 0:
                text = sel.xpath('a/text()').extract()[0]
                href = sel.xpath('a/@href').extract()[0]
                try:
                    code_all = re.findall(r"[s][hz]\d{6}", href)[0]
                    item = StockItem()
                    item["name"] = re.sub(r'\(.*\)', "", text)
                    item["type"] = code_all[0:2]
                    item["code"] = code_all[2:]
                    item["east_money_detail_url"] = href
                    yield item
                except Exception as e:
                    print(e)
                    continue

    def city_item(self, response):
        url = "http://www.stats.gov.cn/tjsj/tjbz/tjyqhdmhcxhfdm/2017/"
        for sel in response.selector.xpath('//table//td'):
            if len(sel.xpath('a/@href').extract()) != 0:
                title = sel.xpath('a/text()').extract()
                link = sel.xpath('a/@href').extract()
                if (not title[0].isdigit()):
                    item = StockItem()
                    code = link[0].split(".")[0].split("/")[1]
                    code += "00000000"
                    item["code"] = code
                    item["name"] = title[0]
                    yield item
                cl = url + link[0]
                yield scrapy.Request(cl, callback=self.county_item)

    def county_item(self, response):
        url = "http://www.stats.gov.cn/tjsj/tjbz/tjyqhdmhcxhfdm/2017/"
        for sel in response.selector.xpath('//table//td'):
            if len(sel.xpath('a/@href').extract()) != 0:
                title = sel.xpath('a/text()').extract()
                link = sel.xpath('a/@href').extract()
                if (not title[0].isdigit()):
                    item = StockItem()
                    code = link[0].split(".")[0].split("/")[1]
                    if len(code) == 9:
                        code += "000"
                    elif len(code) == 6:
                        code += "000000"
                    item["code"] = code
                    item["name"] = title[0]
                    yield item
