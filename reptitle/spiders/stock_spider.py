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

    def parse(self, response):
        for sel in response.selector.xpath('//div[contains(@class, "quotebody")]//div//ul//li'):
            if len(sel.xpath('a/@href').extract()) != 0:
                text = sel.xpath('a/text()').extract()[0]
                href = sel.xpath('a/@href').extract()[0]
                try:
                    code_all = re.findall(r"[s][hz]\d{6}", href)[0]
                    type = code_all[0:2]
                    code = code_all[2:]
                    if code[0] == 6 or code[0] == 3 or code[0] == 0:
                        item = StockItem()
                        item["name"] = re.sub(r'\(.*\)', "", text)
                        item["type"] = type
                        item["code"] = code
                        item["east_money_detail_url"] = href
                        yield item
                except Exception as e:
                    print(e)
                    continue

