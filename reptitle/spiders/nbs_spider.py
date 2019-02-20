import scrapy
from reptitle.items import AreaItem


class NbsSpider(scrapy.Spider):
    name = "nbs"
    allowed_domains = ["stats.gov.cn"]
    start_urls = [
        "http://www.stats.gov.cn/tjsj/tjbz/tjyqhdmhcxhfdm/2017/index.html"
    ]

    def parse(self, response):
        url = "http://www.stats.gov.cn/tjsj/tjbz/tjyqhdmhcxhfdm/2017/"
        for sel in response.selector.xpath('//table//td'):
            if len(sel.xpath('a/@href').extract()) != 0:
                title = sel.xpath('a/text()').extract()
                link = sel.xpath('a/@href').extract()
                if not title[0].isdigit():
                    item = AreaItem()
                    item["code"] = link[0].split(".")[0] + "0000000000"
                    item["name"] = title[0]
                    yield item
                cl = url + link[0]
                yield scrapy.Request(cl, callback=self.city_item)

    def city_item(self, response):
        url = "http://www.stats.gov.cn/tjsj/tjbz/tjyqhdmhcxhfdm/2017/"
        for sel in response.selector.xpath('//table//td'):
            if len(sel.xpath('a/@href').extract()) != 0:
                title = sel.xpath('a/text()').extract()
                link = sel.xpath('a/@href').extract()
                if not title[0].isdigit():
                    item = AreaItem()
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
                if not title[0].isdigit():
                    item = AreaItem()
                    code = link[0].split(".")[0].split("/")[1]
                    if len(code) == 9:
                        code += "000"
                    elif len(code) == 6:
                        code += "000000"
                    item["code"] = code
                    item["name"] = title[0]
                    yield item
