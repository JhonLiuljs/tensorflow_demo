# -*- coding: utf-8 -*-
import urllib.request


def main():
    print(1)
    url = "http://quotes.money.163.com/service/chddata.html?" \
          "code=0600756" \
          "&start=20160902" \
          "&end=20171108" \
          "&fields=TCLOSE;HIGH;LOW;TOPEN;LCLOSE;CHG;PCHG;VOTURNOVER;"
    with urllib.request.urlopen(url) as response:
        html = response.read()
        print(html)
        list_str = bytes(html, encoding='utf8')
        print(list_str)


if __name__ == '__main__':
    main()
