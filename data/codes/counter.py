# coding:utf-8

from bulk_insert import query
import time

for i in range(1000):
    print i,
    print query('select count(auction_id) from item')[0]['count(auction_id)']
    time.sleep(1)
