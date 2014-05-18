# coding:utf-8
import MySQLdb


def add_items(ls):
    ids = []
    for e in ls:
        ids.append("('%s')" % str(e))
    q = "REPLACE INTO item(auction_id) VALUES %s" % ','.join(ids)
    query(q)


def update_item(info):
    pass


def query(sql):
    connector = MySQLdb.connect(
        host="suri-auction.cp8hyreygaih.ap-northeast-1.rds.amazonaws.com",
        db="auction",
        user="suri",
        passwd="surirokken",
        charset="utf8"
    )
    cursor = connector.cursor(MySQLdb.cursors.DictCursor)
    cursor.execute(sql)
    connector.commit()
    cursor.close()
    connector.close()
