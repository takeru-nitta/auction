# -*- coding: utf-8 -*-

# run "pip install mysql-python" before use
import MySQLdb


# read
def get_ids(category, limit=10, offset=0):
    sql = "SELECT * FROM item WHERE category_id = '%s'" % (category)
    sql += " ORDER BY auction_id ASC LIMIT %d OFFSET %d" % (limit, offset)
    return query(sql)


# write
def insert_id(no, ctg):
    sql = "INSERT INTO item (auction_id, category_id)"
    sql += " VALUES ('%s', '%s')" % (str(no), str(ctg))
    query(sql)


def insert_ids(ls):
    for no, ctg in ls:
        insert_id(no, ctg)


# run query
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

    result = cursor.fetchall()
    res = []
    for r in result:
        res.append(dict(r))

    connector.commit()
    cursor.close()
    connector.close()

    return res
