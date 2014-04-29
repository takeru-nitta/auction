# -*- coding: utf-8 -*-

# run "pip install mysql-python" before use
import MySQLdb


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
