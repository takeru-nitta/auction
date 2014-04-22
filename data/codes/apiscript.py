import yahooapi
#import xml.etree.ElementTree
import xml.dom.minidom

#auction1=yahooapi.api_goods('h187497791')
#auction2=api_history('w101107767',1)
category1=yahooapi.api_search(2084307175,page=1)
res=category1.get_response()
#page=res.read()
#print page
doms = xml.dom.minidom.parse(res)
for dom in doms.getElementsByTagName('AuctionID'):
    aucres=yahooapi.api_goods(dom.childNodes[0].data).get_response()
    if aucres == None:
        continue
    aucdoms=xml.dom.minidom.parse(aucres)
    print "1"
    aucdom = aucdoms.getElementsByTagName('EndTime')
    print aucdom[0].childNodes[0].data




#for node in tree.findall('AuctionID'):
#    print node.tag, node.attrib
##res=auction1.get_response()
#f1=open('auction1.xml','w')
#f1.write(page)
##
##page=res.readlines()
##for e in page:
##   f1.write(e)
#f1.close

#f1=open('auction1.json','w')
#f1.write(page)
#f1.close
#tree = xml.etree.ElementTree.XML(page)
#f1=open("test.xml",'w')
#tree.write("test.xml")#, encoding="utf-8")
#f1.close


#jobject=json.loads(page[7:-1],encoding="cp932")
#jobject[u'ResultSet'][u'Result'][u'Description']=u""


#import codecs

#f1 = codecs.open('auction1.json', "w", "utf-8")
#f1=open('auction1.json','w')
#json.dumps(jobject,f1, indent=4,ensure_ascii=False)
#f1.close()



#f = codecs.open('auction2history.json', "w", "utf-8")
#json.dump(jobject, f, indent=2, sort_keys=True, ensure_ascii=False)
#f.close()


#page=json.dumps(jobject, indent=4)
#f1.write(json.dumps(jobject, indent=4))

#print (json.dumps(jobject, indent=4)).encode('utf-8')
#print ("\u5bcc\u58eb\u901a").encode('utf-8')


#print json.dumps(jobject, sort_keys=True, indent=4)
