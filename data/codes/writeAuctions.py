import pickle
import yahooapi

def makeAuction(instr,outstr):
	f1=open(instr)
	l = pickle.load(f1)
	f1.close()
	#l=[u'c335858052']
	f2=open(outstr,'a')
	for aucID in l:
		aucres=yahooapi.api_goods(aucID).get_response()
		if aucres == None:
		    continue
		line=aucres.readline()
		while line!="":
			if "ResultSet" in line or "Description" in line or '?xml' in line: 
				pass
			else:
				f2.write(line)
			line=aucres.readline()
	f2.close()


#aucdoms=xml.dom.minidom.parse(aucres)
		#aucdom = aucdoms.getElementsByTagName('Description')
		# #aucdom[0].childNodes[0].data=u''
		# aucdom = aucdoms.getElementsByTagName('Result')
		# print aucdom[0].childNodes[0].data
		# f2.write(aucdom[0].childNodes[0].data)

