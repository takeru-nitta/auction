# -*- coding: utf-8 -*-
from get_data_DB import get_maker_data
from LDAmodel import make_lda,tokenlize
import crawler 


class LDAmodel(object):
	def __init__(self,maker,filters = True,show=False,no_below=5, no_above=0.75):
		print "begin " + maker
		data = get_maker_data(maker)
		assert len(data) > 1, "cannot get data"
		print "load data from" +maker

		self.auctionID = data["auction_id"].values
		self.price = data["current_price"].values

		self.title_lda = make_lda(data["title"].values)
		self.title_lda.dictionary_corpus(filter=filters,show=show,no_below=no_below, no_above=no_above)
		self.title_lda.LDA_model(load=("./model/"+maker+"_title.model"),show=show)
		
		self.description_lda = make_lda(data["description"].values)
		self.description_lda.dictionary_corpus(filter=filters,show=show,no_below=no_below, no_above=no_above)
		self.description_lda.LDA_model(load=("./model/"+maker+"_description.model"),show=show)	  
		print "load model from" +maker



class LDA_predict(object):
	def __init__(self,categoryIDdic,filters = True,show=False,no_below=5, no_above=0.75,num_topics=150):
		self.categoryIDdic=categoryIDdic
		self.model_dict={}
		for maker in categoryIDdic:
			self.model_dict[maker] = LDAmodel(maker)
			print "Load LDA model of "+maker
		print "all loaded"

	def search_maker(self,categoryID):
		for maker in self.categoryIDdic:
			if categoryID in self.categoryIDdic[maker]:
				return maker
		print "make not found"
		return None


	def setence_to_similarity(self,sentence,model):
		p_words = tokenlize(sentence)
		p_corpus = model.dictionary.doc2bow(p_words)
		return model.matrix[model.lda[p_corpus]]


	def predict(self,ID,threhold = 0.6,rate=1):
		testee = crawler.fetch_item(ID)
		maker = self.search_maker(int(testee['category_id']))
		if maker == None:
			raise NameError('No category_id found')

		title_similarity = self.setence_to_similarity(testee['title'],self.model_dict[maker].title_lda)
		description_similarity = self.setence_to_similarity(testee['title'],self.model_dict[maker].description_lda)
		
		sim = [(n,(rate+1)*s1*s2/(s1+rate*s2)) for ((n,s1),s2) in zip(enumerate(title_similarity),description_similarity) if (s1 !=0 and s2 != 0)]
		
		p_list = sorted([self.model_dict[maker].price[n] for (n,x) in sim if x > threhold ])
		l=len(p_list)
		if l == 0:
			return (None,[])
		else:
			return (p_list[l/2],[p_list[l/4],p_list[l*3/4]])
				




categoryIDdic = {
				 #'NEC':[2084193571,2084193570,2084193569,2084193568,2084193567],
				 'SONY':[2084193581,2084193580,2084193579,2084193578,2084193577],
				 #'FUJITSU':[2084193603,2084193602,2084193601,2084193600,2084193599],
				 #'DELL':[2084193586,2084193585,2084193584,2084193583,2084193582],
				 #'TOSHIBA':[2084193594,2084193593,2084193592,2084193591,2084193590]
				 }  

Predict = LDA_predict(categoryIDdic)