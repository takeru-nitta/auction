# -*- coding: utf-8 -*-

from get_data_DB import get_maker_data
from LDAmodel import make_lda

def LDA_initial(maker,filters = True,show=False,no_below=5, no_above=0.75,num_topics=150):
	print "begin "+maker

	data = get_maker_data(maker)
	data_title = data["title"].values
	data_description = data["description"].values
	print "read data of" + maker

	title_lda = make_lda(data_title)
	title_lda.dictionary_corpus(filter=filters,show=show,no_below=no_below, no_above=no_above)
	title_lda.LDA_model(num_topics=num_topics,save=("./model/"+maker+"_title.model"),show=show)
	print "titile's model of "+maker +" made"

	description_lda = make_lda(data_description)
	description_lda.dictionary_corpus(filter=filters,show=show,no_below=no_below, no_above=no_above)
	description_lda.LDA_model(num_topics=num_topics,save=("./model/"+maker+"_description.model"),show=show,set_matrix=False)      
	print "description's model of "+maker +" made"


def LDA_initial_all(maker_list):
	for maker in maker_list:
		LDA_initial(maker)
	print "all updated"

maker_list=['NEC','SONY','FUJITSU','DELL','TOSHIBA']


