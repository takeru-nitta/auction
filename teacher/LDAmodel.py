# -*- coding: utf-8 -*-
import MeCab
import pandas as pd
import gensim
from get_data_DB import get_maker_data

def get_words(contents):
    ret = []
    for content in contents:
        ret.append(get_words_main(content))
    return ret

def get_words_main(content):
    return tokenlize(content)

def read_data(filename):
    data = pd.read_csv(filename,header=0,sep=",")
    title_list = []
    for i in data.index:
        title_list.append(data.ix[i, 'Title'])#.decode('utf-8'))
    return data , title_list


def tokenlize(text):
    #mecab = MeCab.Tagger("-Owakati")
    #node = mecab.parse(text.encode('utf-8'))
    #return node
    text=text.encode('utf-8')
    tagger = MeCab.Tagger('-Ochasen')
    node = tagger.parseToNode(text)#.encode('utf-8'))
    keywords = []
    while node:
        if node.feature.split(",")[0] == u"名詞":
            #yield node.surface
            if len(node.surface) > 1:
                keywords.append(node.surface)
        node = node.next
    return keywords




class lda_parts(object):
    """docstring for lda_parts"""
    def __init__(self,sentencelist):
        #self.sentencelist = sentencelist
        self.wordslist = get_words(sentencelist)
    def dictionary_corpus(self,filter = True,read = None ,save = None,show=False,no_below=3, no_above=0.6):
        if read == None:
            dictionary = gensim.corpora.Dictionary(self.wordslist)
            if filter == True: 
                #unfiltered = dictionary.token2id.keys()
                dictionary.filter_extremes(no_below,no_above)
                #filtered = dictionary.token2id.keys()
                #filtered_out = set(unfiltered) - set(filtered)
                #for out in filtered_out:
                #    print out
        else:
            dictionary = gensim.corpora.Dictionary.load_from_text(read)

        self.dictionary = dictionary
        if save != None:
            self.dictionary.save_as_text(save)
        
        self.corpus = [self.dictionary.doc2bow(words) for words in self.wordslist]
        
        if show == True:
            print(self.dictionary.token2id)

    def LDA_model(self,num_topics=100,save=None,load=None,show=False,set_matrix = True):
        if load == None:
            self.lda = gensim.models.LdaModel(corpus=self.corpus, id2word=self.dictionary, num_topics=num_topics)    
            self.lda.save(save)
        else:
            self.lda = gensim.models.LdaModel.load(load)

        if show == True:
            for topic in self.lda.show_topics(-1):
                print topic
        if set_matrix:
            self.similarity_matrix()

    def similarity_matrix(self):
        self.matrix = gensim.similarities.MatrixSimilarity(self.lda[self.corpus])



        


class auction_LDA(object):
    def __init__(self,maker,filters = True,show=False,no_below=5, no_above=0.75):
        print "begin " + maker
        data = get_maker_data(maker)
        assert len(data) > 1, "cannot get data"
        print "load data from" +maker

        self.auctionID = data["auction_id"].values
        self.price = data["current_price"].values

        self.title_lda = lda_parts(data["title"].values)
        self.title_lda.dictionary_corpus(filter=filters,show=show,no_below=no_below, no_above=no_above)
        self.title_lda.LDA_model(load=("./model/"+maker+"_title.model"),show=show)
        
        self.description_lda = lda_parts(data["description"].values)
        self.description_lda.dictionary_corpus(filter=filters,show=show,no_below=no_below, no_above=no_above)
        self.description_lda.LDA_model(load=("./model/"+maker+"_description.model"),show=show)    
        print "load model from" +maker






