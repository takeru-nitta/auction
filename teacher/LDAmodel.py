# -*- coding: utf-8 -*-
import MeCab
import pandas as pd
import gensim


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




class make_lda(object):
    """docstring for make_lda"""
    def __init__(self,sentencelist):
        self.sentencelist = sentencelist
        self.wordslist = get_words(sentencelist)
        #self.dictionary_corpus()
        #self.LDA_model
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


    def predict(self,sentence,threhold=0.8, interva=False):
        p_words = tokenlize(sentence)
        p_corpus = self.dictionary.doc2bow(p_words)
        return self.matrix[self.lda[p_corpus]]
        







