#coding:utf-8
import pandas as pd
import MeCab
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn import neighbors

def analyzer(text):

    ret = []
    tagger = MeCab.Tagger('-Ochasen')
    node = tagger.parseToNode(text.encode('utf-8'))
    node = node.next
    while node.next:
        ret.append(node.feature.split(',')[-3].decode('utf-8'))
        node = node.next
    return ret

class data_store:
    
    def __init__(self, filename, MAX_DF=0.1, MAX_FEATURES=300, LSA_DIM=100):
        
        self.MAX_DF = MAX_DF
        self.MAX_FEATURES = MAX_FEATURES
        self.LSA_DIM = LSA_DIM
        
        self.data = pd.read_csv(filename)
        self.price = self.data['Price'].values
        self.title_list = []
        for i in self.data.index:
            self.title_list.append(self.data.ix[i, 'Title'].decode('utf-8'))
        self.tf, self.vectorizer, self.lsa = self.to_vector(self.title_list)
        
    def to_vector(self, title_list):
        
        vectorizer = TfidfVectorizer(analyzer=analyzer, max_df=self.MAX_DF)
        vectorizer.max_features = self.MAX_FEATURES       
        vectorizer.fit(title_list)
        tf = vectorizer.transform(title_list)
        
        lsa = TruncatedSVD(self.LSA_DIM)
        lsa.fit(tf)
        tf = lsa.transform(tf)
        
        return tf, vectorizer, lsa
        
    def add_data(self, filename):
        
        newdata = pd.read_csv(filename)
        
        for i in newdata.index:
            self.title_list.append(newdata.ix[i, 'Title'].decode('utf-8'))
        self.data.append(newdata)
        
        self.price = self.data['Price'].values
        
        self.tf, self.vectorizer, self.lsa = self.to_vector(self.title_list)
    
    def output(self):
        return self.data, self.tf
    
    
class KNeighbor(data_store):
    
    def __init__(self, filename, n_neighbors=5, MAX_DF=0.1, MAX_FEATURES=300, LSA_DIM=100):
        
        data_store.__init__(self, filename, MAX_DF, MAX_FEATURES, LSA_DIM)
        self.n_neighbors = n_neighbors
        
    def predict(self, testee):

        vector = self.vectorizer.transform([testee.decode('utf-8')])
        vector = self.lsa.transform(vector)
        
        knn = neighbors.KNeighborsRegressor(self.n_neighbors, 'distance')
        estimated = knn.fit(self.tf, self.price).predict(vector)
        dist, ind = knn.kneighbors(vector) 
        simmilar = self.data.ix[ind.tolist()[0]]
        return estimated, simmilar, dist

