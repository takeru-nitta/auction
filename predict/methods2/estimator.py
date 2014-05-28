#coding:utf-8
import pandas as pd
import numpy as np
import MeCab
import math
import get_data_DB as db
import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn import neighbors
from sklearn import svm
from sklearn import linear_model
from sklearn import cross_validation
from sklearn.preprocessing import Normalizer
import statsmodels.api as sm
from statsmodels.sandbox.regression.predstd import wls_prediction_std
import crawler

'''
scikitのライブラリを使うためのテンプレート
estimatorを継承して、使いたいモデルを定義

.fit() をしてから
.predict('...')　でタイトルから価格を予測

.cross_validation() でデータを訓練用とテスト用に分け,それに基づいてscoreをだす
'''

def get_data(ID):
    data = crawler.fetch_item(ID)
    if data['condition']== u'new':
        condition = 1.0
    else: condition = 0.0
    data = [data['title'].encode('utf-8'), float(data['init_price'])**(0.5), float(data['seller_point']), condition]
    return data

def analyzer(text):
    ret = []
    tagger = MeCab.Tagger('-Ochasen')
    node = tagger.parseToNode(text.encode('utf-8'))
    while node:
        if node.feature.split(',')[0] == u'名詞'.encode('utf-8'):
            ret.append(node.surface)
        node = node.next
    return ret

def condition(text):
    if text == 'new': return 1
    else: return 0

class data_store:
    
    def __init__(self, maker, MAX_DF=0.1, MAX_FEATURES=300, LSA_DIM=100, exclude_0=True):
        
        self.MAX_DF = MAX_DF
        self.MAX_FEATURES = MAX_FEATURES
        self.LSA_DIM = LSA_DIM
        
        self.data = db.get_maker_data(maker)
        if exclude_0:
            self.data = self.data[self.data['bids']>0]
            
        self.data = self.data[self.data['end_time'] < datetime.datetime.now()]
        self.price = map((lambda x: x**(0.5)),  self.data['current_price'].values)
        data = self.data[['init_price', 'seller_point', 'condition']]
        data['init_price'] = data['init_price'].apply(lambda x: x**(0.5))
        data['condition'] = data['condition'].apply(condition)
        data = data.astype(float)
        
        #self.other =Normalizer(copy=False).fit_transform(data.values.T).T
        #self.other = data
        self.title_list = []
        for i in self.data.index:
            self.title_list.append(self.data.ix[i, 'title']) 
        self.tf, self.vectorizer, self.lsa = self.to_vector(self.title_list)
        
        self.sum = np.sum(data.values**2.0, axis=0)
        self.other = Normalizer(copy=False).fit_transform(data.T).T
        self.x = np.hstack([self.tf, self.other])
        
        
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
        return self.tf, self.data
    
    
class estimator(data_store):
    
    def __init__(self, maker, MAX_DF=0.1, MAX_FEATURES=300, LSA_DIM=100, exclude_0=True):
        
        data_store.__init__(self, maker, MAX_DF, MAX_FEATURES, LSA_DIM, exclude_0)
    
    def fit(self):
        
        self.model.fit(self.tf, self.price)
        
    def predict(self, testee):
        
        vector = self.vectorizer.transform([testee.decode('utf-8')])
        vector = self.lsa.transform(vector)
        estimated = self.model.predict(vector)
        
        return estimated
    
    def cross_validation(self):
        
        train_tf, test_tf, train_price, test_price = cross_validation.train_test_split(self.tf, self.price, test_size=0.1, random_state=0)
        self.model.fit(train_tf, train_price)
        
        return self.model.score(test_tf, test_price)
    
    
    
    
'''
以下のように、使いたいモデルをself.modelに代入すればいい
パラメータや、predictの返り値で追加するものがあれば、改めて定義する
'''     

class KNeighbors(estimator):
    
    def __init__(self, maker, n_neighbors = 5, MAX_DF=0.1, MAX_FEATURES=300, LSA_DIM=100): #追加のパラメータがある場合はここに
        
        estimator.__init__(self, maker, MAX_DF, MAX_FEATURES, LSA_DIM)
        self.model = neighbors.KNeighborsRegressor(n_neighbors, weights='distance')
        
    def predict(self, ID): #近くの商品も調べたいので、改めて定義
        
        list1 = get_data(ID)
        vector = self.vectorizer.transform([list1[0].decode('utf-8')])
        vector = self.lsa.transform(vector)
        array = np.array([list1[1:4]])**2 / self.sum
        array = array**0.5 
        vector= np.hstack([vector, array])
        
        estimated = self.model.predict(vector)
        dist, ind = self.model.kneighbors(vector) 
        simmilar = self.data.ix[self.data.index[ind.tolist()[0]]]
        return estimated**2.0 , simmilar
    
    def fit(self):
        self.model.fit(self.x, self.price)
    
    
class SVR(estimator):
    
    def __init__(self, maker, MAX_DF=0.1, MAX_FEATURES=300, LSA_DIM=100):
        
        estimator.__init__(self, maker, MAX_DF, MAX_FEATURES, LSA_DIM)
        self.model = svm.SVR()
        
        
class LinearRegression(estimator):
    
    def __init__(self, maker, MAX_DF=0.1, MAX_FEATURES=300, LSA_DIM=100):
        
        estimator.__init__(self, maker, MAX_DF, MAX_FEATURES, LSA_DIM)
        self.model = linear_model.LinearRegression()
        
    def fit(self):
        
        self.model.fit(np.hstack([self.tf, self.other]).tolist(), self.price)
        
    def predict(self, ID):
        list1 = get_data(ID)
        vector = self.vectorizer.transform([list1[0].decode('utf-8')])
        vector = self.lsa.transform(vector)
        array = np.array([list1[1:4]])
        vector= np.hstack([vector, array])
        
        estimated = self.model.predict(vector)
        
        return estimated
    
    def cross_validation(self):
        
        train_tf, test_tf, train_price, test_price = cross_validation.train_test_split(np.hstack([self.tf, self.other]).tolist(), self.price, test_size=0.1, random_state=0)
        self.model.fit(train_tf, train_price)
        
        return self.model.score(test_tf, test_price)
        
        
class BayesianRidge(estimator):
    
    def __init__(self, maker, MAX_DF=0.1, MAX_FEATURES=300, LSA_DIM=100):
        
        data_store.__init__(self, maker, MAX_DF, MAX_FEATURES, LSA_DIM)
        self.model = linear_model.BayesianRidge(normalize=True)
        
    def fit(self):
        
        self.model.fit(np.hstack([self.tf, self.other]).tolist(), self.price)
        
    def predict(self, list1):
        vector = self.vectorizer.transform([list1[0].decode('utf-8')])
        vector = self.lsa.transform(vector)
        array = np.array([list1[1:4]])
        vector= np.hstack([vector, array])
        
        estimated = self.model.predict(vector)
        
        return estimated
    
    def cross_validation(self):
        
        train_tf, test_tf, train_price, test_price = cross_validation.train_test_split(np.hstack([self.tf, self.other]).tolist(), self.price, test_size=0.1, random_state=0)
        self.model.fit(train_tf, train_price)
        
        return self.model.score(test_tf, test_price)
    
    
    
class Lasso(estimator):
    
    def __init__(self, maker, MAX_DF=0.1, MAX_FEATURES=300, LSA_DIM=100):
        
        data_store.__init__(self, maker, MAX_DF, MAX_FEATURES, LSA_DIM)
        self.model = linear_model.Lasso(alpha=0.5, normalize=True)
        
    def fit(self):
        
        self.model.fit(np.hstack([self.tf, self.other]).tolist(), self.price)
        
    def predict(self, ID):
        list1 = get_data(ID)
        vector = self.vectorizer.transform([list1[0].decode('utf-8')])
        vector = self.lsa.transform(vector)
        array = np.array([list1[1:4]])
        vector= np.hstack([vector, array])
        
        estimated = self.model.predict(vector)
        
        return estimated**2
    
    def cross_validation(self):
        
        train_tf, test_tf, train_price, test_price = cross_validation.train_test_split(np.hstack([self.tf, self.other]).tolist(), self.price, test_size=0.1, random_state=0)
        self.model.fit(train_tf, train_price)
        
        return self.model.score(test_tf, test_price)
        
class LinearRegression2(estimator):
    
    def __init__(self, maker, MAX_DF=0.1, MAX_FEATURES=300, LSA_DIM=100, exclude_0=True):
        
        data_store.__init__(self, maker, MAX_DF, MAX_FEATURES, LSA_DIM, exclude_0)
        #self.model = sm.OLS(self.price, sm.add_constant(np.hstack([self.tf, self.other]).tolist()))
        self.model = sm.OLS(self.price, self.x)
        
    def fit(self):
        
        results = self.model.fit()
        self.results = results
               
    def predict(self, ID, ALPHA=0.4):
        list1 = get_data(ID)
        vector = self.vectorizer.transform([list1[0].decode('utf-8')])
        vector = self.lsa.transform(vector)
        array = np.array([list1[1:4]])**2.0 / self.sum
        array = array**0.5
        vector= np.hstack([vector, array])
        estimated = self.results.predict(vector)
        prstdn, infa, supa = wls_prediction_std(self.results, vector, alpha = ALPHA)
        if infa[0] < 0:
            infa[0] = 0
        return estimated[0]**2.0, supa[0]**2.0, infa[0]**2.0