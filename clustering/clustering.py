#coding:utf-8
import pandas as pd
from pandas import DataFrame
import numpy as np
import MeCab
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans

def analyzer(text):
    '''
    テキストを名詞に分けて、リストにして返す関数
    '''
    ret = []
    tagger = MeCab.Tagger('-Ochasen')
    node = tagger.parseToNode(text.encode('utf-8'))
    node = node.next
    while node.next:
        ret.append(node.feature.split(',')[-3].decode('utf-8'))
        node = node.next
    return ret


class clustering:
    '''
    csv形式のデータを受け取って、'Title'についてクラスタリングを行う。
    'Title'を 'tf-idf'を用いた'bag of words'表現に変換、次元を削減したのち、
    k最近防法によって指定した数のクラスに分ける。
    output()でラベル付けしたデータを、label_outputで指定したラベルのクラスに属するデータを返す。
    
    MAX_DF [0,1]　この値以上にたくさん出てくる単語は除く
    MAX_FEATURES 単語の種類
    LSA_DIM この数まで次元削減
    NUM_CLUSTERS 分けるクラスの数
    MINIBATCH 0:普通のk最近傍法　1:mini-batch　データ数が大きい時はこっち
    
    '''
    
    def __init__(self, filename,
                 MAX_DF = 0.9, MAX_FEATURES = 300, LSA_DIM = 100,
                 NUM_CLUSTERS = 30, MINIBATCH = 0):
        
        self.text_set, self.data = self.read_data(filename)
        self.X, self.lsa, self.vectorizer = self.to_vector(self.text_set, MAX_DF, MAX_FEATURES, LSA_DIM)
        self.km = self.clustering(self.X, NUM_CLUSTERS, MINIBATCH)

        
    def read_data(self, filename):
        data = pd.read_csv(filename)
        text_set = []
        for i in data.index:
            text_set.append(data.ix[i, 'Title'].decode('utf-8'))
        return text_set, data
    
    def to_vector(self, text_set, MAX_DF, MAX_FEATURES, LSA_DIM):
        '''
        bag of words に変換、次元削減    
        '''
        
        vectorizer = TfidfVectorizer(analyzer=analyzer ,max_df=MAX_DF)
        vectorizer.max_features = MAX_FEATURES
        X = vectorizer.fit_transform(text_set)
        lsa= TruncatedSVD(LSA_DIM)
        X = lsa.fit_transform(X)
        
        return X, lsa, vectorizer
        
    def clustering(self, X, NUM_CLUSTERS, MINIBATCH):
        '''
        k最近傍法によってクラス分け
        '''
        
        if MINIBATCH:
            km = MiniBatchkMeans(n_clusters = NUM_CLUSTERS,
                                 init='k-means++', batch_size=1000,
                                 n_init=10, max_no_improvement=10)
        else:
            km = KMeans(n_clusters=NUM_CLUSTERS, init='k-means++', n_init=1)
        
        km.fit(X)
        '''
        #もしかしたら使うかも
        labels = km.labels_ #商品の属するクラス
        transformed = km.transform(X) #商品の各クラスの中心への距離
        dists = np.zeros(labels.shape)
        for i in range(len(labels)):
            dists[i] = transformed[i, labels[i]] #商品の属するクラスの中心への距離
        '''
        labels = DataFrame(km.labels_)
        labels.columns = ['label']
        self.data = pd.concat([labels, self.data], axis=1) #元のデータにラベルを加える
        
        return km
        
    def output_csv(self):
        self.data.to_csv('output.csv')
        
    def label_output(self, label):
        return self.data[self.data['label']==label]
        
    def predict(self, new_filename):
        '''
        新しいデータに対して、クラス推定を行う。クラスのラベルを返す。
        '''
        text_set = self.read_data(new_filename)
        X = self.vectorizer.transform(text_set)
        X = self.lsa.transform(X)
        result = self.km.predict(X)
        return result
    
    
        
        