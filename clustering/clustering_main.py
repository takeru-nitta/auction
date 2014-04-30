#coding:utf-8
import clustering as cl

if __name__ == '__main__':
    '''
    クラス01234をcsvで出力
    文字コードはutf-8。　エクセルで見る場合は、サクラエディタとかでshift_JISに変換してから    
    '''
    
    current = cl.clustering('sony.csv')
    current.output_csv()
    
    current.label_output(0).to_csv('output_0.csv')
    current.label_output(1).to_csv('output_1.csv')
    current.label_output(2).to_csv('output_2.csv')
    current.label_output(3).to_csv('output_3.csv')
    current.label_output(4).to_csv('output_4.csv')
    

