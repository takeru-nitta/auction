# -*- coding: utf-8 -*-
import LDA_one
import LDA_all

maker_list=['NEC','SONY','FUJITSU','DELL','TOSHIBA']
PredictOne = LDA_one.LDA_one(maker_list)
PredictAll = LDA_all.LDA_all(maker_list)


def predict_by_all(predict_list):
	prices = sorted(predict_list)
	return (prices[len(prices)/2],prices[1],prices[-1])
	#return sorted([p1[0],p1[0],p2[0]])[1],sorted([p1[1],p1[1],p2[1]])[1],sorted([p1[2],p1[2],p2[2]])[1]]



