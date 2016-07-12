# -*- coding: utf-8 -*-
"""
Created on Tue Jul 12 07:25:19 2016

@author: nastya
"""
from gensim.models import Word2Vec
model = Word2Vec.load('~/wiki/wiki.cz.model')
model.most_similar('žena', topn=5)

model.most_similar('muž', topn=5)

model.most_similar('seznam', topn=5)

model.similarity('žena', 'muž')
model.similarity('žena', 'dívka')

model.doesnt_match("oběd snídaně kočka svačina".split())
model.doesnt_match("oběd snídaně hruška svačina".split())

model.most_similar(positive=['žena', 'král'], negative=['muž'])