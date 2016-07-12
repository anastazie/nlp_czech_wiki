# -*- coding: utf-8 -*-
"""
Test created model with Czech words

Example:
>>> model.most_similar(positive=['žena', 'král'], negative=['muž']) 
[('královna', 0.5616521239280701),
 ('manželka', 0.5222904682159424),
 ('alžběta', 0.5171604156494141),
 ('choť', 0.5107160806655884),
 ('císařovna', 0.5097413659095764),
 ('isabela', 0.49988824129104614),
 ('konstancie', 0.48815155029296875),
 ('regentka', 0.48672398924827576),
 ('eleonora', 0.48375043272972107),
 ('kněžna', 0.47714075446128845)]

"""

from gensim.models import Word2Vec
# Load model
model = Word2Vec.load('/home/nasta/nlp_czech_wiki/wiki.cz.model')

model.most_similar(positive=['žena', 'král'], negative=['muž'])
model.most_similar(positive=['prachy', 'pohoda'], negative=['práce'])

# Slovak words are also present
model.most_similar('pieseň', topn=5)

# Words need to be stemmed
model.most_similar('strom', topn=5)

# Catch rare words
model.most_similar('lokna', topn=5)

model.most_similar('seznam', topn=5)

model.similarity('žena', 'muž')
model.similarity('žena', 'dívka')
model.similarity('muž', 'mladík')

model.doesnt_match("oběd snídaně kočka svačina".split())
model.doesnt_match("oběd snídaně hruška svačina".split())

