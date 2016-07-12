# -*- coding: utf-8 -*-
"""
This script trains model on processed Czech wiki data.
Model is also saved in binary format.

source: http://textminingonline.com/training-word2vec-model-on-english-wikipedia-by-gensim

Run example:
python3 word2vec_model.py wiki.cs.text wiki.cs.text_model
"""

#!/usr/bin/env python
# -*- coding: utf-8 -*-
import logging
import os.path
import sys
import multiprocessing
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
 
 
if __name__ == '__main__':
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)
 
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s')
    logging.root.setLevel(level=logging.INFO)
    logger.info("running %s" % ' '.join(sys.argv))
 
    # check and process input arguments
 
    if len(sys.argv) < 3:
        print (globals()['__doc__'] % locals())
        sys.exit(1)
    inp, outp = sys.argv[1:3]
 
    model = Word2Vec(LineSentence(inp), size=400, window=5, min_count=5, workers=multiprocessing.cpu_count())
 
    # trim unneeded model memory = use (much) less RAM
    model.init_sims(replace=True)
 
    model.save(outp)
    outp2 = outp + '_bin'
    model.save_word2vec_format(outp2, binary=True)