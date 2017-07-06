from gensim.models import Word2Vec

import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

with open('tags.txt') as f:
    content = f.readlines()
    sentences = [x.split() for x in content]
    
    model = Word2Vec(sentences, min_count=10)
    
    print(model.most_similar(positive=['nltk'], negative=['java'],  topn=3))
    
f.closed

