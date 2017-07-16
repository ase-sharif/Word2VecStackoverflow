import logging

from gensim.models import Word2Vec

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


def main(positive, negative, topn):
    """ This method train word2vec model, and return most similar tags

    Args:
        positive (list): list of positive tags
        negative (list): list of negative tags
        topn (int): number of top keywords in word2vec

    Returns:
        list: Return list of word2vec

    """
    with open('tags.txt') as f:
        content = f.readlines()
        sentences = [x.split() for x in content]

        model = Word2Vec(sentences, min_count=20)

        return model.most_similar(positive=positive, negative=negative, topn=topn)


if __name__ == '__main__':
    print(main(positive=['nltk'], negative=['java'], topn=5))
