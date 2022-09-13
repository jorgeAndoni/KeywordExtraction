from gensim.models import Word2Vec
import fasttext

class WordEmbeddings(object):

    def __init__(self,corpus, auxiliar=None, type_emb='w2v', size=300):
        self.embedding_type = type_emb
        self.size = size
        self.corpus = corpus
        if auxiliar is not None:
            self.corpus.extend(auxiliar)

    def get_word_embeddings(self):
        if self.embedding_type == 'w2v':
            print('\nTraining word2vec ...')
            return self.train_w2v()
        elif self.embedding_type == 'ft':
            print('\nTraining fastText ...')
            return self.train_ft()

    def train_w2v(self): # testar com skigram=1 y cbow=0
        model = Word2Vec(self.corpus, size=self.size, window=5, min_count=1, workers=4, sg=1)
        words = model.wv.vocab
        model_dict = dict()
        for word in words:
            model_dict[word] = model.wv[word]
        return model_dict

    def train_ft(self):
        path = 'databases/train_file.txt'
        '''
        train_file = open(path, 'w')
        for i in self.corpus:
            texto = ' '.join(i) + '\n'
            train_file.write(texto)
        print(len(self.corpus))
        '''
        #model: skipgram  cbow
        model = fasttext.train_unsupervised(path, model='skipgram', minCount=1, dim=self.size)
        model_dict = dict()
        words = model.words
        for word in words:
            model_dict[word] = model[word]
        return model_dict




if __name__ == '__main__':
    pass