import pandas as pd
from bert_embedding import BertEmbedding
import numpy as np
from nltk.stem import PorterStemmer
from nltk import word_tokenize

#import torch
#from transformers import BertTokenizer, BertModel
class BertEmbeddingsGenerator(object):

    def __init__(self, df_file):
        self.df = df_file

    def prepare_corpus(self):
        text_content = list(self.df['all_content_sentences'])
        all_sentences = []
        sizes = []
        print('All sentences:')
        for i in text_content:
            sentences = i.split('\n')
            all_sentences.extend(sentences)
            sizes.append(len(sentences))
        for i in all_sentences:
            print(i)
        print('All sentences:', len(all_sentences))
        print('sizes:', np.sum(sizes), len(sizes), sizes)
        return all_sentences, sizes

    def train_bert(self, all_sentences):
        print('Training bert ...')
        bert_embedding = BertEmbedding()
        #bert_embedding = BertEmbedding(model='bert_24_1024_16', dataset_name='book_corpus_wiki_en_cased')
        result = bert_embedding(all_sentences)
        return result

    def get_document_embeddings(self, pair_values):
        ps = PorterStemmer()
        dict_container = dict()
        for value in pair_values:
            sentence = value[0]
            print(sentence)
            #print(len(value[1]),value)
            #a = input()
            word_vectors = value[1]
            for word, vector in zip(sentence, word_vectors):
                word = ps.stem(word)
                if word in dict_container:
                    dict_container[word].append(vector)
                else:
                    dict_container[word] = [vector]
        model = dict()
        for word in dict_container:
            vector = np.array(dict_container[word])
            vector = np.mean(vector, axis=0)
            model[word] = vector
        return model

    def get_embeddings(self):
        paper_ids = list(self.df['paper_id'])
        all_sentences, sizes = self.prepare_corpus()
        all_embeddings = self.train_bert(all_sentences)

        positions = [sizes[0]]
        for i in range(1,len(sizes)):
            value = positions[i-1]+sizes[i]
            positions.append(value)
        print(positions)
        ini = 0
        result = dict()
        for index, (p_id,pos) in enumerate(zip(paper_ids,positions)):
            print(index, p_id, ini, pos)
            values = all_embeddings[ini:pos]
            model = self.get_document_embeddings(values)
            print(model.keys())
            result[p_id] = model
            ini = pos
        return result


from bert_serving.client import BertClient

class BertCito(object):

    def __init__(self, sentences):
        number_sentences = len(sentences)
        self.sentences = sentences
        #self.prepare_sentences()

    def prepare_sentences(self):
        # bert-serving-start -model_dir Documents/uncased_L-12_H-768_A-12/ -num_worker=4
        bc = BertClient()
        vectores  = bc.encode(self.sentences)
        for i in vectores:
            print(i.shape)

    def prueba2(self):
        #bert-serving-start -pooling_strategy NONE -model_dir Documents/uncased_L-12_H-768_A-12/ -show_tokens_to_client -num_worker=4
        #bert-serving-start -pooling_strategy NONE -model_dir Documents/uncased_L-12_H-768_A-12/ -show_tokens_to_client -max_seq_len NONE -num_worker=2
        #bert-serving-start -pooling_strategy NONE -model_dir Documents/uncased_L-24_H-1024_A-16/ -show_tokens_to_client -max_seq_len NONE -num_worker=2

        #bert-serving-start -pooling_strategy NONE -model_dir Documentos/uncased_L-12_H-768_A-12/ -show_tokens_to_client -max_seq_len NONE
        bc = BertClient()
        vec = bc.encode(['hey you', 'whats up?'])
        print(len(vec))
        print(vec[0][0][:5]) #`[CLS]`
        print(vec[0][1][:5]) #`hey`
        print(vec[0][2][:5]) #`you`
        print(vec[0][3][:5]) #`[SEP]`
        print(vec[0][4][:5]) #padding symbol
        #print(vec[0][56][:5])


    def prueba3(self):
        bc = BertClient()
        values, tokenized_sents = bc.encode(self.sentences, show_tokens=True)
        print(values.shape)
        for index, (sent, tokenized)  in enumerate(zip(self.sentences,tokenized_sents)):
            print(index,sent)
            print(word_tokenize(sent))
            print(len(tokenized),tokenized)
            for position, tok in enumerate(tokenized):
                print(position, tok, values[index][position][:5])
            print()



class BERTFinal(object):

    def __init__(self, df):
        self.df = df

    def prepare_corpus(self):
        text_content = list(self.df['all_content_sentences'])
        all_sentences = []
        sizes = []
        print('All sentences:')
        for i in text_content:
            sentences = i.split('\n')
            all_sentences.extend(sentences)
            sizes.append(len(sentences))
        for index, i in enumerate(all_sentences):
            print(index, i)
        print('All sentences:', len(all_sentences))
        print('sizes:', np.sum(sizes), len(sizes), sizes)
        return all_sentences, sizes

    def train_bert(self, all_sentences):
        print('Training bert ...')
        bc = BertClient()
        values, tokenized_sents = bc.encode(all_sentences, show_tokens=True)
        return values, tokenized_sents

    def get_sentence_embeddings(self, original_sentence, bert_tokenized, all_embeddings):
        bert_tokenized.remove('[CLS]')
        bert_tokenized.remove('[SEP]')
        print('tokenized:', bert_tokenized)
        buffer = ''
        results = []
        auxiliar = []
        index_position = 1
        for word in original_sentence:
            top = bert_tokenized.pop(0)
            if word == top:
                #print('found:',word, top)
                results.append([top])
                auxiliar.append([index_position])
                index_position+=1
            else:
                vector = []
                vector_pos = []
                vector.append(top)
                vector_pos.append(index_position)
                index_position += 1
                top = top.replace('##', '')
                buffer+=top
                while buffer!=word:
                    top = bert_tokenized.pop(0)
                    vector.append(top)
                    vector_pos.append(index_position)
                    index_position += 1
                    top = top.replace('##', '')
                    buffer += top
                results.append(vector)
                auxiliar.append(vector_pos)
                #print('joined:',word, buffer)
                buffer=''
        dict_container = dict()
        for word, subwords, indexes in zip(original_sentence,results, auxiliar) :
            #print(word, subwords, indexes)
            vector = []
            for w, i in zip(subwords, indexes):
                vector.append(all_embeddings[i])
            vector = np.array(vector)
            vector = np.mean(vector, axis=0)
            #print(vector.shape, vector[:5])
            #print()
            if word in dict_container:
                dict_container[word].append(vector)
            else:
                dict_container[word] = [vector]
        final_result = dict()
        for word in dict_container:
            vector = dict_container[word]
            vector = np.array(vector)
            vector = np.mean(vector, axis=0)
            final_result[word] = vector
        return final_result

    def get_document_embeddings(self, grouped_sentences):
        ps = PorterStemmer()
        dict_container = dict()
        for dictionary in grouped_sentences:
            #print(dictionary.keys())
            for word in dictionary:
                vector = dictionary[word]
                word = ps.stem(word)
                if word in dict_container:
                    dict_container[word].append(vector)
                else:
                    dict_container[word] = [vector]
        model = dict()
        for word in dict_container:
            vector = np.array(dict_container[word])
            vector = np.mean(vector, axis=0)
            model[word] = vector
        return model


    def get_embeddings(self):
        paper_ids = list(self.df['paper_id'])
        all_sentences, sizes = self.prepare_corpus()
        vectors, tokenized_sentences = self.train_bert(all_sentences)
        print('paper ids:',paper_ids)
        print('todas las oraciones:',all_sentences)
        print('sizess:',sizes)

        positions = [sizes[0]]
        for i in range(1, len(sizes)):
            value = positions[i - 1] + sizes[i]
            positions.append(value)
        print('positions:',positions)

        all_sentence_embeddings = []
        for index, (sent, tok_sent) in enumerate(zip(all_sentences,tokenized_sentences)):
            sent_base = word_tokenize(sent)
            val_embeddings = vectors[index]
            print(index, sent_base)
            sentence_embeddings = self.get_sentence_embeddings(sent_base, tok_sent, val_embeddings)
            all_sentence_embeddings.append(sentence_embeddings)
            print('Total words found:',len(sentence_embeddings))
            print()


        ini = 0
        result = dict()
        print('Joining sentences')
        for index, (p_id, pos) in enumerate(zip(paper_ids, positions)):
            print(index, p_id, ini, pos)
            grouped_sentences = all_sentence_embeddings[ini:pos]
            model = self.get_document_embeddings(grouped_sentences)
            print(len(model), model.keys())
            result[p_id] = model
            ini = pos
            print()
        return result






if __name__ == '__main__':

    df = pd.read_csv('databases/hult_db_v2.csv')
    df = df.head(5)
    df.info()
    #obj = BertEmbeddingsGenerator(df)
    obj = BERTFinal(df)
    obj.get_embeddings()



    '''
    sentences = ['After stealing money from the bank vault, the bank robber was seen fishing on the Mississippi river bank.',
                 'Hello my name is Jorge Andoni Valverde Tohalino, bye.',
                 'She is going to Pamela Revuelta school.']

    s1 = 'use of periodic and monotonic activation functions in multilayer feedforward neural networks trained by extended kalman filter algorithm the authors investigate the convergence and pruning performance of multilayer '
    sentences.append(s1)
    obj = BertCito(sentences)
    obj.prueba3()
    '''






