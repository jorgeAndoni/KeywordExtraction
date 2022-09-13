import pandas as pd
import utils
from embeddings import WordEmbeddings
from network import CNetwork
import numpy as np
import matplotlib.pyplot as plt
from bert_manager import BertEmbeddingsGenerator, BERTFinal

class Manager(object):

    def __init__(self, db='hult', metodo=1):
        #hp = 'databases/hult_db.csv'
        hp = 'databases/hult_db_v3.csv'
        sp = 'databases/semEval_db.csv'
        self.metodo = metodo
        if db == 'hult': self.dataset = pd.read_csv(hp)
        else: self.dataset = pd.read_csv(sp)
        #self.dataset = self.dataset.head(20) ###############
        self.dataset.info()
        print(self.dataset.shape)
        print()

    def load_word_embeddings(self):
        train_abstracts = pd.read_csv('databases/abstracts_silva.csv')
        train_abstracts.info()
        dataset_all_text = utils.read_csv_column(self.dataset, 'all_content')
        abstracts_content = utils.read_csv_column(train_abstracts, 'all_content')
        objEmb = WordEmbeddings(dataset_all_text, abstracts_content, 'w2v', 300)
        word_embeddings = objEmb.get_word_embeddings() # revisar nuevamente csv pre-procesamiento
        print('All words:', len(word_embeddings))
        return word_embeddings

    def keyword_analysis(self): # modificar la forma de cargar los embeddings: bien de word2vec o de bert
        #word_embeddings = self.load_word_embeddings()
        word_embeddings = ""
        #bert_obj = BertEmbeddingsGenerator(self.dataset)
        bert_obj = BERTFinal(self.dataset)
        bert_embeddings = bert_obj.get_embeddings()
        #a = input()
        if self.metodo == 1:
            self.metodo_1(word_embeddings)
        elif self.metodo == 2:
            #self.metodo_2(word_embeddings)
            self.metodo_2(bert_embeddings)
        elif self.metodo == 3:
            #self.metodo_3(word_embeddings)
            self.metodo_3(bert_embeddings)

    def metodo_1(self, word_embeddings):
        dataset_all_text = utils.read_csv_column(self.dataset, 'all_content')
        #dataset_only_nouns = utils.read_csv_column(dataset, 'only_nouns')
        reference_keywords = utils.read_csv_column(self.dataset, 'keywords')

        #dataset_all_text = dataset_all_text[0:5]
        #reference_keywords = reference_keywords[0:5]
        results = []
        for count, (texto, keywords) in enumerate(zip(dataset_all_text, reference_keywords)):
            keywords = list(set(keywords))
            obj = CNetwork(texto, word_embeddings)
            network = obj.create_complete_network()
            keyword_lists = obj.sort_words(network, only_weighted=True)
            taxas = utils.get_taxas(keywords, keyword_lists)
            print(count+1, taxas)
            results.append(taxas)

        results = np.array(results)
        avgs = np.mean(results, axis=0)
        avgs = [round(x, 4) for x in avgs]
        print('Final avgs:',avgs)

    def metodo_2(self, embeddings):
        dataset_all_text = utils.read_csv_column(self.dataset, 'all_content')
        reference_keywords = utils.read_csv_column(self.dataset, 'keywords')
        doc_ids = list(self.dataset['paper_id'])

        #dataset_all_text = dataset_all_text[0:5]
        #reference_keywords = reference_keywords[0:5]

        #texto = dataset_all_text[1]
        #keywords = reference_keywords[1]
        #print(texto)
        #print(keywords)
        container = [[] for _ in range(3)]
        for count, (p_id, texto, keywords) in enumerate(zip(doc_ids,dataset_all_text, reference_keywords)):
            print(count+1, texto)
            word_embeddings = embeddings[p_id] ##### cambio a considerar!!!!!!!!!
            obj = CNetwork(texto, word_embeddings)
            janelas = [obj.create_network_janela(i+1) for i in range(3)]
            for index, network in enumerate(janelas):
                keyword_lists = obj.sort_words(network, only_weighted=True)
                taxas = utils.get_taxas(keywords, keyword_lists)
                print(taxas)
                container[index].append(taxas)
            print('\n')

        print('\n Final Results:')
        for i in container:
            i = np.array(i)
            avgs = np.mean(i, axis=0)
            avgs = [round(x, 4) for x in avgs]
            print(avgs)

    def get_all_text_features(self, texto, keywords, word_embeddings):
        print(texto)
        all_taxas = []
        obj = CNetwork(texto, word_embeddings)
        janelas = [obj.create_network_janela(i + 1) for i in range(3)]
        for janela in janelas:
            networks = obj.get_embedding_networks(janela)
            print(networks)
            for network in networks:
                keyword_lists = obj.sort_words(network, only_weighted=False)
                taxas = utils.get_taxas(keywords, keyword_lists)
                print(taxas)
                all_taxas.append(taxas)
        return all_taxas

    def metodo_3(self, embeddings): #### Final para pruebas!
        dataset_all_text = utils.read_csv_column(self.dataset, 'all_content')
        reference_keywords = utils.read_csv_column(self.dataset, 'keywords')
        doc_ids = list(self.dataset['paper_id'])
        container = [[] for _ in range(15)]
        #dataset_all_text = dataset_all_text[0:5]
        #reference_keywords = reference_keywords[0:5]

        #for count, (p_id, texto, keywords) in enumerate(zip(doc_ids, dataset_all_text, reference_keywords)):
        for count, (p_id,texto, keys) in enumerate(zip(doc_ids,dataset_all_text, reference_keywords)):
            print(count + 1, texto)
            word_embeddings = embeddings[p_id]
            taxas = self.get_all_text_features(texto, keys, word_embeddings)
            for index, taxa in enumerate(taxas):
                container[index].append(taxa)

        file = open('bert_results/prueba.txt', 'w')
        print('\n Final results resumo:')
        for index, i in enumerate(container):
            i = np.array(i)
            avgs = np.mean(i, axis=0)
            avgs = [round(x, 4) for x in avgs]
            result = utils.vec_to_str(avgs) + '\n'
            print(index+1, i.shape, result)
            #print(index+1, i.shape, utils.vec_to_str(avgs))
            file.write(result)
            print()
        file.close()


if __name__ == '__main__':

    obj = Manager(db='hult', metodo=3)
    obj.keyword_analysis()

