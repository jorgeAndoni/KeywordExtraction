
import utils
import pandas as pd
from embeddings import WordEmbeddings
from network import CNetwork
import numpy as np
import matplotlib.pyplot as plt

class Manager(object):

    def __init__(self, db='hult', edge_w=None, emb_type=None):
        self.database = db
        self.edge_weight_type = edge_w
        self.embedding_method = emb_type

    def test_embeddings(self):
        dataset = pd.read_csv('databases/hult_db.csv')
        dataset.info()
        print()
        train_abstracts = pd.read_csv('databases/abstracts_silva.csv')
        train_abstracts.info()

        dataset_all_text = utils.read_csv_column(dataset, 'all_content')
        dataset_only_nouns = utils.read_csv_column(dataset, 'only_nouns')
        reference_keywords = utils.read_csv_column(dataset, 'keywords')
        abstracts_content = utils.read_csv_column(train_abstracts, 'all_content')

        objEmb = WordEmbeddings(dataset_all_text, abstracts_content, 'ft', 100)
        word_embeddings = objEmb.get_word_embeddings()


    def evaluate_embeddings(self, dimension):
        dataset = pd.read_csv('databases/hult_db.csv')
        #dataset = pd.read_csv('databases/semEval_db.csv')
        dataset.info()
        print()
        train_abstracts = pd.read_csv('databases/abstracts_silva.csv')
        train_abstracts.info()

        dataset_all_text = utils.read_csv_column(dataset, 'all_content')
        dataset_only_nouns = utils.read_csv_column(dataset, 'only_nouns')
        reference_keywords = utils.read_csv_column(dataset, 'keywords')
        abstracts_content = utils.read_csv_column(train_abstracts, 'all_content')


        objEmb = WordEmbeddings(dataset_all_text, abstracts_content, 'w2v', dimension)
        word_embeddings = objEmb.get_word_embeddings()

        #documents = dataset_only_nouns[0:10]
        #reference_keywords= reference_keywords[0:10]

        results = []
        index = 1
        #for i, j in zip(dataset_only_nouns, reference_keywords) :
        for  i, j in zip(dataset_all_text, reference_keywords):
            obj = CNetwork(i, word_embeddings)
            keyword_dict = obj.sort_words()
            keywords = list(set(j)) # deberia corregirlo antes de generar el csv
            taxas = utils.get_taxas(keywords, keyword_dict)
            print(index, taxas)
            index+=1
            results.append(taxas)
            print()

        print('Testing ...')
        results = np.array(results)
        avgs = np.mean(results, axis=0)
        avgs = [round(x,4) for x in avgs]
        return avgs

    def evaluate_found_keywords(self):
        #dataset = pd.read_csv('databases/hult_db.csv')
        dataset = pd.read_csv('databases/semEval_db.csv')
        dataset.info()
        print()
        dataset_all_text = utils.read_csv_column(dataset, 'all_content')
        dataset_only_nouns = utils.read_csv_column(dataset, 'only_nouns')
        reference_keywords = utils.read_csv_column(dataset, 'keywords')

        print(len(dataset_all_text), len(dataset_only_nouns), len(reference_keywords))
        proportions = []
        sizes = []
        for index, (allT, keys) in enumerate(zip(dataset_all_text, reference_keywords)):
            print(index+1, len(allT),  len(keys))
            inter = len(set(allT)&set(keys))
            prop = inter/len(keys)
            sizes.append(len(set(allT)))
            print('Intersection',inter)
            print('Proportion', prop)
            print()
            proportions.append(prop)
        print('\nFinals:')
        print(min(sizes), max(sizes), np.mean(sizes))
        sizes.sort(reverse=True)
        print(sizes)

        plt.plot(sizes)
        #plt.title('HultDb : Common words between abstracts and reference keywords')
        #plt.title('SemEvalDb : Common words between content paper and reference keywords')
        #plt.title('HultDb : Number of unique words per abstracts')
        plt.title('SemEvalDb : Number of unique words per complete papers')
        plt.xlabel('Complete content papers')
        plt.ylabel('Number of unique words')
        plt.grid(True)
        plt.show()








    def keyword_analysis(self):
        #d = {'paper_id': paper_ids, 'all_content': contents}
        #dgr, stg, pr, pr_w, btw, btw_w, cc, cc_w, clos, clos_w
        measures = ['dgr', 'stg', 'pr', 'pr_w', 'btw', 'btw_w', 'cc', 'cc_w', 'clos', 'clos_w']
        embeddings = [50,100, 200, 300, 500, 1000]
        #embeddings = [100, 200]
        results = []
        for dimension in embeddings:
            print('---- Evaluating ' + str(dimension) + ' dimensions ----' )
            values = self.evaluate_embeddings(dimension)
            results.append(values)
            print('------ End evaluation -------')

        for i,j in zip(embeddings, results):
            print(i,j)

        data_dict = dict()
        data_dict['dimension'] = embeddings
        results = np.array(results)
        for i in range(len(measures)):
            print(measures[i],results[:,i])
            data_dict[measures[i]] = results[:,i]

        df = pd.DataFrame(data=data_dict)
        df.to_csv('testing_w2v_skipgram_hultDb.csv')  # revisar cuantos keywords tienen los abstracts
                                                    # revisar otros datasets de key extraction










if __name__ == '__main__':
    pass