from nltk import bigrams
import igraph
from igraph import *
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from scipy import spatial
import utils
from sklearn.metrics import pairwise_distances
import xnet
import platform

class CNetwork(object):

    def __init__(self, document, word_embeddings):
        self.document = document
        self.words = list(set(self.document))
        self.word_index = {index:word for index,word in enumerate(self.words)}
        self.vocab_index = {word: i for i, word in enumerate(self.words)}
        self.embeddings = word_embeddings
        self.percentages = [5, 10, 20, 50]
        plataforma = platform.platform()
        if plataforma.find('Linux') != -1:
            self.operating_system = 'linux'
        else:
            self.operating_system = 'mac'


    def get_weights(self, edge_list):
        weights = []
        for edge in edge_list:
            word_1 = self.word_index[edge[0]]
            word_2 = self.word_index[edge[1]]
            #try:
            v1 = self.embeddings[word_1]
            v2 = self.embeddings[word_2]
            w = 1 - spatial.distance.cosine(v1, v2)
            #except:
            #    w = -1
            weights.append(w)
        return weights

    def create_network(self):
        string_bigrams = bigrams(self.document)
        edges = []
        for i in string_bigrams:
            edges.append((i[0],i[1]))
        print('Words:',len(self.words), 'Document:', len(self.document))
        #print(self.document)
        #print(self.words)
        network = Graph()
        network.add_vertices(self.words)
        network.add_edges(edges)
        network.simplify()
        edge_list = network.get_edgelist()
        weights = self.get_weights(edge_list)
        network.es['weight'] = weights
        return network

    def create_network_janela(self, window):
        matrix = np.zeros((len(self.words), len(self.words)))
        for index, word in enumerate(self.document):
            neighbors = utils.get_neighbors(self.document, index, window)
            word_index = self.vocab_index[word]
            for neighbor in neighbors:
                neighbor_index = self.vocab_index[neighbor]
                matrix[word_index][neighbor_index] = 1
        np.fill_diagonal(matrix, 0)
        network = igraph.Graph.Adjacency(matrix.tolist(), mode="undirected")
        print('Nodes:', len(self.words), '-', 'Edges:', len(network.get_edgelist()))
        weights = self.get_weights(network.get_edgelist())
        network.vs['name'] = self.words
        network.es['weight'] = weights
        return network

    def create_complete_network(self):
        print('Creating complete network')
        matrix = []
        for word in self.words:
            embedding = self.embeddings[word]
            matrix.append(embedding)
        matrix = np.array(matrix)
        similarity_matrix = 1 - pairwise_distances(matrix, metric='cosine')
        similarity_matrix[similarity_matrix<=0.3] = 0
        #similarity_matrix[similarity_matrix <= 0.5] = 0
        simList = similarity_matrix.tolist()
        grafo = Graph.Weighted_Adjacency(simList, mode="undirected", attr="weight", loops=False)
        pesos = grafo.es['weight']
        print('size grafo:', grafo.vcount())
        print('aristas',len(pesos))
        return grafo

    def add_embeddings(self, network): # con bert no va a dar cierto :(
        network_size = network.vcount()
        actual_edges = network.get_edgelist()
        num_edges = network.ecount()
        original_weight = network.es['weight']
        maximum_num_edges = int((network_size * (network_size - 1)) / 2)
        remaining_edges = maximum_num_edges - num_edges
        edges_to_add = []
        for percentage in self.percentages:
            value = int(num_edges * percentage / 100) + 1
            edges_to_add.append(value)
        print('Edges to add:', edges_to_add)
        matrix = []
        for word in self.words: ### error com BERT :(
            embedding = self.embeddings[word]
            matrix.append(embedding)
        matrix = np.array(matrix)
        similarity_matrix = 1 - pairwise_distances(matrix, metric='cosine')
        similarity_matrix[np.triu_indices(network_size)] = -1
        similarity_matrix[similarity_matrix == 1.0] = -1
        largest_indices = utils.get_largest_indices(similarity_matrix, maximum_num_edges)
        #print('testing largest indices')
        #print(largest_indices)
        #a = input()
        max_value = np.max(edges_to_add)
        counter = 0
        index = 0
        new_edges = []
        new_weights = []
        while counter < max_value:
            try:
                x = largest_indices[0][index]
                y = largest_indices[1][index]
                if not network.are_connected(x, y):
                    new_edges.append((x, y))
                    word_1 = self.word_index[x]
                    word_2 = self.word_index[y]
                    v1 = self.embeddings[word_1]
                    v2 = self.embeddings[word_2]
                    w = 1 - spatial.distance.cosine(v1, v2)
                    new_weights.append(w)
                    counter += 1
                index += 1
            except:
                print('Red completa')
                counter=max_value
                #a = input()
        networks = []
        for value in edges_to_add:
            edges = []
            weights = []
            edges.extend(actual_edges)
            weights.extend(original_weight)
            edges.extend(new_edges[0:value])
            weights.extend(new_weights[0:value])
            new_network = Graph()
            new_network.add_vertices(self.words)
            new_network.add_edges(edges)
            new_network.es['weight'] = weights
            networks.append(new_network)
        return networks

    def get_embedding_networks(self, network):
        networks = self.add_embeddings(network)
        networks.insert(0, network)
        prueba = [len(net.get_edgelist()) for net in networks]
        print('Num edges in networks:', prueba)
        return networks

    def accessibility(self, network, h):
        in_network = 'extra/auxiliar_network.xnet'
        extra_file = 'extra/acc_results.txt'
        xnet.igraph2xnet(network, in_network)
        if self.operating_system == 'linux':
            path_command = './accessibility/CVAccessibility_linux -l ' + h + ' ' + in_network + ' > ' + extra_file
        else:
            path_command = './accessibility/CVAccessibility_mac -l ' + h + ' ' + in_network + ' > ' + extra_file

        os.system(path_command)
        accs_values2 = utils.read_result_file_v2(extra_file)
        return accs_values2

    def sort_words(self, network, only_weighted=False):
        stg = network.strength(weights=network.es['weight'])
        pr_w = network.pagerank(weights=network.es['weight'])
        cc_w = network.transitivity_local_undirected(weights=network.es['weight'])
        weight = np.array(network.es['weight'])
        weight[weight < 0] = 0.0001
        btw_w = network.betweenness(weights=weight)
        clos_w = network.closeness(weights=weight)
        accs = self.accessibility(network, '1')
        accs2 = self.accessibility(network, '2')
        eigen_w = network.eigenvector_centrality(weights=network.es['weight'])

        if only_weighted:
            measures = [stg, pr_w, cc_w, btw_w, clos_w, accs, accs2, eigen_w]
            #measures = [stg, pr_w, accs, accs2, eigen_w]
        else:
            dgr = network.degree()
            pr = network.pagerank()
            btw = network.betweenness()
            cc = network.transitivity_local_undirected()
            clos = network.closeness()
            eigen = network.eigenvector_centrality()
            measures = [dgr, stg, pr, pr_w, btw, btw_w, cc, cc_w, clos, clos_w, accs, accs2, eigen, eigen_w]
            #measures = [dgr, pr]
        result = []
        for val in measures:
            result.append(utils.get_keywords(val, self.words))
        return result





if __name__ == '__main__':

    v1 = np.array([1,2,3])
    v2 = np.array([30,20,10])

    print(cosine_similarity([v1], [v2]))
    print(1 - spatial.distance.cosine(v1, v2))