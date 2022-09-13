
import os
from sklearn.feature_extraction import stop_words
from nltk import word_tokenize, sent_tokenize
import string
import nltk
from nltk.stem import PorterStemmer
from collections import Counter
import pickle
import pandas as pd
import numpy as np

def remove_mac_strings(lista):
    trash = ['__MACOSX', '.DS_Store']
    if trash[0] in lista: lista.remove(trash[0])
    if trash[1] in lista: lista.remove(trash[1])
    return lista

def read_document(path):
    tags = dict()
    tags['FW'] = 'foreign word'
    tags['JJ'] = 'adjective'
    tags['JJR'] = 'adjective, comparative'
    tags['JJS'] = 'adjective, superlative'
    tags['NN'] = 'noun, singular'
    tags['NNS'] = 'noun plural '
    tags['NNP'] = 'proper noun, singular'
    tags['NNPS'] = 'proper noun, plural'

    stop_set = stop_words.ENGLISH_STOP_WORDS
    content = open(path, 'r').read()
    content = content.lower()
    for c in string.punctuation:
        content = content.replace(c, " ")
    content = word_tokenize(content)
    words = []
    for word in content:
        if not word in stop_set and len(word)>2:
            words.append(word)

    ps = PorterStemmer()
    tagged = nltk.pos_tag(words)
    only_nouns = []
    all_words = []
    for pair in tagged:
        word = pair[0]
        tag = pair[1]
        stemmed = ps.stem(word)
        if tag in tags:
            only_nouns.append(stemmed)
        all_words.append(stemmed)
    return only_nouns, all_words


def process_keywords(lista):
    stop_set = stop_words.ENGLISH_STOP_WORDS
    ps = PorterStemmer()
    return [ps.stem(word) for word in lista if not word in stop_set]


def process_text(content):
    ruin_words = ['not']
    content = ' '.join(content.split())
    content = content.lower()
    for c in string.punctuation:
        content = content.replace(c, " ")
    words = word_tokenize(content)
    lista = [word for word in words if not word in ruin_words]
    return ' '.join(lista)


def read_document_v2(path):
    content = open(path, 'r').read()
    content = ' '.join(content.split())
    sentences = sent_tokenize(content)
    print('Num sentences:',len(sentences))
    content = ''
    for sentence in sentences:
        sentence = process_text(sentence)
        #if sentence!=' ':
        if len(sentence)>=2:
            content+=sentence + '\n'
    content = content[:-1]
    return content

def db_hulth_reading(): # 500 documents - uncontroled keywords
    path = 'databases/Hulth2003/'
    files = os.listdir(path)
    files = remove_mac_strings(files)
    path_dictionary = dict()
    for file in files:
        file_id = file[:file.find('.')]
        if file_id in path_dictionary:
            pass
        else:
            path_abst = path + str(file_id) + '.abstr'
            path_keywords = path + str(file_id) + '.uncontr'
            path_dictionary[file_id] = [path_abst, path_keywords]

    #result = dict()
    paper_ids = []
    all_texts = []
    all_nouns = []
    all_keywords = []
    all_sentences = []
    for index, file in enumerate(path_dictionary):
        documents = path_dictionary[file]
        abstract_nouns, abstract_all = read_document(documents[0])
        all_content_sentences = read_document_v2(documents[0])

        keywords = open(documents[1], 'r').read()
        keywords = keywords.replace('\n\t', ' ')
        keywords = keywords.replace('\n', ' ')
        keywords = keywords.replace(';', '')
        keywords = keywords.replace('-', ' ')
        keywords = keywords.lower()
        keywords = word_tokenize(keywords)
        keywords = list(set(keywords))
        keywords = process_keywords(keywords)
        #result[file] = [abstract_all, abstract_nouns, keywords]
        paper_ids.append(file)
        all_sentences.append(all_content_sentences)
        all_texts.append(abstract_all)
        all_nouns.append(abstract_nouns)
        all_keywords.append(keywords)

    d = {'paper_id': paper_ids, 'all_content':all_texts, 'only_nouns':all_nouns, 'keywords':all_keywords, 'all_content_sentences':all_sentences}
    df = pd.DataFrame(data=d)
    df.to_csv('databases/hult_db_v3.csv')
    #return result

def db_semeval_reading(): # 100 documents  # procesamiento medio raro
    stop_set = stop_words.ENGLISH_STOP_WORDS
    path_abst = 'databases/SemEval2010/test/'
    path_keys = 'databases/SemEval2010/keywords.txt'
    files = os.listdir(path_abst)
    files = remove_mac_strings(files)
    path_dictionary = dict()
    for doc in files:
        doc_id = doc[:doc.find('.')]
        path_dictionary[doc_id] = path_abst + doc
    doc_keys = open(path_keys, 'r').readlines()

    paper_ids = []
    all_texts = []
    all_nouns = []
    all_keywords = []
    for index, line in enumerate(doc_keys):
        print(index+1)
        line = line.rstrip('\n')
        key = line[:line.find(':') - 1]
        #key_list = (line[line.find(':') + 2:]).split(',')
        keywords = line[line.find(':') + 2:]
        keywords = keywords.replace('-', ' ')
        keywords = keywords.replace(',', ' ')
        keywords = keywords.replace('+', ' ')
        keywords = word_tokenize(keywords)
        keywords = [word for word in keywords if not word in stop_set and len(word)>2]
        keywords = list(set(keywords))
        abstract_nouns, abstract_all = read_document(path_dictionary[key])
        all_content_sentences = read_document_v2(path_dictionary[key])
        print(key, len(abstract_nouns), len(abstract_all))
        print(abstract_nouns)
        print(len(all_content_sentences))

        #a = input()

        #print(line)
        #print(keywords)
        paper_ids.append(key)
        all_texts.append(abstract_all)
        all_nouns.append(abstract_nouns)
        all_keywords.append(keywords)
    d = {'paper_id': paper_ids, 'all_content': all_texts, 'only_nouns': all_nouns, 'keywords': all_keywords}
    df = pd.DataFrame(data=d)
    #df.to_csv('databases/semEval_db.csv')



def load_data_from_disk(file):
    print('loading data from disk ....')
    with open(file, 'rb') as fid:
        data = pickle.load(fid)
    print('data loaded!')
    return data

def get_full_abstracts():
    path = 'databases/processed_all_abstracts.pk'
    data = load_data_from_disk(path)
    ps = PorterStemmer()
    print('Reading training abstracts ...')
    paper_ids = []
    contents = []
    for index, i in enumerate(data):
        abstract = data[i][0]
        abstract = [ps.stem(word) for word in abstract]
        contents.append(abstract)
        paper_ids.append(i)
        print(index, i,  len(abstract))
    d = {'paper_id': paper_ids, 'all_content': contents}
    df = pd.DataFrame(data=d)
    df.to_csv('databases/abstracts_silva.csv')
    #return corpus

#p1['processed'] = p1['texts_with_stop_words'].apply(text_processing_v2)

def str_to_list(i):
    i = i.replace('[', '')
    i = i.replace(']', '')
    i = i.replace('\'', '')
    i = i.replace(',', '')
    return word_tokenize(i)
    #return i.split(',')

def read_csv_column(file, column):
    column_list = list(file[column].apply(str_to_list))
    return column_list

def get_keywords(values, words):
    sort_values = (-np.array(values)).argsort()
    return [words[index] for index in sort_values]

def get_taxas(references, automatic_keys):
    n = len(references)
    #result = dict()
    result = []
    #for measure in automatic_keys:
    for measure in automatic_keys:
        #keys = automatic_keys[measure][0:n]
        keys = measure[0:n]
        taxa = len(set(keys) & set(references)) /n
        #result[measure] = taxa
        result.append(taxa)
    return result

def get_neighbors(word_list, index, w):
  if index - w >= 0:
    left = word_list[index - w:index]
  else:
    left = word_list[:index]
  right = word_list[index + 1:index + 1 + w]
  return list(set(left + right))

def read_result_file(path):
  index = 0
  result = dict()
  file = open(path)
  for line in file.readlines():
    line = line.rstrip('\n')
    value = float(line)
    result[index] = value
    index += 1
  return result

def read_result_file_v2(path):
  result = []
  file = open(path)
  for line in file.readlines():
    line = line.rstrip('\n')
    value = float(line)
    result.append(value)
  return result

def get_largest_indices(ary, n):
    """Returns the n largest indices from a numpy array."""
    flat = ary.flatten()
    indices = np.argpartition(flat, -n)[-n:]
    indices = indices[np.argsort(-flat[indices])]
    return np.unravel_index(indices, ary.shape)


def get_words(sentence, tokenized):
    tokenized.remove('[CLS]')
    tokenized.remove('[SEP]')
    print(sentence)
    print(tokenized)
    index_sentence = 0
    buffer_word = ''
    for token in tokenized:
        if len(buffer_word)!=0:
            actual_word = buffer_word
        else:
            actual_word = sentence[index_sentence]
        print('Token:', token)
        print('Actual word:', actual_word)
        if token == actual_word:
            print('ok:',token, actual_word)
            index_sentence+=1
        else:
            token = token.replace('#', '')
            buffer_word+=token
        print('Buffer:', buffer_word)
        print()

def get_words_v2(sentence, tokenized):
    tokenized.remove('[CLS]')
    tokenized.remove('[SEP]')
    print(sentence)
    print(tokenized)
    buffer = ''
    results = []
    for word in sentence:
        top = tokenized.pop(0)
        if word == top:
            print(word, top)
            results.append([top])
        else:
            vector = []
            vector.append(top)
            top = top.replace('##', '')
            buffer+=top
            while buffer != word:
                top = tokenized.pop(0)
                vector.append(top)
                top = top.replace('##', '')
                buffer += top
            results.append(vector)
            print(word, buffer)
            buffer=''
    print()
    for i in results:
        print(i)

def vec_to_str(averages):
    content = ''
    for value in averages:
        value = str(value)
        value = value.replace('.', ',')
        content+=value + '#'
    content = content[:-1]
    return content


if __name__ == '__main__':

    sentence = ['hello', 'my', 'name', 'is', 'jorge', 'andoni', 'hi' ,'valverde', 'tohalino', ',', 'bye', '.']
    tokenized = ['[CLS]', 'hello', 'my', 'name', 'is', 'jorge', 'and', '##oni', 'hi' ,'valve', '##rde', 'to', '##hal', '##ino', ',', 'bye', '.', '[SEP]']

    #get_words_v2(sentence, tokenized)
    #db_hulth_reading()
    db_semeval_reading()

