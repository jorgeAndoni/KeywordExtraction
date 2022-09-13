'''
from bert_embedding import BertEmbedding
bert_abstract = """We introduce a new language representation model called BERT, which stands for Bidirectional Encoder Representations from Transformers.
 Unlike recent language representation models, BERT is designed to pre-train deep bidirectional representations by jointly conditioning on both left and right context in all layers.
 As a result, the pre-trained BERT representations can be fine-tuned with just one additional output layer to create state-of-the-art models for a wide range of tasks, such as question answering and language inference, without substantial task-specific architecture modifications. 
BERT is conceptually simple and empirically powerful. 
It obtains new state-of-the-art results on eleven natural language processing tasks, including pushing the GLUE benchmark to 80.4% (7.6% absolute improvement), MultiNLI accuracy to 86.7 (5.6% absolute improvement) and the SQuAD v1.1 question answering Test F1 to 93.2 (1.5% absolute improvement), outperforming human performance by 2.0%."""
sentences = bert_abstract.split('\n')

for sentence in sentences:
    print(sentence)

print('Training ?')

bert_embedding = BertEmbedding()
result = bert_embedding(sentences)

print('size result:',len(result))

first_sentence = result[0]
text = first_sentence[0]
word_vectors = first_sentence[1]
print(len(text),text)
for index, vector in enumerate(word_vectors):
    print(index+1, vector.shape)
'''

import torch
from transformers import BertTokenizer, BertModel
from scipy.spatial.distance import cosine
import matplotlib.pyplot as plt
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

text = "After stealing money from the bank vault, the bank robber was seen " \
       "fishing on the Mississippi river bank. " \
       "My name is Andoni Valverde Tohalino bye" # probar con palabras q no estan en el vocabulario

marked_text = "[CLS] " + text + " [SEP]"
tokenized_text = tokenizer.tokenize(marked_text)
indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text) # indices en el voc de los tokens
segments_ids = [1] * len(tokenized_text) # para id de las sentencias

#Training
tokens_tensor = torch.tensor([indexed_tokens])
segments_tensors = torch.tensor([segments_ids])
model = BertModel.from_pretrained('bert-base-uncased',
                                  output_hidden_states = True, # Whether the mode l returns all hidden-states.
                                  )
model.eval()

# Run the text through BERT, and collect all of the hidden states produced from all 12 layers.
with torch.no_grad():
    outputs = model(tokens_tensor, segments_tensors)
    hidden_states = outputs[2]

#print(hidden_states)
# 4 dimensiones -> 13 layers - 1 batch number(sentence) - 22 words in sentence - 768 features


# `hidden_states` is a Python list.
print('      Type of hidden_states: ', type(hidden_states))
# Each layer in the list is a torch tensor.
print('Tensor shape for each layer: ', hidden_states[0].size())

token_embeddings = torch.stack(hidden_states, dim=0)
print(token_embeddings.size())
# Remove dimension 1, the "batches". ##
token_embeddings = torch.squeeze(token_embeddings, dim=1)
print(token_embeddings.size())
# Swap dimensions 0 and 1.
token_embeddings = token_embeddings.permute(1,0,2)
print(token_embeddings.size())

# Stores the token vectors, with shape [22 x 768]
token_vecs_sum = []
# `token_embeddings` is a [22 x 12 x 768] tensor.
for token in token_embeddings:
    # `token` is a [12 x 768] tensor
    # Sum the vectors from the last four layers
    sum_vec = torch.sum(token[-4:], dim=0)
    token_vecs_sum.append(sum_vec)
print ('Shape is: %d x %d' % (len(token_vecs_sum), len(token_vecs_sum[0])))

for i, (ind_tok,token_str) in enumerate(zip(indexed_tokens,tokenized_text)):
  print (i, ind_tok, token_str)

#Testing the word embeddings of word 'bank' Acceder as sus indices 6,10,19
print('\nFirst 5 vector values for each instance of "bank".')
print('')
print("bank vault   ", str(token_vecs_sum[6][:5]))
print("bank robber  ", str(token_vecs_sum[10][:5]))
print("river bank   ", str(token_vecs_sum[19][:5]))

diff_bank = 1 - cosine(token_vecs_sum[10], token_vecs_sum[19])
same_bank = 1 - cosine(token_vecs_sum[10], token_vecs_sum[6])
print('Vector similarity for  *similar*  meanings:  %.2f' % same_bank)
print('Vector similarity for *different* meanings:  %.2f' % diff_bank)