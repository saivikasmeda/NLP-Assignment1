#!/usr/bin/env python
# coding: utf-8

# In[5]:


import re
import numpy as np
import json


corpus = 'corpus_for_language_models.txt'

# sentences from file to array
r = open(corpus,'r')
sentences = r.readlines()


# In[3]:


# adding start and end mark
big_one_string=''
for i in range(len(sentences)):
    sentences[i] = '<s> ' + sentences[i][:-2] + ' </s>'
    big_one_string += sentences[i] + ' '

# Assuming the case all the words in the sentences are split by _ (a single space)
words = (big_one_string.split(' '))

corpus_dict = {}
index=0
for word in words:
    if word in corpus_dict.keys():
        corpus_dict[word]['count'] +=1
    else:
        corpus_dict[word]={'count':1,'index':index}
        index+=1


# In[4]:


# to store the bigrams we can use the indexes from the sentences so and store in 2 d array

unique_tokens = corpus_dict.__len__()
Matrix = np.zeros((unique_tokens,unique_tokens))

# Bigrams here we consider the left (row) as the fist word and top (column) as the second word for the bigrams
for i in range(len(words)-1):
    left = words[i]
    top = words[i+1]
    l_index = corpus_dict[left]['index']
    t_index = corpus_dict[top]['index']
    Matrix[l_index][t_index]+=1



# In[6]:


unique_keys = list(corpus_dict.keys())

bigramCounts ={}
for i in range(len(Matrix)):
    for j in range(len(Matrix)):
        if Matrix[i][j] > 0:
            bigramCounts['('+unique_keys[i]+','+unique_keys[j]+')']=Matrix[i][j]
with open('Bigram_Counts.json', 'w') as fp:
    json.dump(bigramCounts, fp)


for i in corpus_dict.values():
    index = i['index']
    count = i['count']
    Matrix[index]/=count

S1 = '<s> Sales of the company to return to normalcy . </s>'
S2 = '<s> The new products and services contributed to increase revenue . </s>'


# In[7]:


def words_from_sentences(sentence):
    return(sentence.split(' '))

def bigrams_count(words,corpus_dict,Matrix):
    Bis = 1
    for i in range(len(words)-1):
        left = words[i]
        top = words[i+1]
        if left in corpus_dict.keys() and top in corpus_dict.keys():
            l_index = corpus_dict[left]['index']
            t_index = corpus_dict[top]['index']
            Bis*=(Matrix[l_index][t_index])
            print('('+left+','+top+')='+ str(Matrix[l_index][t_index]))
        else:
            print('word not found in corp')
            Bis = 0
    return Bis



# In[8]:




s1_words = words_from_sentences(S1)
Bis1 = bigrams_count(s1_words,corpus_dict,Matrix)

print("Bigram probablity for: ",S1," => ",Bis1,'\n\n')


s2_words = words_from_sentences(S2)
Bis2 = bigrams_count(s2_words,corpus_dict,Matrix)


print("Bigram probablity for: ",S2," => ",Bis2,'\n\n')

np.savetxt("BigramMatrix.xlsx", Matrix, delimiter=",")

# In[10]:


'''
# this section is to let user check and try their own sentences
want_to_try = input('if you want to try custom string press Y\y: \n')

if (want_to_try == 'Y' or want_to_try == 'y'):
    demo_sent = input("Sentence ")
    demo_sent = '<s> '+demo_sent+' </s>'
    d_words = words_from_sentences(demo_sent)
    Bis1 = bigrams_count(d_words,corpus_dict,Matrix)
    print("Bigram probablity for: ",demo_sent," => ",Bis1,'\n\n')
'''


# In[9]:



# now for Laplacian Normalization
unique_tokens = corpus_dict.__len__()
MatrixLaplace = np.zeros((unique_tokens,unique_tokens))
# Bigrams here we consider the left (row) as the fist word and top (column) as the second word for the bigrams
for i in range(len(words)-1):
    left = words[i]
    top = words[i+1]
    l_index = corpus_dict[left]['index']
    t_index = corpus_dict[top]['index']
    MatrixLaplace[l_index][t_index]+=1

# Adding one smoothing
MatrixLaplace+=1

for i in corpus_dict.values():
    index = i['index']
    count = i['count']
    # Adding one Smoothing will change the count so we add count with unique number of words
    MatrixLaplace[index]/=(count + unique_tokens)



print('\n\n\n\n\n')


# In[11]:



s1_words = words_from_sentences(S1)
Bis1 = bigrams_count(s1_words,corpus_dict,MatrixLaplace)

print("Bigram probablity after Laplacian normalcy for: ",S1," => ",Bis1,'\n\n')


s2_words = words_from_sentences(S2)
Bis2 = bigrams_count(s2_words,corpus_dict,MatrixLaplace)



print("Bigram probablity after Laplacian normalcy for: ",S2," => ",Bis2,'\n\n')

np.savetxt("BigramMatrixLaplace.xlsx", MatrixLaplace, delimiter=",")
# In[ ]:



'''
want_to_try = input('if you want to try custom string press Y\y: \n')

if (want_to_try == 'Y' or want_to_try == 'y'):
    demo_sent = input("Sentence ")
    demo_sent = '<s> '+demo_sent+' </s>'
    d_words = words_from_sentences(demo_sent)
    Bis1 = bigrams_count(d_words,corpus_dict,MatrixLaplace)
    print("Bigram probablity after Laplacian normalcy for: ",demo_sent," => ",Bis1,'\n\n')
'''


# In[ ]:



'''
#  helps in reuability of trained data
want_to_print = input('if you want to print the bigram matrix and dictionary press Y\y: \n')

if (want_to_print == 'Y' or want_to_print == 'y'):
    # documenting the Data dictionary and the bigram matrix so that we can reuse it or for evealuation purpose
    import json
    with open('Q2Word_Index.json', 'w') as fp:
        json.dump(corpus_dict, fp)
    np.savetxt("BigramMatrix.csv", Matrix, delimiter=",")
    np.savetxt("BigramMatrixLaplace.csv", MatrixLaplace, delimiter=",")
'''

