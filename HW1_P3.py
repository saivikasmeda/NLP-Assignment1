#!/usr/bin/env python
# coding: utf-8

# In[1]:


import re
import math
import numpy as np
from nltk.tokenize import word_tokenize
# import pandas as pd


# sentences from file to array
r = open('corpus_for_language_models.txt','r')
sentences = r.readlines()


# In[2]:


# print(sentences)
# Preprocessing retaining words and numbers
for i in range(len(sentences)):
    sentences[i] = re.findall(r'[^\s]*[\w]+[^\s]*',sentences[i].lower())


# In[3]:


def PPMI_nil_padding(contexts,words):
    word_matrix = np.zeros((len(words),len(contexts)),dtype = 'double')
    for sentence in sentences:
        for w in range(len(words)):
            word = words[w]
            if word in sentence:
                word_index = sentence.index(word)
                for c in range(len(contexts)):
                    context = contexts[c]
                    if context in sentence:
                        context_index = sentence.index(context)
                        if((word_index+5)>= context_index and (word_index-5)<= context_index):
                            word_matrix[w][c]+=1
#     print("count \n",word_matrix)
    N = sum(sum(word_matrix))
    word_matrix/=N
    return word_matrix
    
    


# In[4]:


def PPMI_two_padding(contexts , words):
    word_matrix = np.zeros((len(words),len(contexts)),dtype = 'double')
    for sentence in sentences:
        for w in range(len(words)):
            word = words[w]
            if word in sentence:
                word_index = sentence.index(word)
                for c in range(len(contexts)):
                    context = contexts[c]
                    if context in sentence:
                        context_index = sentence.index(context)
                        if((word_index+5)>= context_index and (word_index-5)<= context_index):
                            word_matrix[w][c]+=1
    word_matrix +=2
#     print("count \n",word_matrix)
    N = sum(sum(word_matrix))
    word_matrix/=N
    return word_matrix
    


# In[5]:


def PPMI_WordContext_Matrix(contexts,words,word_context_matrix):
    words_sum = word_context_matrix.sum(axis=1)
    contexts_sum = word_context_matrix.sum(axis=0)
    pmi_nil_matrix = np.zeros((len(words),len(contexts)), dtype= 'double')
    for w in range(len(words)):
        for c in range(len(contexts)):
            pmi_nil_matrix[w][c] = max(math.log2(word_context_matrix[w][c]/(words_sum[w]*contexts_sum[c])),0)
            print(words[w], " with the context of ", contexts[c], ": ", pmi_nil_matrix[w][c])
    return pmi_nil_matrix


# In[6]:


contexts = ["said","of","board"]
words = ["chairman","company"]

print("With nil padding")
word_nil_matrix = PPMI_nil_padding(contexts,words)
# word_nil_matrix_df =  pd.DataFrame(word_nil_matrix, columns = contexts, index=words)
# print("\n\n")
# print(word_nil_matrix_df)

ppmi_nil_word_matrix = PPMI_WordContext_Matrix(contexts,words,word_nil_matrix)
print("\n\n")
print("with two padding")
word_two_matrix = PPMI_two_padding(contexts,words)
# word_two_matrix_df =  pd.DataFrame(word_two_matrix, columns = contexts, index=words)
# print("\n\n")
# print(word_two_matrix_df)
# print("\n\n")

ppmi_two_word_matrix = PPMI_WordContext_Matrix(contexts,words,word_two_matrix)




# In[7]:


def Count_Matrix(contexts,words):
    word_matrix = np.zeros((len(words),len(contexts)),dtype = 'double')
    for sentence in sentences:
        for w in range(len(words)):
            word = words[w]
            if word in sentence:
                word_index = sentence.index(word)
                for c in range(len(contexts)):
                    context = contexts[c]
                    if context in sentence:
                        context_index = sentence.index(context)
                        if((word_index+5)>= context_index and (word_index-5)<= context_index):
                            word_matrix[w][c]+=1
    word_matrix +=2
    return word_matrix


# In[8]:



# HW3_3 chairman,company,sales,economy   Cosine Similarity

contexts_3 = ["said","of","board"]
words_3 = ["chairman","company","sales","economy"]

print( "for words: ",words_3)
print("in contexts: ",contexts_3)
print("// using 2padding")
word3_two_matrix = Count_Matrix(contexts_3,words_3)
# ppmi3_two_word_matrix = PPMI_WordContext_Matrix(contexts_3,words_3,word3_two_matrix)

print(word3_two_matrix)


# In[9]:


def similarityCalculation(vector):
    dim = len(vector)
    similarity_matrix = np.zeros((dim,dim))
    for i in range(dim):
        for j in range(dim):
            similarity_matrix[i][j] = np.dot(vector[i],vector[j]/(np.sqrt(np.dot(vector[i],vector[i]))*np.sqrt(np.dot(vector[j],vector[j]))))
    return similarity_matrix


# In[10]:


similarity = similarityCalculation(word3_two_matrix)
max_i = 0
max_j = 0
max_similarity = 0
for i in range(len(words_3)):
    for j in range(i+1,len(words_3)):
        if(max_similarity < similarity[i][j]):
            max_similarity = similarity[i][j]
            max_i = i
            max_j = j
        print(words_3[i],words_3[j]," Similarity: ",similarity[i][j])

print("\n\n")
# print("Max similarity is between: (",words_3[max_i], ") and (",words_3[max_j] , ") with value of: ",max_similarity)


# In[11]:


# Problem3 4 Glove data
# sentences from file to array
glove_data = open('glove.6B.50d.txt','r')

glove_words = ["chairman","company","sales","economy"]

ppmi_glove_dict = {}
count = 0

for line in glove_data:
    word_mat = line.split(" ")
    if("chairman" in ppmi_glove_dict and "company" in ppmi_glove_dict and "sales" in ppmi_glove_dict and "economy" in ppmi_glove_dict):
        break
    if(word_mat[0]=="chairman" or word_mat[0]=="company" or word_mat[0]=="sales" or word_mat[0]=="economy"):
        arr = word_mat[1:]
        arr[-1] = arr[-1].strip()
        ppmi_glove_dict[word_mat[0]] =  [float(mat) for mat in arr]
ppmi_glove_matrix = [ppmi_glove_dict["chairman"],ppmi_glove_dict["company"],ppmi_glove_dict["sales"],ppmi_glove_dict["economy"]]
print(ppmi_glove_matrix)


# In[12]:


similarity = similarityCalculation(ppmi_glove_matrix)
max_i = 0
max_j = 0
max_similarity = 0
for i in range(len(words_3)):
    for j in range(i+1,len(words_3)):
        if(max_similarity < similarity[i][j]):
            max_similarity = similarity[i][j]
            max_i = i
            max_j = j
        print(words_3[i],words_3[j]," Similarity: ",similarity[i][j])

print("\n\n")


