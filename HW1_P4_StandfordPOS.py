#!/usr/bin/env python
# coding: utf-8

# In[2]:


import re
import numpy as np

from nltk import pos_tag

def words_from_sentences(sentence):
    return(sentence.split(' '))

def POScheck(sentence):
    sent = words_from_sentences(sentence)
    return (pos_tag(sent))

S1 = 'The chairman of the board is completely bold .'
pos_s1 = POScheck(S1)
print("pos for: ",S1," => ",pos_s1,'\n\n')

S2 = 'A chair was found in the middle of the road .'
pos_s2 = POScheck(S2)
print("pos for: ",S2," => ",pos_s2,'\n\n')


want_to_try = input('if you want to try custom string press Y\y: \n')

if (want_to_try == 'Y' or want_to_try == 'y'): 
    demo_sent = input("Sentence ")
    pos_s3 = POScheck(demo_sent)
    print("pos for: ",demo_sent," => ",pos_s3,'\n\n')


# In[ ]:





# In[ ]:




