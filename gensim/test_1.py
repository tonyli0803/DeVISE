
# coding: utf-8

# In[2]:


#reference   https://www.jianshu.com/p/52ee8c5739b6


# In[3]:


import gensim
import os
import time
import multiprocessing


# In[4]:


sentences = [['first', 'sentence'], ['second', 'sentence']]
# train word2vec on the two sentences
model = gensim.models.Word2Vec(sentences, min_count=1)


# In[5]:


class MySentences(object):
    def __init__(self, dirname):
        self.dirname = dirname
 
    def __iter__(self):
        for fname in os.listdir(self.dirname):
            for line in open(os.path.join(self.dirname, fname)):
                yield line.split()
 
sentences = MySentences('./data') # a memory-friendly iterator



# In[17]:


t0 = time.time()
model = gensim.models.Word2Vec(sentences,workers = 8)
t1 = time.time()


# In[18]:


print(t1 - t0)


# In[8]:


print((model["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]).shape)


# In[9]:


print(model["airplane"])


# In[10]:


print(len(model.wv.vocab))


# In[25]:


#/home/tony/Desktop/NLP/gensim/my_trained_model/     I save my model here

#save model!!!
model.save("/tmp/mymodel")




# In[11]:


#model = gensim.models.Word2Vec.load('./mymodel/mymodel')


# In[12]:


#gmodel = gensim.models.KeyedVectors.load_word2vec_format('./google_model/GoogleNews-vectors-negative300.bin', binary=True)


# In[13]:


#print((gmodel["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]).shape)

