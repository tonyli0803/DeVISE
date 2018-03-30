
# coding: utf-8

# In[1]:


# To support both python 2 and python 3
from __future__ import division, print_function, unicode_literals

from tqdm import tqdm
import os
import os.path
import multiprocessing
import time
import sys

import numpy as np
import tensorflow as tf
import gensim
import tflearn
from tflearn.datasets import cifar10
from tflearn.data_utils import shuffle, to_categorical

sys.path.append("./vgg")

import input_data
import VGG
import tools
import pickle


###parameters
# In[3]:
IMG_W = 32
IMG_H = 32
N_CLASSES = 10
BATCH_SIZE = 1
learning_rate = 0.01
MAX_STEP = 10000   # it took me about one hour to complete the training.
IS_PRETRAIN = True
word_vector_dim = 200
# In[4]:



def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo)
    return dict


# In[5]:

###load w2v model
#gmodel = gensim.models.KeyedVectors.load_word2vec_format('./gensim/google_model/GoogleNews-vectors-negative300.bin', binary=True)
#print((gmodel["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]).shape)
#model = gensim.models.Word2Vec.load('./gensim/mymodel/mymodel')
model = gensim.models.Word2Vec.load("./gensim/w2v_database/wiki.en.text.model")   ###read w2v model from path
#model = gensim.models.Word2Vec.load('./gensim/w2v_database/wiki.en.text.utf-8')
print((model["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]).shape)


# In[6]:

###load vgg visual model graph
label_string = unpickle('./batches.meta')['label_names']
label_string_vector = model[label_string]
print(len(model.wv.vocab))
print(label_string)


# In[7]:
tmparray = model[ (model.wv.vocab).keys() ]
#tmparray = tmparray[:2000]
# In[8]:
#print((model.wv.vocab).keys())
#print(tmparray)
#print(tmparray.shape)
keys =list((model.wv.vocab).keys() )
 
###print labels and their ids in the w2v model 
id_array = np.zeros(N_CLASSES)

word_id = 0
pointer = 0
for j in label_string:
    word_id = 0
    for i in (model.wv.vocab).keys():    
        if i==j:
            print('word = ',i,',       id = ',word_id)
            id_array[pointer] = word_id
            pointer = pointer + 1
            break
        word_id = word_id + 1
print("--end extract id--")



# In[10]:
##vector to word
#reference::https://stackoverflow.com/questions/32759712/how-to-find-the-closest-word-to-a-vector-using-word2vec
topn = 1
for i in label_string_vector:
    #print(i.shape)
    most_similar_words = model.most_similar( [i] , [], topn)
    print(most_similar_words)




# In[11]:
#%%   Test the accuracy on test dataset. got about 85.69% accuracy.

import math
def train():
    #with tf.device('/device:GPU:0'):
    with tf.Graph().as_default():

#        log_dir = 'C://Users//kevin//Documents//tensorflow//VGG//logsvgg//train//'
        log_dir = './vgg/logs/train/'

        #input where you put your binary cifar10 image
        Data_dir = '/home/tony/Desktop/Datasets/cifar_10_batches_binary_version/'

        
        train_log_dir = './devise_logs/train/'
        val_log_dir = './devise_logs/val/'
        #n_test the number of image data 
        n_test = 10000

        
        with tf.name_scope('input'):
            tra_image_batch, tra_label_batch = input_data.read_cifar10(data_dir=Data_dir,
                                                 is_train=True,
                                                 batch_size= BATCH_SIZE,
                                                 shuffle=True)
            val_image_batch, val_label_batch = input_data.read_cifar10(data_dir=Data_dir,
                                                 is_train=False,
                                                 batch_size= BATCH_SIZE,
                                                 shuffle=False)
        
        



        #images
        x = tf.placeholder(tf.float32, shape=[BATCH_SIZE, IMG_W, IMG_H, 3],name = 'input_x')

        #labels
        y_ = tf.placeholder(tf.int16, shape=[BATCH_SIZE, N_CLASSES],name = 'input_y_')
        

        
        logits = VGG.VGG16N(x, N_CLASSES, is_pretrain =False)
        saver = tf.train.Saver(tf.global_variables())
        
        id_array_tensor = tf.placeholder(tf.int32 ,[N_CLASSES])
        
        
        
        
        softmax_accuracy = tools.accuracy(logits, y_)
        vgg_predict = tf.argmax(logits,-1)
        vgg_predict_id = tf.gather_nd(id_array_tensor,vgg_predict)
        
        
        



        get_label_string_tensor = tf.placeholder(tf.float32 , shape = [None,word_vector_dim],name = "label_string")##word_vector_dim

        tmp = tf.matmul(tf.cast(y_,tf.float32),get_label_string_tensor)
        #print(tmp.shape)





        initializer = tf.contrib.layers.variance_scaling_initializer()
        fc7 = tf.get_default_graph().get_tensor_by_name("VGG16/fc7/Relu:0")

        fc8 = tf.layers.dense(inputs = fc7, units = 1024, kernel_initializer = initializer, name = "combination_hidden1")
        image_feature_output = tf.layers.dense(inputs = fc8, units = word_vector_dim, kernel_initializer = initializer, name = "combination_hidden2")


        #devise_loss

        tmparray_tensor = tf.placeholder(tf.float32 , shape = [None,word_vector_dim],name = 'tmparray')

        margin = 0.1
        #tMV here means that max (tJ *M* V - tLabel *M *V ,0 ) in essay
        #tmparray_tensor mearns tJ     and tmp means tmp
        tMV = tf.nn.relu( margin + tf.matmul((tmparray_tensor - tmp),tf.transpose(tf.cast(image_feature_output,tf.float32))))
        hinge_loss = tf.reduce_mean(tf.reduce_sum(tMV,0) , name = 'hinge_loss')

        train_step1 =tf.train.AdamOptimizer(0.0001, name="optimizer").minimize(hinge_loss)

        #tMV here means that tJ *M* V in essay
        tMV_ = tf.matmul(tmparray_tensor,tf.transpose(tf.cast(image_feature_output,tf.float32)))

        



        #accuracy
        predict_label = tf.argmax(tMV_, 0)
        predict_label = tf.cast(predict_label, tf.int32)
        predict_label = tf.reshape(predict_label,[-1,1],name = 'predict_label_text')

        #id_array_tensor = tf.placeholder(tf.int32 ,[N_CLASSES])
        select_id = tf.cast(tf.argmax(input = y_, axis = -1),tf.int32)
        select_id = tf.reshape(select_id,[1])
        y_label = tf.gather_nd(id_array_tensor,select_id)
        y_label = tf.reshape(y_label,[-1,1],name ='true_label_text')


        #y_label = tf.argmax(tf.matmul(tmparray_tensor , tf.transpose(tmp)), 0) #(2000,word_vector_dim)*(word_vector_dim,1)
        #y_label = tf.reshape(y_label,[-1,1])

        print(y_label.shape)
        print(predict_label.shape)


        acc,acc_op = tf.metrics.accuracy(labels = y_label, predictions = predict_label
                                         , weights=None, metrics_collections=None
                                         , updates_collections=None, name="acc")
        
        summary_op = tf.summary.merge_all() 
        
        saver2 = tf.train.Saver(tf.global_variables())
        
        with tf.Session() as sess:
            tra_summary_writer = tf.summary.FileWriter(train_log_dir, sess.graph)
            val_summary_writer = tf.summary.FileWriter(val_log_dir, sess.graph)
        
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())
            

            
            print("Reading checkpoints...")
            ckpt = tf.train.get_checkpoint_state(log_dir)
            if ckpt and ckpt.model_checkpoint_path:
                global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                #saver.restore(sess, ckpt.model_checkpoint_path)
                saver.restore(sess,"/home/tony/Desktop/DeVise/vgg/logs/train/model.ckpt-14999")
                print('Loading success, global_step is %s' % global_step)
            else:
                print('No checkpoint file found')
                return
##---------------------------------------------------------------Training-------------------------------------------------------------------------------------------------
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess = sess, coord = coord)
            print('\Triaining....')
            try:
                vgg_total_correct = 0
                for step in tqdm(range(MAX_STEP)):
                    if coord.should_stop():
                            break
                    #print("step %d"%step)
                    tra_images,tra_labels = sess.run([tra_image_batch, tra_label_batch])

                    loss,_,accuracy_,acc_operator,predict_label_,y_label_,summary_str,vgg_correct,vgg_predict_id_ = sess.run(
                                              [hinge_loss,train_step1,acc,acc_op,predict_label,y_label,summary_op,softmax_accuracy,vgg_predict_id],
                                              feed_dict = {get_label_string_tensor:label_string_vector,tmparray_tensor:tmparray,id_array_tensor:id_array,x:tra_images, y_:tra_labels})
                    
                    #print(vgg_correct)
                    if vgg_correct > 50 :
                        vgg_total_correct = vgg_total_correct + 1
                    
                    if step%100 == 0:
                        print("step %d"%step)
                        print('%d / %d steps'%(step,MAX_STEP),'loss = ',loss,'    acc = ',accuracy_,'\n\n')
                        print ('vgg predict acc  ',vgg_total_correct*1.0/(step+1),' ---->vgg predict     ',keys[int(vgg_predict_id_)])
                        print ('           devise predict_label',predict_label_,' ---->DeVise predict  ',keys[int(predict_label_)])
                        print ('                y_label             ',y_label_,      ' ---->ground  truth is',keys[int(y_label_)],'\n\n-------\n\n')
                    
                    
                    if step % 2000 == 0 or (step + 1) == MAX_STEP:
                        checkpoint_path = os.path.join(train_log_dir, 'model.ckpt')
                        saver2.save(sess, checkpoint_path, global_step=step)
                
            except tf.errors.OutOfRangeError:
                print('Done training -- epoch limit reached')
            finally:
                coord.request_stop()
                coord.join(threads)

        print("end training\n\n\n------------------------------------------\n")
##-----------------------------------------------------------------Testing---------------------------------------
        with tf.Session() as sess:
            print('----Testing----')
            #input num of data
            
            print("Reading checkpoints...")
            ckpt = tf.train.get_checkpoint_state(train_log_dir)
            print(ckpt)
            if ckpt and ckpt.model_checkpoint_path:
                global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                #saver.restore(sess, ckpt.model_checkpoint_path)
                print(global_step)
                print(ckpt.model_checkpoint_path)
                saver2.restore(sess,ckpt.model_checkpoint_path)
                print('Loading success, global_step is %s' % global_step)
            else:
                print('No checkpoint file found')
                return
            sess.run(tf.local_variables_initializer())
            
            num_of_test = MAX_STEP
            get_acc = -1
            try:
                coord = tf.train.Coordinator()
                threads = tf.train.start_queue_runners(sess = sess, coord = coord)
                for step in tqdm(range(MAX_STEP)):
                    if coord.should_stop():
                            break
                    #print("step %d"%step)
                    val_images, val_labels = sess.run([val_image_batch, val_label_batch])

                    loss,accuracy_,acc_operator,predict_label_,y_label_ = sess.run([hinge_loss,acc,acc_op,predict_label,y_label],
                                              feed_dict = {get_label_string_tensor:label_string_vector,tmparray_tensor:tmparray,id_array_tensor:id_array,x:val_images, y_:val_labels})
                    
                    
                    #print('%d / %d steps'%(step,MAX_STEP),'loss = ',loss,'    acc = ',accuracy_,'\n\n')
                    #print ('predict_label',predict_label_,' ----> I predict ',keys[int(predict_label_)])
                    #print ('y_label      ',y_label_,      ' ----> true ans',keys[int(y_label_)],'\n\n-------\n\n')
                    get_acc = accuracy_
                
            except tf.errors.OutOfRangeError:
                print('Done test -- epoch limit reached')
            finally:
                print('test acc = ',get_acc)
                coord.request_stop()
                coord.join(threads)
            

#%%


# In[12]:


t0 = time.time()
print(t0)
train()
t1 = time.time()


# In[ ]:


print('time = ',t1-t0,' s')


# In[9]:



            


# In[10]:


t0 = time.time()
#evaluate()
t1 = time.time()


# In[48]:


print(t1-t0)

