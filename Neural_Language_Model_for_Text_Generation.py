# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 21:38:31 2019

@author: Deepika
"""

import re
import os
import string
from numpy import array
from pickle import dump
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from keras.utils.vis_utils import plot_model
from keras.models import Sequential
from keras.layers import LSTM,Dense,Embedding
 
os.chdir('D:\\Python\\Data_Science\\Deep_Learning\\NLP_Learning\\NLP_TASK\\Neural_Language_Model_Text_Generation_20\\')


# load doc into memory
def load_doc(filename):
    # open the file as read only
    file = open(filename, 'r')
    # read all text
    text = file.read()
    # close the file
    file.close()
    return text

# load document
in_filename = 'republic_clean.txt'
doc = load_doc(in_filename)
print(doc[:200])


# turn a doc into clean tokens
def clean_doc(doc):
    # replace '--' with a space ' '
    doc = doc.replace('--', ' ')
    # split into tokens by white space
    tokens = doc.split()
    # prepare regex for char filtering
    re_punc = re.compile('[%s]' % re.escape(string.punctuation))
    # remove punctuation from each word
    tokens = [re_punc.sub('', w) for w in tokens]
    # remove remaining tokens that are not alphabetic
    tokens = [word for word in tokens if word.isalpha()]
    # make lower case
    tokens = [word.lower() for word in tokens]
    return tokens

# clean document
tokens = clean_doc(doc)
print(tokens[:200])
print('Total Tokens: %d' % len(tokens))
print('Unique Tokens: %d' % len(set(tokens)))

### Save clean Text


# organize into sequences of tokens
length = 50 + 1
sequences = list()
for i in range(length, len(tokens)):
    # select sequence of tokens
    seq = tokens[i-length:i]
    # convert into a line
    line = ' '.join(seq)
    # store
    sequences.append(line)
print('Total Sequences: %d' % len(sequences))

# save tokens to file, one dialog per line
def save_doc(lines, filename):
    data = '\n'.join(lines)
    file = open(filename, 'w')
    file.write(data)
    file.close()

# save sequences to file
out_filename = 'republic_sequences.txt'
save_doc(sequences, out_filename)

#### Train Language Model

###define Model

def define_model(vocab_size,seq_length):
    model=Sequential()
    model.add(Embedding(vocab_size,50,input_length=seq_length))
    model.add(LSTM(100,return_sequences=True))
    model.add(LSTM(100))
    model.add(Dense(100,activation='relu'))
    model.add(Dense(vocab_size,activation='softmax'))
    ##Comple Nnetwork
    model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
    ###summarize defined model
    model.summary()
    plot_model(model,to_file='model.png',show_shapes=True)
    return model


##load 

in_filename='republic_sequences.txt' 
doc=load_doc(in_filename)
lines=doc.split('\n')
###integer encode sequence of words
tokenizer=Tokenizer()
tokenizer.fit_on_texts(lines)
sequences=tokenizer.texts_to_sequences(lines)
##vocabulary Size

vocab_size=len(tokenizer.word_index)+1
##seperate into input and output

sequences=array(sequences)
X,y=sequences[:,:-1],sequences[:,-1]
y=to_categorical(y,num_classes=vocab_size)
seq_length=X.shape[1]

###define model 

model=define_model(vocab_size,seq_length)
###fit model
model.fit(X,y,batch_size=128,epochs=100)
##save model to file
model.save('model.h5')
## save the tokenizer
dump(tokenizer,open('tokenizer.pkl','wb'))



  