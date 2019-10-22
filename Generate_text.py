# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 10:54:50 2019

@author: Deepika
"""


from random import randint
from pickle import load
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences


### load doc into memory 

def load_doc(filename):
    file=open(filename,'r')
    text=file.read()
    file.close()
    return text

##generate sequence from language model
    
def generate_seq(model,tokenizer,seq_length,seed_text,n_words):
    result=list()
    in_text=seed_text
    ##generate a fixed no of words
    for _ in range(n_words):
        ##incode the text as integer
        encoded=tokenizer.texts_to_sequences([in_text])[0]
        ##truncate sequences to a fixed length
        encoded=pad_sequences([encoded],maxlen=seq_length,truncating='pre')
        ##Predict probabilities index to word
        yhat=model.predict_classes(encoded,verbose=2)
        ##map predicted word index to word
        out_word=''
        for word,index in tokenizer.word_index.items():
            if index==yhat:
                out_word=word
                break
            ##append to input
        in_text += ' ' + out_word
        result.append(out_word)
    return ' '.join(result)


##load cleaned text sequences
    
in_filename='republic_sequences.txt' 
doc=load_doc(in_filename)
lines=doc.split('\n')
seq_length=len(lines[0].split())-1
##load model
model=load_model('model.h5')
##load Tikenizer
tokenizer=load(open('tokenizer.pkl','rb'))
##select a seed text
seed_text=lines[randint(0,len(lines))]
print(seed_text+'\n')
##generate new text
generated=generate_seq(model,tokenizer,seq_length,seed_text,50)
print(generated)
   


        