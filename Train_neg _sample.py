# -*- coding: utf-8 -*-
"""
Created on Tue Jun  7 15:16:48 2022

@author: Charbel
"""
import string
import random as rd
import numpy as np
import matplotlib.pyplot as plt 
import torch 
import torch.nn.functional as F
from Neg_class import Neg_samp
import torch.nn as nn
import string

from torch.utils.data import Dataset, DataLoader
text=' drinking a lot of water reduce the risk of heart attack  drinking many glass of  alcohol improve the risk of heart attack drinking too much juice reduce the risk of heart attack  drinking tons of whisky improve the risk of heart attack  drinking high amount of  tea reduce the risk of heart attack  drinking bunch of wine improve the risk of heart attack  '
text=text.translate(str.maketrans('', '', string.punctuation))
raw_text=text.lower()
raw_text=raw_text.split()

### fixer le shyperparametres

vocab = set(raw_text)   ## we create a vocab  by removing duplicates 

vocab_size = len(vocab) ## taille du vocabulaire
window_size=2  ## taille de la fentere de mots (c)
k_num=10 ### le nombre d'echantillons negatifs (negative samples)



#### creer une liste de paire de mots contentant pour exemple positif k exemples negatifs
word_to_ix = {word:ix for ix, word in enumerate(vocab)}
ix_to_word = {ix:word for ix, word in enumerate(vocab)}

data = []
for i in range(window_size,len(raw_text) - window_size):
    context=raw_text[i]
    target=[]
    for j in range(i-window_size,i+window_size+1):
        if(j!=i):
            data.append((context,raw_text[j],1))
            for k in range(k_num):
                data.append((context,ix_to_word[rd.randrange(1,len(vocab),window_size)],0))
            
 
one_hot=F.one_hot(torch.arange(0,vocab_size))
    
word_hot={word:hot for word,hot in zip(vocab,one_hot)} 
 


## construire un objet de type dataset pour avoir un jeu de donnes compatible avec package torch
class neg_data(Dataset):
    def __init__(self,data,word_to_ix):
        self.x=torch.zeros((len(data),2))
        self.y=torch.zeros(len(data))
        self.n_samples=len(data)
        k=0
        for context,target,label in data:
            self.x[k,0]=word_to_ix[context]
            self.x[k,1]=word_to_ix[target]
            self.y[k]=torch.tensor(label)
            k+=1
    def __getitem__(self,index):
        return self.x[index],self.y[index]
    def __len__(self):
        return self.n_samples   

### generer des representations one hot 
def Gen_one_hot(pos,vocab):
    if(type(pos)!=int):
        x=torch.zeros(len(pos),len(vocab))
        s=0
        for k in pos:
            x[s,int(k)]=1
            s=s+1
        return x
    else:
        x=torch.zeros(len(vocab))
        x[pos]=1
        return x
    



####
batch_size=round(len(data)*1)  ## le nombre de donnes par lot
neg_dataset=neg_data(data, word_to_ix)          
         

data_load=DataLoader(dataset=neg_dataset, batch_size=batch_size,shuffle=True)


## define the input hidden size
input_size=vocab_size ## fier la taille V 
hidden_size=2  ### dimesion de la couche cachee



## the model

neg_sampling=Neg_samp(input_size, hidden_size)

## loss function fonction de cout 
func_loss=nn.BCELoss() ### fonction de vraisemblance pour le modele 


## optimizer : algorithme d'optimisation 
optimizer=torch.optim.Adam(neg_sampling.parameters(),lr=0.01 )


num_epochs=500 ## nombre de passe sur tous les mots 

## Training
if(True):
 
    
    for epoch in range(num_epochs):
        total_loss = 0
    
        for inputs,label in data_load: 
            outputs=neg_sampling.forward(Gen_one_hot(inputs[:,0],vocab),Gen_one_hot(inputs[:,1],vocab)) ### calculer la sortie

            outputs.requires_grad_().long() ## autoriser torch a claculer les gradients
            loss=func_loss(outputs,label.float()) ## cacluler l'erreur 
            
            total_loss+=loss ## sommer les erreurs
            
            optimizer.zero_grad()
            
            loss.backward() #### calculer les gradients de la fonction cout 
            optimizer.step() ## mettre a jour les poids 
        
     
        print(total_loss) ## afficher l'erreur




    




