# -*- coding: utf-8 -*-
"""
Created on Mon May 30 13:00:40 2022

@author: Charbel
"""
##########################################
######         Skip-gram           #######
##########################################
## import library
import matplotlib.pyplot as plt 
import numpy as np

import torch 
import torch.nn.functional as F
from Word2vec_class import Word2vec
import torch.nn as nn
import string
from sklearn.linear_model import LogisticRegression
from torch.utils.data import Dataset, DataLoader
### preparer les donnes 

text=' drinking a lot of water reduce the risk of heart attack  drinking many glass of  alcohol improve the risk of heart attack drinking many cans of juice reduce the risk of heart attack  drinking tons of whisky improve the risk of heart attack  drinking high amount of  tea reduce the risk of heart attack  drinking bunch of wine improve the risk of heart attack  '
text=text.translate(str.maketrans('', '', string.punctuation))
raw_text=text.lower()
raw_text=raw_text.split()

vocab = set(raw_text)   



# fixer les hyperparametres
vocab_size = len(vocab)
window_size=4 # taille de la fenetre 


word_to_ix = {word:ix for ix, word in enumerate(vocab)}
ix_to_word = {ix:word for ix, word in enumerate(vocab)}
## creer une liste de context et mot cible 
data = []
for i in range(window_size,len(raw_text) - window_size):
    context=raw_text[i]
    target=[]
    for j in range(i-window_size,i+window_size+1):
        if(j!=i):
            data.append((context,word_to_ix[raw_text[j]]))
        
        
    
    
one_hot=F.one_hot(torch.arange(0,vocab_size))
    
word_hot={word:hot for word,hot in zip(vocab,one_hot)} 
 

# construire un objet de type dataset compatible avec le package pytorch 
class bow_data(Dataset):
    def __init__(self,data,word_hot):
        self.x=torch.zeros((len(data),len(word_hot)))
        self.y=torch.zeros(len(data))
        self.n_samples=len(data)
        k=0
        for context,target in data:
            self.x[k,:]=word_hot[context]
            self.y[k]=torch.tensor(target)
            k+=1
    def __getitem__(self,index):
        return self.x[index],self.y[index]
    def __len__(self):
        return self.n_samples   




## obtenir les parametres du modele
def param_embedding(model):
    k=0
    for params in model.parameters():
        if(k==0):
            break 
    return params 
batch_size=round(len(data)*1)
Cbow_dataset=bow_data(data, word_hot)          
         

data_load=DataLoader(dataset=Cbow_dataset, batch_size=batch_size,shuffle=True)


## define the input hidden size
input_size=vocab_size ## taille de V de l'entree
hidden_size=2 # dimension de la couche cachee

## the model
skipgram=Word2vec(input_size, hidden_size)    


 
##loss fonction de cout 

func_loss= nn.NLLLoss()

##reductiom=sum()

## optimizer
optimizer=torch.optim.Adam(skipgram.parameters(),lr=0.01)#,betas=(0.9,0.999), weight_decay=0.1, amsgrad=True)

num_epochs=1000 # nombre de passe 

## Training
if(1):
    ax_loss=np.zeros(num_epochs)
    
    for epoch in range(num_epochs):
        total_loss = 0
    
        for context, target in data_load: 
            outputs=skipgram.forward(context) # calculer la sortie en utilisant la methode forward 
         

            outputs.requires_grad_().long() # autoriser le module pytorch a calculer autoomatiquement les gradients
            loss=func_loss(outputs,target.long()) # calculer l'erreur a chaque sortie
            
            total_loss+=loss # sommer l'erreur sur tous les lots
            
            optimizer.zero_grad()
            
            loss.backward() # trouver les poids optimaux automatiquement 
            optimizer.step() # mettre a jour les parametres du modele 
            print(total_loss) # afficher l'erreur totale a chaque passse
            
            
        ax_loss[epoch]=total_loss
        
     
        
        
   

               

    

        
