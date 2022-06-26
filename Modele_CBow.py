# -*- coding: utf-8 -*-
"""
Created on Mon May 23 15:41:50 2022

@author: Charbel
"""
import numpy as np
import torch 
import torch.nn.functional as F
from Word2vec_class import Word2vec
import torch.nn as nn
#import torchvision 
import string
from torch.utils.data import Dataset, DataLoader
from sklearn.linear_model import LogisticRegression


## construire la classe de type dataset pour obtenir un jeu de donnes compatible avec le package pytorch
class bow_data(Dataset):
    def __init__(self,data,word_hot,word_to_ix):
        self.x=torch.zeros((len(data),len(word_hot)))
        self.y=torch.zeros(len(data))
        self.n_samples=len(data)
        k=0
        for context,target in data:
            self.x[k,:]=Generate_input(context, word_hot)
            self.y[k]=word_to_ix[target]
            k+=1
    def __getitem__(self,index):
        return self.x[index],self.y[index]
    def __len__(self):
        return self.n_samples



## texte servant comme ensemble d'apprentissage
text=' drinking a lot of water reduce the risk of heart attack  drinking many glass of  alcohol improve the risk of heart attack drinking many cans of juice reduce the risk of heart attack  drinking tons of whisky improve the risk of heart attack  drinking high amount of  tea reduce the risk of heart attack  drinking bunch of wine improve the risk of heart attack  '
text=text.translate(str.maketrans('', '', string.punctuation))
raw_text=text.lower()
raw_text=raw_text.split()

#fixer les parametres
vocab = set(raw_text)   ## we create a vocab  by removing duplicates 
vocab_size = len(vocab)
window_size=2
one_hot=F.one_hot(torch.arange(0,vocab_size))

word_hot={word:hot for word,hot in zip(vocab,one_hot)}
word_hot_inverse={hot:word for hot,word in zip(one_hot,vocab)}
word_to_ix = {word:ix for ix, word in enumerate(vocab)}
ix_to_word = {ix:word for ix, word in enumerate(vocab)}

## créer des données d'entraînement avec une liste de mots de contexte et de mots cibles

data = []
for i in range(window_size,len(raw_text) - window_size):
    target=raw_text[i]
    context=[]
    for j in range(i-window_size,i+window_size+1):
        if(j!=i):
            context.append(raw_text[j])
    data.append((context,target))
        
            




## Generer l'entree du modele 
def Generate_input(context,word_hot):
    """
    

    Parameters
    ----------
    context : list of string 
        .
     word_hot : dictionnary of string as keys and one hot tensor as values
        contain our one hot representation of each word.

    Returns
    -------
    TYPE Tensor 
        DESCRIPTION.
        the averaged tensor 

    """
    idxs = [word_hot[w] for w in context]
    return (sum(idxs)*(1/len(idxs)))

# fixer la taille du lot de donnes 
batch_size=len(data)

# construire un objet de type dataset de pytorch 
Cbow_dataset=bow_data(data, word_hot, word_to_ix)          
         
train_data,test_data=torch.utils.data.random_split(Cbow_dataset,[95*len(data)//100,len(data)-95*len(data)//100])
data_load=DataLoader(dataset=Cbow_dataset, batch_size=batch_size,shuffle=True)


## definir la taille du vocavulaire
input_size=vocab_size
# la dimension de la representation vectorielle
hidden_size=2

## Construire un objet de type Word2vec pour generer le modele 
Cbow=Word2vec(input_size, hidden_size)    

 
##definir la fonction de cout automatiquementen utilisant le package pytorch
func_loss= nn.NLLLoss()


## definir l'agorithme d'optimisation en utilisant le package torch.optim 
optimizer=torch.optim.Adam(Cbow.parameters(),lr=0.01 )


## Entrainement de reseau 
if(1):
    
    for epoch in range(5000): # le nombre de passs 
        
        total_loss = 0
    
        for context, target in data_load: # pour chaque lot de contexte et mot cible dans l'ensemble de donnes
            outputs=Cbow.forward(context) # calculer la sortie en utilisant la methode forward 
           
            outputs.requires_grad_().long() # autoriser le module pytorch a calculer autoomatiquement le sgradients
            loss=func_loss(outputs,target.long())  # calculer l'erreur a chaque sortie
            
            total_loss+=loss # sommer l'erreur sur tous les lots
            optimizer.zero_grad() 
            loss.backward() # trouver les poids optimaux automatiquement 
            optimizer.step() # mettre a jour les parametres du modele 
      
        print(total_loss) # afficher l'erreur totale a chaque passse
    
    
    

## obtenir les parametres du modeles pour la representation vectorielle
def param_embedding(model):
    k=0
    for params in model.parameters():
        if(k==0):
            break 
    return params 
        
## obtenir la nouvelle representation   
def vec(one_hot,model):
    E=param_embedding(model)
    return torch.matmul(E,one_hot)















##########################################
#####  Visualiser un exemple 
#################################


import matplotlib.pyplot as plt 
plt.close()



sub_vocab=['alcohol','wine','whisky','water','tea','juice']
#sub_vocab=[w for w in vocab]
def stack_one_hot(sub_vocab):
    one=torch.zeros(len(vocab),len(sub_vocab))
    for k in range(len(sub_vocab)):
        one[:,k]=word_hot[sub_vocab[k]]
    return one
X=vec(stack_one_hot(sub_vocab).float(),Cbow).detach().numpy()
y=np.array([1,1,1,0,0,0])      
fig,ax=plt.subplots()
ax.scatter(vec(stack_one_hot(sub_vocab).float(),Cbow).detach().numpy()[0],vec(stack_one_hot(sub_vocab).float(),Cbow).detach().numpy()[1])

for i,txt in enumerate(sub_vocab):
    ax.annotate(txt,(vec(word_hot[txt].float(),Cbow).detach().numpy()[0],vec(word_hot[txt].float(),Cbow).detach().numpy()[1]))

plt.show()
  
clf = LogisticRegression()
clf.fit(X.T,y)
# Retrieve the model parameters.
b = clf.intercept_[0]
w1, w2 = clf.coef_.T
# Calculate the intercept and gradient of the decision boundary.
c = -b/w2
m = -w1/w2
xmin, xmax = min(X[0])-1, max(X[0])+1
ymin, ymax =min(X[1])-1, max(X[1])+1
xd = np.array([xmin, xmax])
yd = m*xd + c
ax.plot(xd, yd, 'k', lw=1, ls='--')
ax.fill_between(xd, yd, ymin, color='tab:blue', alpha=0.2)
ax.fill_between(xd, yd, ymax, color='tab:orange', alpha=0.2)
ax.legend(['decision','donnees','non-alcohol','alcohol'])
plt.xlim(xmin, xmax)
plt.ylim(ymin, ymax)

fig1,ax1=plt.subplots()
ax1.scatter(vec(stack_one_hot(sub_vocab).float(),Cbow).detach().numpy()[0],vec(stack_one_hot(sub_vocab).float(),Cbow).detach().numpy()[1])

for i,txt in enumerate(sub_vocab):
    ax1.annotate(txt,(vec(word_hot[txt].float(),Cbow).detach().numpy()[0],vec(word_hot[txt].float(),Cbow).detach().numpy()[1]))



