# -*- coding: utf-8 -*-
"""
Created on Thu Jun  2 14:04:24 2022

@author: Charbel
"""

# librairie pytorch 
import torch
import torch.nn as nn


# classe qui permet de creer un objet de type reseau de neurone pour les deux modeles skipgram et cbow
# la methode forward permet de creer la ''forward propagation''

class Word2vec(nn.Module):
      
    def __init__(self,input_size,hidden_size):
        """
        Constructeur de classe heritee de la classe nn.Module   

        Parameters
        ----------
        input_size : V taille du vocabulaire 
        
        hidden_size :
        Dimension de la representation vectorielle cherchee 

        Returns
        -------
        None.

        """
        super(Word2vec, self).__init__()
        ## fonction 
        self.fct1=nn.Linear(input_size,hidden_size) ## partie lineaire pour le passage de l'entree a la couche cachee
        self.fct2=nn.Linear(hidden_size,input_size) ## partie lineaire pour le passage de la couche cachee a la sortie 
        self.soft=nn.LogSoftmax(dim=-1) ### non-linearite pour transformer la sortie en  probabilite
        
    def forward(self,x):
        """
        

        Parameters
        ----------
        x : TYPE: vecteur de taille V 
        entree du modele.

        Returns
        -------
        out :
            la sortie sous forme d'une log probabilite.

        """
        out=self.fct1(x)
        out=self.fct2(out)
        out=self.soft(out)
        return out
    def forward_embed(self,x):
        out=self.fct1(x)
        return out
        





