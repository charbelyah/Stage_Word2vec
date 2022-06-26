# -*- coding: utf-8 -*-
"""
Created on Tue Jun  7 15:09:47 2022

@author: Charbel
"""

import torch
import torch.nn as nn


## classe qui permet de construire un objet de type Modele Negative Sampling


class Neg_samp(nn.Module):
    
    def __init__(self,input_size,hidden_size):  ### heriter du Module nn le contructeur de l'objet reseau de neurones
        super(Neg_samp, self).__init__()
        ## fonction 
        self.fct1=nn.Linear(input_size,hidden_size) ### partie lineaire passsage de l'entree vers la couche cachee
        self.fct2=nn.Linear(input_size,hidden_size) ### partie lineaire passsage de l'entree vers la couche cachee
        self.sig=torch.sigmoid  ### non-linearite pour transformer la sortie en  probabilite
        
    def forward(self,x,y):
        """
        Simuler la forward propagation 
        

      

        """
        outx=self.fct1(x)
      
        outy=self.fct2(y)
       
        out=outx*outy
        out=self.sig(out.sum(axis=1))
     
        return out
    
    def forward_embed(self,x):
        out=self.fct1(x)
        return out
        x