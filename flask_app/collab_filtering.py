#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 24 15:37:21 2019

@author: vcharvet
"""
import numpy as np
import pandas as pd



def df_to_ratings(df):
    df_pivot = pd.pivot_table(data=df,
                              index='user_history',
                              columns='item_history',
                              values='rating_history')
    ratings = df_pivot.fillna(0).values
    
    return ratings



def similarity(ratings):
    """
    
    """
    
    
    
    # vecteur contenant pour chaque utilisateur le nombre de notes données
    r_user = (ratings>0).sum(axis=1)  
    
    # vecteur contenant pour chaque utilisateur la moyenne des notes données
    m_user = np.divide(ratings.sum(axis=1) , r_user, where=r_user!=0)
    
    # Notes recentrées par la moyenne par utilisateur : chaque ligne i contient le vecteur \bar r_i
    ratings_ctr = ratings.T - ((ratings.T!=0) * m_user)
    ratings_ctr = ratings_ctr.T

    # Matrice de Gram, contenant les produits scalaires
    sim = ratings_ctr.dot(ratings_ctr.T)
    
    # Renormalisation
    norms = np.array([np.sqrt(np.diagonal(sim))])
    sim = sim / norms / norms.T  
    # (En numpy, diviser une matrice par un vecteur ligne (resp. colonne) 
    # revient à diviser chaque ligne (resp. colonne) terme à terme par les éléments du vecteur)
    
    return sim



def phi(x):
    return np.maximum(x, 0)


def pred_ratings(ratings):
    sim = similarity(ratings)
    numerator = phi(sim).dot(ratings)
    denominator = phi(sim).dot(ratings>0)
    pred_ratings = np.divide(numerator,denominator,where = denominator!=0)
    
    return pred_ratings



def predict_ratings_bias_sub(ratings,phi=(lambda x:x)):
    sim = similarity(ratings)
    
    r_user = (ratings>0).sum(axis=1)
    m_user = np.divide(ratings.sum(axis=1) , r_user, where=(r_user!=0))
    ratings_moyens = np.dot(m_user.reshape(len(m_user),1), np.ones((1,ratings.shape[1])))
   
    wsum_sim = np.abs(phi(sim)).dot(ratings>0)
    pred = ratings_moyens + np.divide(phi(sim).dot(ratings-(ratings>0)*ratings_moyens),wsum_sim, where= wsum_sim!=0)
    
    return np.minimum(5,np.maximum(1,pred))
    
    
    
    
    
    
    
    
    
    
    
    