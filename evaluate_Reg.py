#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Buid smoteR to correct imbalance for regression tasks """ 

__author__ = "Bolin Li"
__date__ = "02 Sep 2020"
__revised__ = "16 Sep 2020"

from sklearn.neighbors import KNeighborsRegressor
from tqdm import tqdm
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics.pairwise import euclidean_distances, cosine_similarity
from sklearn.utils import resample
from sklearn.metrics import mean_squared_error
from scipy.integrate import simps
import numpy as np
import pandas as pd
import matplotlib as plt

def reg_evaluatematrix(actual, pred, model_name='model'):
  actual_pred = pd.DataFrame(actual.reset_index(drop=True),columns=['target']).join(pd.DataFrame(pred, columns=['pred']))
  actual_pred_min = actual_pred[actual_pred.target!=0.0]
  actual_pred_maj = actual_pred[actual_pred.target==0.0]
  
  RMSE_all = np.sqrt(mean_squared_error(actual_pred['target'], actual_pred['pred']))
  RMSE_min = np.sqrt(mean_squared_error(actual_pred_min['target'], actual_pred_min['pred']))
  RMSE_maj = np.sqrt(mean_squared_error(actual_pred_maj['target'], actual_pred_maj['pred']))    
                      
  try:
    dflorentz = buildLorentzDataFrame(actual, pred)
    gini_non0 = computeGini(dflorentz.optimalmodel, dflorentz.predictivemodel)
  except: 
    gini_non0 = 'error'
  scores = np.array([RMSE_all,RMSE_min, RMSE_maj, gini_non0])
  cols  = ['RMSE_all','RMSE_min', 'RMSE_maj', 'Gini']
  model_score_cols = [model_name+'_'+col for col in cols]
  model_scores = pd.DataFrame(scores.reshape(1,4), columns=model_score_cols)
  
  return model_scores
  
def buildLorentzDataFrame(ytrue, ypred, yscore=None, target=1):
    """Build the Lorentz metrics and store it in a pd dataframe.

    Used to later compute the Lorentz curve and the Gini coefficient.

    Parameters
    ----------
    ytrue : pandas series
        true labels
    
    ypred : array
        target predictions
    
    yscore : array
        target scores. If None, regression problem
    
    Returns
    -------
    Lorentz pandas dataframe
    """
    indx = ytrue.keys()

    if yscore is not None:
        yscorepd = yscore[:, target]    

        ypredpd = pd.Series(data=ypred, index=indx)
        yscorepd = pd.Series(data=yscorepd, index=indx)
        yscorepdsort = yscorepd.sort_values(ascending=False)
        ypredpd = ypredpd[yscorepdsort.keys()]

        d = {'pred': ypredpd}
        lorentzdf = pd.DataFrame(data=d, index=yscorepdsort.keys())
        lorentzdf['true'] = ytrue[lorentzdf.index]

        totaltrue = sum(lorentzdf.true)
        lorentzdf['optimalmodel'] = np.cumsum(
            np.sort(lorentzdf.true)[::-1]/totaltrue)
        lorentzdf['indicatorfunction'] = np.int64(
            (lorentzdf.pred == lorentzdf.true) & (
                1 == lorentzdf.true))
        lorentzdf['predictivemodel'] = np.cumsum(
            lorentzdf.indicatorfunction/totaltrue)
    else:
        ypredpd = pd.Series(data=ypred, index=indx)
        ypredpdsort = ypredpd.sort_values(ascending=False)

        d = {'pred': ypredpd}
        lorentzdf = pd.DataFrame(data=d, index=ypredpdsort.keys())
        lorentzdf['indicatorpred'] = np.int64(0 != lorentzdf.pred)
        lorentzdf['true'] = ytrue[lorentzdf.index]
        lorentzdf['indicatortrue'] = np.int64(0 != lorentzdf.true)
        
        totaltrue = sum(lorentzdf.indicatortrue)
        lorentzdf['optimalmodel'] = np.cumsum(
            np.sort(lorentzdf.indicatortrue)[::-1]/totaltrue)
        lorentzdf['indicatorfunction'] = np.int64(
            (1 == lorentzdf.indicatorpred ) & (
                1 == lorentzdf.indicatortrue))
        lorentzdf['predictivemodel'] = np.cumsum(
            lorentzdf.indicatorfunction/totaltrue)

    return lorentzdf


def computeGini(lorentzoptmdl, lorentzpredmdl):
    """Compute the gini score.

    Parameters
    ----------
    lorentzoptmdl : array
        true lorentz labels
    
    lorentzpredmdl : array
        target lorentz scores
    
    Returns
    -------
    gini : float
        gini coefficient such that A / (A + B), best 0.5
    """
    xi = np.arange(1, len(lorentzoptmdl)+1, step=1)
    yrandomguess = np.arange(
        0, max(lorentzoptmdl), step=max(lorentzoptmdl)/len(xi))
    aucrandomguess = simps(yrandomguess, xi, dx=1e-3)
    aucoptmodel = simps(lorentzoptmdl, xi, dx=1e-3)
    aucpredmodel = simps(lorentzpredmdl, xi, dx=1e-3)
    A = aucpredmodel - aucrandomguess
    B = aucoptmodel - aucrandomguess
    gini = A / (A+B)
    return gini


def plotLorentzCurve(lorentzoptmdl, lorentzpredmdl, ystr=None):
    """Compute the Lorentz curve graph.

    Parameters
    ----------
    lorentzoptmdl : array
        true lorentz labels
    
    lorentzpredmdl : array
        target lorentz scores
    """
    if ystr is None:
        ystr = "Number of Defaults\nas % of Total Defaults in first k loans"
    plt.figure(1, figsize=(12,8))
    xi = np.arange(1, len(lorentzoptmdl)+1, step=1)
    plt.plot([0, len(lorentzoptmdl)], [0, 1], 'k--', label='random guess')
    plt.plot(xi, lorentzoptmdl, label='optimal model')
    plt.plot(xi, lorentzpredmdl, label='predictive model')
    plt.xlabel('Number of Samples')
    plt.ylabel(ystr)
    plt.legend(loc='best')
    plt.show()
