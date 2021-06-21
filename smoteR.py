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


def relevance_rule(x):
    """
    Domain based relevance function of target values. 
    relevance map target values into the importance of values ranging from [0,1]
    1 means most relevant cases and should be used to synthesize new samples.  
    """
    x = np.array(x)
    x_relevance = (x != 0)*1
    return x_relevance
    
def relevance_sigmoid(x):
    """
    Apply sigmoid function as one relevance function. 
    """
    x_relevance = 1/(1 + np.exp(-x))
    return x_relevance

def create_synth_samples(df, target, over_rate=2, k=3, categorical_cols = [], random_state=42):
    ''' generate new samples based on input df by SMOTER
    
    Parameters
    ----------
        df: dataframe 
            contains the initial samples
        target: string 
            name of the target column 
        over_rate: intger 
            decides the number of synthesized sample(s) per sample
        k: intger
            the number of nearest neighbors for each sample
        categorical_cols: list 
            contains all categorical feature names
    Return:
        df_new: pd.DataFrame containing synthesized samples
    '''
    np.random.seed(random_state)
    df_new = pd.DataFrame(columns = df.columns) # initialize empty dataframe 
    
    knn = KNeighborsRegressor(n_neighbors = k+1, n_jobs = -1) # k+1 because one is the nearest neighbor to itself
    knn.fit(df.drop(columns = [target]).values, df[[target]])
    
    for index, case in df.iterrows(): # iterate through each row and extract the index & feature values of each sample
        neighbors = knn.kneighbors(case.drop(labels = [target]).values.reshape(1, -1), n_neighbors=k+1, return_distance=False).reshape(-1)
        neighbors = np.delete(neighbors, np.where(neighbors == index))
        for i in range(0, int(over_rate)):
            x = df.iloc[neighbors[np.random.randint(k)]]# randomly choose one of the neighbors
            attr = {}
            all_columns = df.columns.tolist()
            numeric_cols = [feat for feat in all_columns if feat not in categorical_cols and feat != target] 
            for feat in all_columns:
                if feat in categorical_cols:# if categorical then choose randomly one of values
                    if np.random.randint(2) == 0:
                        attr[feat] = case[feat]
                    else:
                        attr[feat] = x[feat]
                if feat in numeric_cols: # if continious column, compute based on SMOTER
                    diff = case[feat] - x[feat]
                    attr[feat] = case[feat] + np.random.randint(2) * diff
                else:
                    continue
            # compute target value by weighted average of d1 and d2
            new = np.array(list(attr.values()))
            d1 = euclidean_distances(new.reshape(1, -1), case.drop(labels = [target]).values.reshape(1, -1))[0][0]
            d2 = euclidean_distances(new.reshape(1, -1), x.drop(labels = [target]).values.reshape(1, -1))[0][0]
            if (d1 + d2 == 0):
                attr[target] = (case[target] + x[target])/2  #fully replicated feature values of case in x
            else:
                attr[target] = (d2 * case[target] + d1 * x[target])/(d1 + d2) #original weighted average to compute target
            df_new = df_new.append(attr, ignore_index = True)
    return df_new

def smoteR(D, target, th=0, over_rate = 2, under_rate = 0.5, k = 3, categorical_cols = [], relevance=relevance_sigmoid):
    '''
    Construct SmoteR algorithm: https://core.ac.uk/download/pdf/29202178.pdf
    Parameters
    ----------
        D: dataframe 
            contains the initial samples
        target: string 
            name of the target column 
        th: threshold used to define two groups of minorities
        over_rate: intger 
            decides the number of synthesized sample(s) per sample
        under_rate: float 
            decides the number of total majority samples by downsampling
        k: intger
            the number of nearest neighbors for each sample
        categorical_cols: list 
            contains all categorical feature names
    Return:
        new_data: dataframe
            new dataset contains reduced majority and sythesized minorities 
    '''
    
    if len(categorical_cols) == 0:
      for feat in D.columns:
        if D[feat].nunique()<=31:
          categorical_cols.append(feat) #for features with few unique values
          
    y_median = D[target].median() # median of the target variable
    
    rareL = D[(relevance(D[target]) > th) & (D[target] < y_median)]# rare cases where target less than median 
    if len(rareL) == 0:
      new_casesL = pd.DataFrame(columns = D.columns)
      print('no samples in rare & low range')
    else:
      new_casesL = create_synth_samples(rareL, target, over_rate, k, categorical_cols)
      print(f"Create {len(new_casesL)} minorities that have lower traget values.")
    
    rareH = D[(relevance(D[target]) > th) & (D[target] > y_median)]# rare cases where target greater than median
    if len(rareH) == 0:
      new_casesH = pd.DataFrame(columns = D.columns)
      print('no samples in rare & high rrange')
    else:
      new_casesH = create_synth_samples(rareH, target, over_rate, k, categorical_cols)
      print(f"Create {len(new_casesH)} minorities that have higher traget values.")
    
    new_cases = pd.concat([new_casesL, new_casesH], axis=0)# combine two types of minorities 
   
  
    dfMaj = D[relevance(D[target]) <= th]# cases in the majority
    MajDown_num = int(len(dfMaj)*under_rate)
    if MajDown_num > 100:
      dfMajDown = resample(dfMaj, replace=False, n_samples = MajDown_num, random_state = 42)
      print(f"Downsample {len(dfMaj)} majority into {len(dfMajDown)} samples.")
    else:
      dfMajDown = dfMaj
      print('too few samples in majority class, increase threshold.')
    
    df_SmoteR = pd.concat([new_cases, dfMajDown, rareL, rareH], axis=0).reset_index(drop=True)
    print(f'SMOTER generates {len(df_SmoteR)} samples')
    
    return df_SmoteR

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
