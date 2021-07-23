
import pandas as pd
import astropy.units as u
import thejoker as tj
from astropy.time import Time
import warnings
warnings.filterwarnings('ignore')
import numpy as np

def format_columns(column):
    format_column = column[(column!='') & (column!=None)]
    return format_column

if __name__ == '__main__':
    system='737m0797111.vzero.txt'
    df = pd.read_csv('Ruprecht/'+ system, header=None)
    df = df[0].str.split('  ',expand=True)
    df_2 = pd.DataFrame()
    df_1 = df.transpose()
    for column in df_1.columns:
        df_2[column]=(format_columns(df_1[column]).dropna()).reset_index(drop=True)
    df_2 = df_2.transpose().astype('float')
    df_2.columns = ['BJD','RV','ERR']
    t=Time(df_2['BJD'], format='mjd')
    rv=list(df_2['RV'])* u.m/u.s
    err=list(df_2['ERR'])* u.m/u.s
    #params = JokerParams(P_min=8*u.day, P_max=512*u.day)
    data = tj.RVData(t=t, rv=rv, rv_err=err)
    #data = RVData(t=t, rv=rv, stddev=err)
    prior = tj.JokerPrior.default(P_min=2*u.day, P_max=256*u.day,
                              sigma_K0=30*u.km/u.s,
                              sigma_v=100*u.km/u.s)
    joker = tj.TheJoker(prior)
    prior_samples = prior.sample(size=100)
    samples = joker.rejection_sample(data, prior_samples) 
    print(samples.pack()[0][0])
