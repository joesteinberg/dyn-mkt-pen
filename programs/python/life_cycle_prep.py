import os
import numpy as np
import pandas as pd
import sys
from statsmodels.api import OLS
from statsmodels.formula.api import ols
import patsy
import locale
locale.setlocale(locale.LC_ALL,'en_US.utf8')

dopath = '/home/linux/Documents/Research/dyn_mkt_pen/v3/programs/python/'
outpath = '/home/linux/Documents/Research/dyn_mkt_pen/v3/programs/python/output/'

alt_models=True
alt_suffs=['smp','sunkcost','acr','abn1','abn2','abo1','abo2','a0']

max_tenure_scalar=5

#############################################################################

print('\tLoading the processed microdata...')

# load the preprocessed data
df = pd.read_pickle(outpath + 'pik/bra_microdata_processed.pik')

df['nf'] = df.groupby(['d','y'])['f'].transform(lambda x: x.nunique())
tmp1 = df.groupby('d')['nf'].mean().reset_index().rename(columns={'nf':'avg_nf'})
df = pd.merge(left=df,right=tmp1,on='d')
p50 = tmp1.avg_nf.quantile(0.5)
df['grp'] = np.nan
df.loc[df.avg_nf<p50,'grp']=0
df.loc[df.avg_nf>p90,'grp']=1
df=df[df.tenure.notnull()].reset_index(drop=True)
df.loc[df.tenure>max_tenure_scalar,'tenure']=max_tenure_scalar
df.loc[df.max_tenure>max_tenure_scalar,'max_tenure']=max_tenure_scalar
df.rename(columns={'exit':'xit'}).to_stata(outpath + 'stata/bra_microdata_processed.dta')
dfs = df[df.max_tenure>=max_tenure_scalar].reset_index(drop=True)

print('\tLoading the simulated data...')

df2 = pd.read_pickle(outpath + 'pik/model_microdata_processed.pik')
df2['nf'] = df2.groupby(['d','y'])['f'].transform(lambda x: x.nunique())
tmp2 = df2.groupby('d')['nf'].mean().reset_index().rename(columns={'nf':'avg_nf'})
df2 = pd.merge(left=df2,right=tmp2,on='d')
p50 = tmp2.avg_nf.quantile(0.5)
df2['grp'] = np.nan
df2.loc[df2.avg_nf<p50,'grp']=0
df2.loc[df2.avg_nf>p90,'grp']=1
df2=df2[df2.tenure.notnull()].reset_index(drop=True)
df2.loc[df2.tenure>max_tenure_scalar,'tenure']=max_tenure_scalar
df2.loc[df2.max_tenure>max_tenure_scalar,'max_tenure']=max_tenure_scalar
df2.rename(columns={'exit':'xit'}).to_stata(outpath + 'stata/model_microdata_processed.dta')
df2s = df2[df2.max_tenure>=max_tenure_scalar].reset_index(drop=True)

df2s_alt = []
if(alt_models==True):
    df2s_alt = [None for s in alt_suffs]
    for i in range(len(alt_suffs)):
        tmp = pd.read_pickle(outpath + 'pik/' + alt_suffs[i]+'_microdata_processed.pik')
        tmp['nf'] = tmp.groupby(['d','y'])['f'].transform(lambda x: x.nunique())
        tmp2 = tmp.groupby('d')['nf'].mean().reset_index().rename(columns={'nf':'avg_nf'})
        tmp = pd.merge(left=tmp,right=tmp2,on='d')
        p50 = tmp2.avg_nf.quantile(0.5)            
        tmp['grp'] = np.nan
        tmp.loc[tmp.avg_nf<p50,'grp']=0
        tmp.loc[tmp.avg_nf>p90,'grp']=1
        tmp=tmp[tmp.tenure.notnull()].reset_index(drop=True)
        tmp.loc[tmp.tenure>max_tenure_scalar,'tenure']=max_tenure_scalar
        tmp.loc[tmp.max_tenure>max_tenure_scalar,'max_tenure']=max_tenure_scalar
        tmp.rename(columns={'exit':'xit'}).to_stata(outpath + 'stata/' + alt_suffs[i]+'_microdata_processed.dta')
        df2s_alt[i] = tmp[tmp.max_tenure>=max_tenure_scalar].reset_index(drop=True)   

#############################################################################
print('\tEstimating regressions on actual data...')
os.system('stata -b ' + dopath + 'life_cycle_data.do')

print('\tEstimating regressions on simulated data...')
os.system('stata -b ' + dopath + 'life_cycle_model.do')

if(alt_models==True):
    print('\tEstimating regressions on simulated data from alternative models...')
    os.system('stata -b ' + dopath + 'life_cycle_alt_models.do')

