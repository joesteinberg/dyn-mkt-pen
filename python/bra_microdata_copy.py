import numpy as np
import pandas as pd

##############################################################################################3
# read the microdata

# ignore great recession
years = range(1996,2008)

df = None
for y in years:
    
    print('\t%d'%y)
    
    s = str(y)[2:]
    if y==2000:
        s=s+'_original'
        
    tmp = pd.read_stata('/home/joseph/Research/datasets/secex_brazil_customs_data/Secex_Stata/Exports/exp'+s+'.dta')
    if 'cnpj' in tmp.columns.tolist():
        tmp.loc[tmp.cnpj8=='','cnpj8'] = tmp.cnpj[tmp.cnpj8==''].str[0:8]
    
    tmp = tmp[['ano','mes','cnpj8','ncm','pais','valor']]
    tmp.reset_index(drop=True,inplace=True)
    if df is None:
        df=tmp
    else:
        df=df.append(tmp)

df.to_csv('/home/joseph/Research/datasets/secex_brazil_customs_data/secex.csv',sep=',')

