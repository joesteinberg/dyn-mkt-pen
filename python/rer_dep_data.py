import numpy as np
import pandas as pd

import locale
locale.setlocale(locale.LC_ALL,'en_US.utf8')

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator
import matplotlib.dates as mdates

mpl.rc('text', usetex=True)
mpl.rc('savefig',bbox='tight')
mpl.rc('savefig',format='pdf')
mpl.rc('font',**{'family':'serif','serif':['Palatino'],'size':8})
mpl.rc('font',size=8)
mpl.rc('lines',linewidth=1)

alpha=0.8
colors=['#377eb8','#e41a1c','#4daf4a','#984ea3']



def reset_multiindex(df,n,suff):
    levels=df.columns.levels
    labels=df.columns.labels
    df.columns=levels[0][labels[0][0:n]].tolist()+[s+suff for s in levels[1][labels[1][n:]].tolist()]
    return df

def pct_chg(x):
        return 100*(x/x.iloc[0]-1.0)

agg_fns={'f':[('nf',lambda x:x.nunique())],
         'v':[('exports',np.sum)]}
    
###########################################################################

    
df = pd.read_pickle('output/bra_microdata_processed.pik')
y0=1998
y1=2003
df=df[df.y>=y0]
df=df[df.y<=y1]

rer = pd.read_csv('../../data/RBBRBIS.csv')
df=pd.merge(left=df,right=rer,how='left',on='y')

##### aggregate
agg = reset_multiindex(df.groupby(['y','rer']).agg(agg_fns).reset_index(),2,'')
agg['nf_pct_chg'] = agg['nf'].transform(pct_chg)
agg['exports_pct_chg'] = agg['exports'].transform(pct_chg)
agg['ex_log_chg'] = agg['exports'].transform(lambda x: np.log(x/x.iloc[0]))
agg['rer_log_chg'] = agg['rer'].transform(lambda x: np.log(x/x.iloc[0]))
agg['te']=agg.ex_log_chg/agg.rer_log_chg
agg['te'][0]=0

fig=plt.figure(figsize=(4,3))
ln1=agg.rer.plot(color=colors[0],label='Real effective exchange rate, left',marker='o',alpha=0.75)
ln2=agg.exports_pct_chg.plot(secondary_y=True,color=colors[1],label='Exports (\% chg.), right',marker='s',alpha=0.75)
ln3=agg.nf_pct_chg.plot(secondary_y=True,color=colors[2],label='Num. firms. (\% chg.), right',marker='x',alpha=0.75)

handles,labels = [],[]
for ax in fig.axes:
    for h,l in zip(*ax.get_legend_handles_labels()):
        handles.append(h)
        labels.append(l)

plt.legend(handles,labels,loc='best',prop={'size':6})

plt.savefig("output/bra_rer_dep1.pdf",bbox='tight')
plt.close('all')


##### by destination
x = reset_multiindex(df.groupby(['d','y','rer']).agg(agg_fns).reset_index(),3,'')

x['nf_pct_chg'] = x.groupby('d')['nf'].transform(pct_chg)
x['exports_pct_chg'] = x.groupby('d')['exports'].transform(pct_chg)
x['ex_log_chg'] = x['exports'].transform(lambda x: np.log(x/x.iloc[0]))
x['rer_log_chg'] = x['rer'].transform(lambda x: np.log(x/x.iloc[0]))
x['te']=x.ex_log_chg/x.rer_log_chg
    
x['min_y'] = x.groupby('d')['y'].transform(lambda xx: xx.iloc[0])
x['nf0']=x.groupby('d')['nf'].transform(lambda xx: xx.iloc[0])
x=x[x.min_y==y0]

p90 = x['nf0'].quantile(0.9)
p50 = x['nf0'].quantile(0.5)
x['grp'] = np.nan
x.loc[x.nf0<=p50,'grp']=0
x.loc[x.nf0>p50,'grp']=1

y = x.groupby(['grp','y'])['nf_pct_chg','exports_pct_chg','te'].mean().reset_index()

##### by destination, no new firms
df0=df
for yy in range(1998,2004):
    f0 = df[df.y==yy][['d','f']].drop_duplicates().reset_index(drop=True)
    df0 = pd.merge(left=f0,right=df0,how='inner',on=['d','f'])
                                            
x0 = reset_multiindex(df0.groupby(['d','y','rer']).agg(agg_fns).reset_index(),3,'')

x0['nf_pct_chg'] = x0.groupby('d')['nf'].transform(pct_chg)
x0['exports_pct_chg'] = x0.groupby('d')['exports'].transform(pct_chg)
x0['ex_log_chg'] = x0['exports'].transform(lambda x: np.log(x/x.iloc[0]))
x0['rer_log_chg'] = x0['rer'].transform(lambda x: np.log(x/x.iloc[0]))
x0['te']=x0.ex_log_chg/x0.rer_log_chg


x0['min_y'] = x0.groupby('d')['y'].transform(lambda xx: xx.iloc[0])
x0['nf0']=x0.groupby('d')['nf'].transform(lambda xx: xx.iloc[0])
x0=x0[x0.min_y==y0]


p90 = x0['nf0'].quantile(0.9)
p50 = x0['nf0'].quantile(0.5)
x0['grp'] = np.nan
x0.loc[x0.nf0<=p50,'grp']=0
x0.loc[x0.nf0>p50,'grp']=1

y0 = x0.groupby(['grp','y'])['nf_pct_chg','exports_pct_chg','te'].mean().reset_index()


#####
fig,axes=plt.subplots(1,3,figsize=(7,3),sharex=True,sharey=False)

axes[0].plot(y[y.grp==0].y,y[y.grp==0].exports_pct_chg,color=colors[0],label='Hard',alpha=0.75,marker='o')
axes[1].plot(y[y.grp==0].y,y[y.grp==0].nf_pct_chg,color=colors[0],label='Hard',alpha=0.75,marker='o')
axes[2].plot(y0[y0.grp==0].y,y0[y0.grp==0].exports_pct_chg,color=colors[0],label='Hard',alpha=0.75,marker='o')

axes[0].plot(y[y.grp==1].y,y[y.grp==1].exports_pct_chg,color=colors[1],label='Easy',alpha=0.75,marker='s')
axes[1].plot(y[y.grp==1].y,y[y.grp==1].nf_pct_chg,color=colors[1],label='Easy',alpha=0.75,marker='s')
axes[2].plot(y0[y0.grp==1].y,y0[y0.grp==1].exports_pct_chg,color=colors[1],label='Easy',alpha=0.75,marker='s')

axes[0].set_title(r'(a) Exports (\% chg.)')
axes[1].set_title(r'(b) Num. firms (\% chg.)')
axes[2].set_title(r"(c) Incumbents' exports (\% chg.)")
axes[0].set_xlim(1998,2003)
axes[0].set_xticks([1998,1999,2000,2001,2002,2003])
axes[0].legend(loc='best')
fig.subplots_adjust(hspace=0.15,wspace=0.225)
plt.savefig("output/bra_rer_dep2.pdf",bbox='tight')
plt.close('all')

