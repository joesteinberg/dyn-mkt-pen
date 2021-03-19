import numpy as np
import pandas as pd
from statsmodels.formula.api import ols

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
mpl.rc('savefig',format='pdf')
mpl.rc('font',**{'family':'serif','serif':['Palatino'],'size':8})
mpl.rc('font',size=8)
mpl.rc('lines',linewidth=1)

alpha=0.8
colors=['#377eb8','#e41a1c','#4daf4a','#984ea3']


def reset_multiindex(df,n,suff):
    tmp = df.columns
    #levels=df.columns.levels
    #labels=df.columns.labels
    df.columns=[x[0] for x in tmp[0:n]] + [x[1]+suff for x in tmp[n:]]
    #df.columns=levels[0][labels[0][0:n]].tolist()+[s+suff for s in levels[1][labels[1][n:]].tolist()]
    return df

def pct_chg(x):
        return 100*(x/x.iloc[0]-1.0)

agg_fns={'f':[('nf',lambda x:x.nunique())],
         'v':[('exports',np.sum)]}
    
###########################################################################
########### Aggregate plot


##### load the secex data
df = pd.read_pickle('output/bra_microdata_processed.pik')
df=df[df.y>=1998]
#y0=1997
#y1=2005
#df=df[df.y>=y0]
#df=df[df.y<=y1]

##### merge on RER
rer = pd.read_csv('../../data/RBBRBIS.csv')
df=pd.merge(left=df,right=rer,how='left',on='y')

##### aggregate
agg = reset_multiindex(df.groupby(['y','rer']).agg(agg_fns).reset_index(),2,'')
agg['nf_pct_chg'] = agg['nf'].transform(pct_chg)
agg['exports_pct_chg'] = agg['exports'].transform(pct_chg)
agg['ex_log_chg'] = agg['exports'].transform(lambda x: np.log(x/x.iloc[0]))
agg['rer_log_chg'] = agg['rer'].transform(lambda x: np.log(x/x.iloc[0]))
agg['te']=agg.ex_log_chg/agg.rer_log_chg
agg.loc[0,'te']=0

y0 = agg.y.min()
y1 = agg.y.max()
agg.set_index('y',inplace=True)
fig=plt.figure(figsize=(3.5,3.5))
ln1=agg.rer.plot(color=colors[0],label=r'REER (left)',marker='o',alpha=0.75)
ln2=agg.exports_pct_chg.plot(secondary_y=True,color=colors[1],label=r'Exports (\% chg.), right',marker='s',alpha=0.75)
ln3=agg.nf_pct_chg.plot(secondary_y=True,color=colors[2],label=r'Num. firms. (\% chg.), right',marker='x',alpha=0.75)

handles,labels = [],[]
for ax in fig.axes:
    for h,l in zip(*ax.get_legend_handles_labels()):
        handles.append(h)
        labels.append(l)

ax = fig.axes[0]
ax.set_xlabel('')
ax.set_xticks(range(y0,y1+1))
plt.legend(handles,labels,loc='best',prop={'size':6})

plt.savefig("output/bra_rer_dep1.pdf",bbox_inches='tight')
plt.close('all')


########################################################################
##### by destination
x = reset_multiindex(df.groupby(['d','y','rer']).agg(agg_fns).reset_index(),3,'')
    
x['min_y'] = x.groupby('d')['y'].transform(lambda x: x.min())
x['max_y'] = x.groupby('d')['y'].transform(lambda x: x.max())
x['nf0']=x.groupby('d')['nf'].transform(lambda xx: xx.iloc[0])
x=x[x.min_y==y0].reset_index(drop=True)
x=x[x.max_y==y1].reset_index(drop=True)

p90 = x['nf0'].quantile(0.9)
p50 = x['nf0'].quantile(0.5)
x['grp'] = np.nan
x.loc[x.nf0<=p50,'grp']=0
x.loc[x.nf0>p50,'grp']=1

df0=df
for yy in range(y0,y1+1):
    f_ycnt = df.groupby(['f','d'])['y'].nunique().reset_index()
    f0 = f_ycnt.loc[f_ycnt.y==(y1-y0+1),["d","f"]].reset_index(drop=True)
    df0 = pd.merge(left=df0,right=f0,how='inner',on=['d','f'])
                                            
xf0 = df0.groupby(['d','y'])['v'].sum().reset_index().rename(columns={'v':'exports_f0'})
x = pd.merge(left=x,right=xf0,how='left',on=['d','y'])

##### add on the WDI controls
controls = pd.read_pickle('output/wdi_data.pik')
x = pd.merge(left=x,right=controls,how='left',on=['d','y'])
x = x.sort_values(by=['d','y'],ascending=[True,True]).reset_index(drop=True)

##### normalize everything to 1 in initial period
g = x.groupby(['d'])
for c in ['exports','exports_f0','nf','rer','NER','CPI','NGDP_USD','NGDP_LCU','RGDP_USD','RGDP_LCU','Imports_USD','Imports_LCU']:
    x[c+'_log_chg'] = g[c].transform(lambda z: np.log(z/z.iloc[0]))
    x[c+'_pct_chg'] = g[c].transform(lambda z: 100*(z/z.iloc[0]-1.0))
    
#x['te']=x.exports_log_chg/x.rer_log_chg

##### estimate time fixed effects by group
#f1 = 'exports_log_chg ~ NGDP_USD_log_chg + Imports_USD_log_chg + C(d) + C(y):C(grp) -1'
f1 = 'np.log(exports) ~ np.log(NGDP_USD) + np.log(Imports_USD) + C(d) + C(y):C(grp)'
res1 = ols(formula=f1,data=x).fit(cov_type='HC0')

f2 = 'np.log(nf) ~ np.log(NGDP_USD) + np.log(Imports_USD) + C(d) + C(y):C(grp)'
res2 = ols(formula=f2,data=x).fit(cov_type='HC0')

f3 = 'np.log(exports_f0) ~ np.log(NGDP_USD) + np.log(Imports_USD) + C(d) + C(y):C(grp)'
res3 = ols(formula=f3,data=x).fit(cov_type='HC0')

f4 = 'np.log(exports_f0) ~ np.log(NGDP_USD) + np.log(Imports_USD) + C(d) + C(y)'
res40 = ols(formula=f4,data=x[x.grp==0]).fit(cov_type='HC0')
res41 = ols(formula=f4,data=x[x.grp==1]).fit(cov_type='HC0')

y = x.groupby(['grp','y'])[['nf_pct_chg','nf_log_chg','exports_pct_chg','exports_log_chg']].mean().reset_index()


# #####
effect1 = np.zeros((2,y1-y0+1))
effect2 = np.zeros((2,y1-y0+1))
effect3 = np.zeros((2,y1-y0+1))
for y in range(1,y1-y0+1):
    effect1[0,y] = res1.params['C(y)[T.%d]:C(grp)[0.0]'%(y+y0)]
    effect2[0,y] = res2.params['C(y)[T.%d]:C(grp)[0.0]'%(y+y0)]
    effect3[0,y] = res3.params['C(y)[T.%d]:C(grp)[0.0]'%(y+y0)]

    effect1[1,y] = res1.params['C(y)[T.%d]:C(grp)[1.0]'%(y+y0)]
    effect2[1,y] = res2.params['C(y)[T.%d]:C(grp)[1.0]'%(y+y0)]
    effect3[1,y] = res3.params['C(y)[T.%d]:C(grp)[1.0]'%(y+y0)]

fig,axes=plt.subplots(1,2,figsize=(6.5,3),sharex=True,sharey=False)

years=range(y0,y1+1)

axes[0].plot(years,effect1[0,:],color=colors[0],label='Hard',alpha=0.75,marker='o')
axes[0].plot(years,effect1[1,:],color=colors[1],label='Easy',alpha=0.75,marker='s')

axes[1].plot(years,effect2[0,:],color=colors[0],label='Hard',alpha=0.75,marker='o')
axes[1].plot(years,effect2[1,:],color=colors[1],label='Easy',alpha=0.75,marker='s')

#axes[2].plot(years,effect3[0,:],color=colors[0],label='Hard',alpha=0.75,marker='o')
#axes[2].plot(years,effect3[1,:],color=colors[1],label='Easy',alpha=0.75,marker='s')

axes[0].set_title(r'(a) Exports')
axes[1].set_title(r'(b) Num. firms')
#axes[2].set_title(r"(c) Exports (1998 exporters only)")
axes[0].set_xlim(y0,y1)
axes[0].set_xticks(range(y0,y1+1))
axes[0].legend(loc='best',prop={'size':6})
fig.subplots_adjust(hspace=0.15,wspace=0.225)
plt.savefig("output/bra_rer_dep2.pdf",bbox_inches='tight')
plt.close('all')

