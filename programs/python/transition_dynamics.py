import numpy as np
import pandas as pd
import sys

import locale
locale.setlocale(locale.LC_ALL,'en_US.utf8')

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

mpl.rc('text', usetex=True)
#mpl.rc('savefig',bbox_inches='tight')
mpl.rc('savefig',format='pdf')
mpl.rc('font',**{'family':'serif','serif':['Palatino'],'size':8})
mpl.rc('font',size=8)
mpl.rc('lines',linewidth=1)
mpl.rcParams['savefig.pad_inches'] = 0

alpha=0.8
colors=['#377eb8','#e41a1c','#4daf4a','#984ea3','#ff7f00','#ffff33']
fmts = ['o','s','d','x','p']

figpath = 'output/transitions/'

fnames=['../c/output/tr_dyn_perm_tau_drop.csv',
        '../c/output/tr_dyn_perm_tau_drop_static_mkt_pen.csv',
        '../c/output/tr_dyn_perm_tau_drop_sunkcost.csv',
        '../c/output/tr_dyn_perm_tau_drop_acr.csv']

def pct_chg(x):
        return 100*(x/x.iloc[0]-1.0)

#############################################################################
# average transition dynamics by group

grps=None

def load_tr_dyn(fname,calc_grps):
        global grps
        df = pd.read_csv(fname,sep=',')

        df['n0'] = df.groupby('d')['expart_rate'].transform(lambda x:x.min())
        df = df[df.n0>0.004].reset_index(drop=True)

        if(calc_grps):
                tmp = df.loc[df.t==0,:]
                p90 = tmp['expart_rate'].quantile(0.9)
                p50 = tmp['expart_rate'].quantile(0.5)
                tmp['grp'] = np.nan
                tmp.loc[tmp.expart_rate<p50,'grp']=0
                tmp.loc[tmp.expart_rate>p90,'grp']=1
                grps=tmp
                
        df=pd.merge(left=df,right=grps[['d','grp']],how='left',on='d')

        df['expart_rate_pct_chg'] = df.groupby(['d'])['expart_rate']\
                                      .transform(pct_chg)

        if('mktpen_rate' in df.columns):
                df['mktpen_rate_pct_chg'] = df.groupby(['d'])['mktpen_rate']\
                                              .transform(pct_chg)

        df['exports_pct_chg'] = df.groupby(['d'])['exports']\
                                      .transform(pct_chg)

        df['tau_pct_chg'] = df.groupby(['d'])['tau']\
                                      .transform(pct_chg)

        cols=[]
        if('mktpen_rate' in df.columns):
                cols = ['expart_rate','exports_pct_chg','expart_rate_pct_chg','mktpen_rate','mktpen_rate_pct_chg',
                        'tau_pct_chg','trade_elasticity']
        else:
                cols = ['expart_rate','exports_pct_chg','expart_rate_pct_chg','tau_pct_chg','trade_elasticity']
                
        total = df.groupby('t').mean().reset_index()
        total['grp']=-999
        grouped = df.groupby(['grp','t']).mean().reset_index()
        
        return total.append(grouped)

results=[]
flag=1
for f in fnames:
        results.append(load_tr_dyn(f,flag))
        flag=0

cols = ['trade_elasticity','expart_rate_pct_chg','mktpen_rate_pct_chg']
grps=[-999,0,1]
titles=[r'(a) Trade elast: all markets',
        r'(b) Trade elast: hard markets',
        r'(c) Trade elast: easy markets',
        r'(d) Num exporters: all markets',
        r'(e) Num exporters: hard markets',
        r'(f) Num exporters: easy markets',
        r'(g) Mkt pen: all markets',
        r'(h) Mkt pen: hard markets',
        r'(i) Mkt pen: easy markets']
        
labs=['Baseline','Static MP','Sunk cost','Exog NED']

fig,axes=plt.subplots(3,3,figsize=(7.5,7.5),sharex=True,sharey=False)

cnt=0
for k in range(len(cols)):

        ax=axes[k]
        
        for i in range(len(grps)):
                
                ax[i].set_title(titles[cnt],y=1.0)
                cnt += 1
                
                for j in range(len(results)):
                        if(k<2 or j<2):
                                ax[i].plot(results[j][results[j].grp==grps[i]][cols[k]].reset_index(drop=True),
                                           color=colors[j],
                                           marker=fmts[j],
                                           markersize=3,
                                           alpha=0.65,
                                           label=labs[j])


                                
                if(k==0):
                        ax[i].set_ylim(3.5,5.5)
                elif(k==1):
                        ax[i].set_ylim(0,30)
                else:
                        ax[i].set_ylim(0,8)

for i in range(len(grps)):

        for k in range(len(grps)):
                axes[-1][k].set_xlim(1,10)
                
        axes[-1][i].set_xlabel('Years since policy change')
        axes[-1][i].set_xticks(range(1,11))


axes[0][0].set_ylabel('$\\Delta$ log exports/$\\Delta$ log $\\tau$')
axes[1][0].set_ylabel('percent change')
axes[2][0].set_ylabel('percent change')


axes[0][2].legend(loc='upper right',prop={'size':6})
fig.subplots_adjust(hspace=0.25,wspace=0.25)
plt.savefig(figpath + 'fig7_tr_dyn_perm_tau_drop.pdf',bbox_inches='tight')
