import numpy as np
import pandas as pd
import sys
from statsmodels.formula.api import ols
from statsmodels.formula.api import wls

import locale
locale.setlocale(locale.LC_ALL,'en_US.utf8')

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator

mpl.rc('text', usetex=True)
#mpl.rc('savefig',bbox_inches='tight')
mpl.rc('savefig',format='pdf')
mpl.rc('font',**{'family':'serif','serif':['Palatino'],'size':8})
mpl.rc('font',size=8)
mpl.rc('lines',linewidth=1)

alpha=0.8
colors=['#377eb8','#e41a1c','#4daf4a','#984ea3','#ff7f00','#ffff33']
fmts = ['o','s','x','d','p']

def pct_chg(x):
        return 100*(x/x.iloc[0]-1.0)

#############################################################################

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
                tmp.loc[tmp.expart_rate<=p50,'grp']=0
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

        if('mktpen_rate' in df.columns):
                g = df.groupby(['grp','t'])[['expart_rate','exports_pct_chg','expart_rate_pct_chg','mktpen_rate','mktpen_rate_pct_chg','tau_pct_chg','trade_elasticity']].mean().reset_index()
        else:
                g = df.groupby(['grp','t'])[['expart_rate','exports_pct_chg','expart_rate_pct_chg','tau_pct_chg','trade_elasticity']].mean().reset_index()

        return g

#############################################################################

fnames=['../c/output/tr_dyn_perm_tau_drop.csv',
        '../c/output/tr_dyn_rer_dep.csv']
G=[]
for f in fnames:
    G.append(load_tr_dyn(f,1))

G2=None
pref=''
altlab=''

if len(sys.argv)>1 and sys.argv[1]=='sunk':
        pref='sunkcost'
        altlab='sunk cost'
        fnames=['../c/output/tr_dyn_perm_tau_drop_sunkcost.csv',
                '../c/output/tr_dyn_rer_dep_sunkcost.csv']

elif len(sys.argv)>1 and sys.argv[1]=='acr':
        pref='acr'
        altlab='exog. entrant dyn.'
        fnames=['../c/output/tr_dyn_perm_tau_drop_acr.csv',
                '../c/output/tr_dyn_rer_dep_acr.csv']

elif len(sys.argv)>1 and sys.argv[1]=='smp':
        pref='smp'
        altlab='static mkt. pen.'
        fnames=['../c/output/tr_dyn_perm_tau_drop_static_mkt_pen.csv',
                '../c/output/tr_dyn_rer_dep_static_mkt_pen.csv']
        
if pref!='':
        G2=[]
        for f in fnames:
                G2.append(load_tr_dyn(f,0))

#############################################################################

fig,axes=plt.subplots(1,3,figsize=(7,2.5),sharex=True,sharey=False)

cols=['expart_rate_pct_chg','mktpen_rate_pct_chg','trade_elasticity']
titles=[r'(a) Num. exporters (\% chg)',
        r'(b) Avg. mkt. pen. (\% chg)',
        r'(c) Trade elasticity']

g=G[0]
for i in range(3):
        c=cols[i]
        axes[i].axhline(0,linestyle=':',color='black',alpha=0.5)
        
        axes[i].plot(g[g.grp==0][c].reset_index(drop=True),
                     color=colors[0],
                     marker=fmts[0],
                     alpha=0.75,
                     markeredgewidth=0,
                     label=r'Hard dests.')
        
        axes[i].plot(g[g.grp==1][c].reset_index(drop=True),
                     color=colors[1],
                     marker=fmts[1],
                     alpha=0.75,
                     markeredgewidth=0,
                     label=r'Easy dests.')

        axes[i].set_title(titles[i],y=1.03)
        
#axes[0].set_ylim(-40,60)
#axes[1].set_ylim(-12,12)
#axes[2].set_ylim(0,6)
axes[1].set_xlim(0,10)
axes[1].set_xlabel('Years since policy change')
axes[1].set_xticks(range(11))
axes[1].set_xticklabels([('%d'%t if t%2==0 else '') for t in range(11)])
axes[2].legend(loc='lower right',prop={'size':6})


#ann1=axes[0].annotate(xy=(55,150),xytext=(0,0),xycoords='axes points',textcoords='offset points',s=r"Avg. of hard dests.",size=6)
#ann2=axes[0].annotate(xy=(43,102),xytext=(0,0),xycoords='axes points',textcoords='offset points',s=r"Avg. of easy dests.",size=6)

fig.subplots_adjust(hspace=0.2,wspace=0.25)
plt.savefig('output/tr_dyn_perm_tau_drop.pdf',bbox_inches='tight')


if pref !='':
        g=G2[0]

        R = [0,2]
        if pref=='smp':
                R = [0,1,2]
                
        for i in R:
                c=cols[i]
                axes[i].axhline(0,linestyle=':',color='black',alpha=0.5)
        
                axes[i].plot(g[g.grp==0][c].reset_index(drop=True),
                             color=colors[3],
                             marker=fmts[3],
                             alpha=0.75,
                             markeredgewidth=1,
                             linestyle='--',
                             label=r'Hard dests. ('+altlab+')')
                
                axes[i].plot(g[g.grp==1][c].reset_index(drop=True),
                             color=colors[4],
                             marker=fmts[4],
                             alpha=0.75,
                             markeredgewidth=1,
                             linestyle='--',
                             label=r'Easy dests. ('+altlab+')')

        

        #axes[0].annotate(xy=(55,150),xytext=(0,0),xycoords='axes points',textcoords='offset points',s=r"Avg. of hard dests.",size=6)
        #axes[0].annotate(xy=(43,89),xytext=(0,0),xycoords='axes points',textcoords='offset points',s=r"Avg. of easy dests.",size=6)
        
        #ann1.remove()
        #ann2.remove()
        axes[2].legend(loc='lower right',prop={'size':6})
        fig.subplots_adjust(hspace=0.2,wspace=0.25)
        plt.savefig('output/tr_dyn_perm_tau_drop_'+pref+'.pdf',bbox_inches='tight')
        plt.close('all')



#############################################################################

fig,axes=plt.subplots(1,3,figsize=(7,2.5),sharex=True,sharey=False)

cols=['exports_pct_chg','expart_rate_pct_chg','mktpen_rate_pct_chg']
titles=[r'(a) Exports (\% chg)',
        r'(b) Num. exporters (\% chg)',
        r'(c) Avg. mkt. pen (\% chg)']

g=G[1]
for i in range(3):
        c=cols[i]

        axes[i].axhline(0,linestyle=':',color='black',alpha=0.5)
        
        ln1=axes[i].plot(g[g.grp==0][c].reset_index(drop=True),
                         color=colors[0],
                         marker=fmts[0],
                         alpha=0.75,
                         markeredgewidth=0,
                         label=r'Avg. of hard dests.')

        ln2=axes[i].plot(g[g.grp==1][c].reset_index(drop=True),
                         color=colors[1],
                         marker=fmts[1],
                         alpha=0.75,
                         markeredgewidth=0,
                         label=r'Avg. of easy dests.')

        ax2 = axes[i].twinx()
        ax2.set_ylim(0,9)
        ln3=ax2.plot(-g[g.grp==1]['tau_pct_chg'].reset_index(drop=True),
                      color=colors[2],
                      marker=fmts[2],
                      alpha=0.75,
                      markeredgewidth=1,
                      label=r'RER (right axis)')
        

        axes[i].set_title(titles[i],y=1.03)
        
#axes[0].set_ylim(-40,60)
#axes[1].set_ylim(-12,12)
#axes[2].set_ylim(0,6)
axes[1].set_xlim(0,10)
axes[1].set_xlabel('Years since policy change')
axes[1].set_xticks(range(11))
axes[1].set_xticklabels([('%d'%t if t%2==0 else '') for t in range(11)])

lns=ln1+ln2+ln3
labs = [l.get_label() for l in lns]
axes[0].legend(lns,labs,prop={'size':6},loc='lower right')
#ann1=axes[2].annotate(xy=(47,150),xytext=(0,0),xycoords='axes points',textcoords='offset points',s=r"Avg. of hard dests.",size=6)
#ann2=axes[2].annotate(xy=(35,35),xytext=(0,0),xycoords='axes points',textcoords='offset points',s=r"Avg. of easy dests.",size=6)
#ann3=axes[2].annotate(xy=(28,80),xytext=(0,0),xycoords='axes points',textcoords='offset points',s=r"RER (right axis)",size=6)

fig.subplots_adjust(hspace=0.2,wspace=0.35)
plt.savefig('output/tr_dyn_rer_dep.pdf',bbox_inches='tight')



if pref!='':
        g=G2[1]

        R = [0,1]
        if pref=='smp':
                R = [0,1,2]

        for i in R:
                c=cols[i]

                axes[i].axhline(0,linestyle=':',color='black',alpha=0.5)
        
                ln4=axes[i].plot(g[g.grp==0][c].reset_index(drop=True),
                                 color=colors[3],
                                 marker=fmts[2],
                                 alpha=0.75,
                                 markeredgewidth=1,
                                 linestyle='--',
                                 label=r'Hard dests. ('+altlab+')')

                ln5=axes[i].plot(g[g.grp==1][c].reset_index(drop=True),
                                 color=colors[4],
                                 marker=fmts[4],
                                 alpha=0.75,
                                 markeredgewidth=1,
                                 linestyle='--',
                                 label=r'Easy dests. ('+altlab+')')

   
        #ann1.remove()
        #ann2.remove()
        #ann3.remove()

        lns = ln1+ln2+ln3+ln4+ln5
        labs = [l.get_label() for l in lns]
        axes[0].legend(lns,labs,loc='best',prop={'size':6})

        fig.subplots_adjust(hspace=0.2,wspace=0.35)
        plt.savefig('output/tr_dyn_rer_dep_'+pref+'.pdf',bbox_inches='tight')



plt.close('all')
