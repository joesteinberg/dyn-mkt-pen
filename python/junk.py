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
mpl.rc('savefig',bbox='tight')
mpl.rc('savefig',format='pdf')
mpl.rc('font',**{'family':'serif','serif':['Palatino'],'size':8})
mpl.rc('font',size=8)
mpl.rc('lines',linewidth=1)

alpha=0.8
colors=['#377eb8','#e41a1c','#4daf4a','#984ea3']

def pct_chg(x):
        return 100*(x/x.iloc[0]-1.0)

#############################################################################

def load_tr_dyn(fname):
        df = pd.read_csv(fname,sep=',')

        tmp = df.loc[df.t==0,:]
        p90 = tmp['expart_rate'].quantile(0.9)
        p50 = tmp['expart_rate'].quantile(0.5)
        tmp['grp'] = np.nan
        tmp.loc[tmp.expart_rate<p50,'grp']=0
        tmp.loc[tmp.expart_rate>p90,'grp']=1
        df=pd.merge(left=df,right=tmp[['d','grp']],how='left',on='d')

        df['expart_rate_pct_chg'] = df.groupby(['d'])['expart_rate']\
                                      .transform(pct_chg)

        df['mktpen_rate_pct_chg'] = df.groupby(['d'])['mktpen_rate']\
                                      .transform(pct_chg)

        df['exports_pct_chg'] = df.groupby(['d'])['exports']\
                                      .transform(pct_chg)

        df['tau_pct_chg'] = df.groupby(['d'])['tau']\
                                      .transform(pct_chg)

        g = df.groupby(['grp','t'])[['expart_rate','exports_pct_chg','expart_rate_pct_chg',
                                     'mktpen_rate','mktpen_rate_pct_chg','tau_pct_chg',
                                     'trade_elasticity']].mean().reset_index()

        return g

#############################################################################

fnames=['../c/output/tr_dyn_perm_tau_drop.csv',
        '../c/output/tr_dyn_rer_dep.csv',
        '../c/output/tr_dyn_perm_tau_drop_uncertain.csv']


G=[]
for f in fnames:
    G.append(load_tr_dyn(f))

#############################################################################

fig,axes=plt.subplots(1,3,figsize=(7,3.0),sharex=True,sharey=False)

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
                     marker='o',
                     alpha=0.75,
                     markeredgewidth=0,
                     label=r'Avg. of hard dests.')
        
        axes[i].plot(g[g.grp==1][c].reset_index(drop=True),
                     color=colors[1],
                     marker='s',
                     alpha=0.75,
                     markeredgewidth=0,
                     label=r'Avg. of eaasy dests.')

        axes[i].set_title(titles[i],y=1.03)
        
#axes[0].set_ylim(-40,60)
#axes[1].set_ylim(-12,12)
#axes[2].set_ylim(0,6)
axes[1].set_xlim(0,10)
axes[1].set_xlabel('Years since policy change')
axes[1].set_xticks(range(11))
axes[1].set_xticklabels([('%d'%t if t%2==0 else '') for t in range(11)])
#axes[2].legend(loc='lower right',prop={'size':8})


axes[0].annotate(xy=(55,150),xytext=(0,0),xycoords='axes points',textcoords='offset points',s=r"Avg. of hard dests.",size=6)
axes[0].annotate(xy=(43,89),xytext=(0,0),xycoords='axes points',textcoords='offset points',s=r"Avg. of easy dests.",size=6)

fig.subplots_adjust(hspace=0.2,wspace=0.25)
plt.savefig('output/tr_dyn_perm_tau_drop.pdf',bbox='tight')
plt.close('all')





fig,axes=plt.subplots(1,3,figsize=(7,3.0),sharex=True,sharey=False)

cols=['expart_rate_pct_chg','mktpen_rate_pct_chg','trade_elasticity']
titles=[r'(a) Num. exporters (\% chg)',
        r'(b) Avg. mkt. pen. (\% chg)',
        r'(c) Trade elasticity']

for i in range(3):
        c=cols[i]

        g=G[0]

        axes[i].axhline(0,linestyle=':',color='black',alpha=0.5)
                
        axes[i].plot(g[g.grp==0][c].reset_index(drop=True),
                     color=colors[0],
                     marker='o',
                     alpha=0.75,
                     markeredgewidth=0,
                     label=r'Hard dests. ($\tau\downarrow$)')

        axes[i].plot(g[g.grp==1][c].reset_index(drop=True),
                     color=colors[1],
                     marker='s',
                     alpha=0.75,
                     markeredgewidth=0,
                     label=r'Easy dests. ($\tau\downarrow$)')

        g=G[1]
        axes[i].plot(g[g.grp==0][c].reset_index(drop=True),
                     color=colors[2],
                     marker='o',
                     alpha=0.75,
                     markeredgewidth=0,
                     label=r'Hard dests. ($\tau\uparrow$)')

        axes[i].plot(g[g.grp==1][c].reset_index(drop=True),
                     color=colors[3],
                     marker='s',
                     alpha=0.75,
                     markeredgewidth=0,
                     label=r'Easy dests. ($\tau\uparrow$)')

        axes[i].set_title(titles[i],y=1.03)

axes[0].set_ylim(-40,60)
axes[1].set_ylim(-12,12)
axes[2].set_ylim(0,6)
axes[1].set_xlim(0,10)
axes[1].set_xlabel('Years since policy change')

axes[1].set_xticks(range(11))
axes[1].set_xticklabels([('%d'%t if t%2==0 else '') for t in range(11)])
#axes[2].legend(loc='lower right',prop={'size':8})

axes[0].annotate(xy=(65,133),xytext=(0,0),xycoords='axes points',textcoords='offset points',s=r"Hard dest, $\tau\downarrow$",size=6)
axes[0].annotate(xy=(43,92),xytext=(0,0),xycoords='axes points',textcoords='offset points',s=r"Easy dest, $\tau\downarrow$",size=6)
axes[0].annotate(xy=(43,45),xytext=(0,0),xycoords='axes points',textcoords='offset points',s=r"Hard dest, $\tau\uparrow$",size=6)
axes[0].annotate(xy=(65,15),xytext=(0,0),xycoords='axes points',textcoords='offset points',s=r"Easy dest, $\tau\uparrow$",size=6)


fig.subplots_adjust(hspace=0.2,wspace=0.25)
plt.savefig('output/tr_dyn_perm_tau_drop_rise.pdf',bbox='tight')
plt.close('all')


#############################################################################

fig,axes=plt.subplots(1,3,figsize=(7,3.0),sharex=True,sharey=False)

cols=['exports_pct_chg','expart_rate_pct_chg','mktpen_rate_pct_chg']
titles=[r'(a) Exports (\% chg)',
        r'(b) Num. exporters (\% chg)',
        r'(c) Avg. mkt. pen (\% chg)']

g=G[2]
for i in range(3):
        c=cols[i]

        axes[i].axhline(0,linestyle=':',color='black',alpha=0.5)
        
        axes[i].plot(g[g.grp==0][c].reset_index(drop=True),
                     color=colors[0],
                     marker='o',
                     alpha=0.75,
                     markeredgewidth=0,
                     label=r'Hard dests. ($RER\uparrow$)')

        axes[i].plot(g[g.grp==1][c].reset_index(drop=True),
                     color=colors[1],
                     marker='s',
                     alpha=0.75,
                     markeredgewidth=0,
                     label=r'Easy dests. ($RER\uparrow$)')

        axes[i].set_title(titles[i],y=1.03)
        
#axes[0].set_ylim(-40,60)
#axes[1].set_ylim(-12,12)
#axes[2].set_ylim(0,6)
axes[1].set_xlim(0,10)
axes[1].set_xlabel('Years since policy change')
axes[1].set_xticks(range(11))
axes[1].set_xticklabels([('%d'%t if t%2==0 else '') for t in range(11)])

axes[0].annotate(xy=(22,140),xytext=(0,0),xycoords='axes points',textcoords='offset points',s=r"Avg. of hard dests.",size=6)
axes[0].annotate(xy=(48,20),xytext=(0,0),xycoords='axes points',textcoords='offset points',s=r"Avg. of easy dests.",size=6)

fig.subplots_adjust(hspace=0.2,wspace=0.25)
plt.savefig('output/tr_dyn_rer_dep.pdf',bbox='tight')
plt.close('all')





fig,axes=plt.subplots(1,3,figsize=(7,3.0),sharex=True,sharey=False)

cols=['exports_pct_chg','expart_rate_pct_chg','mktpen_rate_pct_chg']
titles=[r'(a) Exports (\% chg)',
        r'(b) Num. exporters (\% chg)',
        r'(c) Avg. mkt. pen (\% chg)']

for i in range(3):
        c=cols[i]

        g=G[2]

        axes[i].axhline(0,linestyle=':',color='black',alpha=0.5)
        
        axes[i].plot(g[g.grp==0][c].reset_index(drop=True),
                     color=colors[0],
                     marker='o',
                     alpha=0.75,
                     markeredgewidth=0,
                     label=r'Hard dests. ($RER\uparrow$)')

        axes[i].plot(g[g.grp==1][c].reset_index(drop=True),
                     color=colors[1],
                     marker='s',
                     alpha=0.75,
                     markeredgewidth=0,
                     label=r'Easy dests. ($RER\uparrow$)')

        g=G[3]
        
        axes[i].plot(g[g.grp==0][c].reset_index(drop=True),
                     color=colors[2],
                     marker='o',
                     alpha=0.75,
                     markeredgewidth=0,
                     label=r'Hard dests. ($RER\downarrow$)')

        axes[i].plot(g[g.grp==1][c].reset_index(drop=True),
                     color=colors[3],
                     marker='s',
                     alpha=0.75,
                     markeredgewidth=0,
                     label=r'Easy dests. ($RER\downarrow$)')

        axes[i].set_title(titles[i],y=1.03)

#axes[0].set_ylim(-40,60)
#axes[1].set_ylim(-12,12)
#axes[2].set_ylim(0,6)
axes[1].set_xlim(0,10)
axes[1].set_xlabel('Years since policy change')

axes[1].set_xticks(range(11))
axes[1].set_xticklabels([('%d'%t if t%2==0 else '') for t in range(11)])
#axes[2].legend(loc='lower right',prop={'size':8})

axes[0].annotate(xy=(20,140),xytext=(0,0),xycoords='axes points',textcoords='offset points',s=r"Hard dest, $RER\uparrow$",size=6)
axes[0].annotate(xy=(30,98),xytext=(0,0),xycoords='axes points',textcoords='offset points',s=r"Easy dest, $RER\uparrow$",size=6)
axes[0].annotate(xy=(30,60),xytext=(0,0),xycoords='axes points',textcoords='offset points',s=r"Hard dest, $RER\downarrow$",size=6)
axes[0].annotate(xy=(20,15),xytext=(0,0),xycoords='axes points',textcoords='offset points',s=r"Easy dest, $RER\downarrow$",size=6)


fig.subplots_adjust(hspace=0.2,wspace=0.25)
plt.savefig('output/tr_dyn_rer_app.pdf',bbox='tight')
plt.close('all')









#########################################################################

fig,axes=plt.subplots(1,3,figsize=(7,3.0),sharex=True,sharey=False)

cols=['expart_rate_pct_chg','mktpen_rate_pct_chg','trade_elasticity']
titles=[r'(a) Num. exporters (\% chg)',
        r'(b) Avg. mkt. pen. (\% chg)',
        r'(c) Trade elasticity']

for i in range(3):
        c=cols[i]
        axes[i].axhline(0,linestyle=':',color='black',alpha=0.5)

        g=G[0]
        
        axes[i].plot(g[g.grp==0][c].reset_index(drop=True),
                     color=colors[0],
                     marker='o',
                     alpha=0.75,
                     markeredgewidth=0,
                     label=r'Avg. of hard dests.')
        
        axes[i].plot(g[g.grp==1][c].reset_index(drop=True),
                     color=colors[1],
                     marker='s',
                     alpha=0.75,
                     markeredgewidth=0,
                     label=r'Avg. of easy dests.')

        g=G[4]
        
        axes[i].plot(g[g.grp==0][c].reset_index(drop=True),
                     color=colors[2],
                     marker='x',
                     alpha=0.75,
                     markeredgewidth=1,
                     label=r'Avg. of hard dests. (uncertain)')
        
        axes[i].plot(g[g.grp==1][c].reset_index(drop=True),
                     color=colors[3],
                     marker='+',
                     alpha=0.75,
                     markeredgewidth=1,
                     label=r'Avg. of easy dests. (uncertain)')


        axes[i].set_title(titles[i],y=1.03)
        
#axes[0].set_ylim(-40,60)
#axes[1].set_ylim(-12,12)
#axes[2].set_ylim(0,6)
axes[1].set_xlim(0,10)
axes[1].set_xlabel('Years since policy change')
axes[1].set_xticks(range(11))
axes[1].set_xticklabels([('%d'%t if t%2==0 else '') for t in range(11)])
#axes[2].legend(loc='lower right',prop={'size':6})


axes[0].annotate(xy=(55,150),xytext=(0,0),xycoords='axes points',textcoords='offset points',s=r"Avg. of hard dests.",size=6)
axes[0].annotate(xy=(43,89),xytext=(0,0),xycoords='axes points',textcoords='offset points',s=r"Avg. of easy dests.",size=6)

axes[0].annotate(xy=(30,125),xytext=(0,0),xycoords='axes points',textcoords='offset points',s=r"Avg. of hard dests. (uncertain)",size=6)
axes[0].annotate(xy=(30,64),xytext=(0,0),xycoords='axes points',textcoords='offset points',s=r"Avg. of easy dests. (uncertain)",size=6)

fig.subplots_adjust(hspace=0.2,wspace=0.25)
plt.savefig('output/tr_dyn_perm_tau_drop_uncertain.pdf',bbox='tight')
plt.close('all')
