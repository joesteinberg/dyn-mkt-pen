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

#############################################################################

df = pd.read_csv('../c/output/lf_dyn.csv',sep=',')

d_easy='USA'
d_hard='COL'
ix0=35
ix1=40
iz=100

dfh = df[df.d==d_hard].reset_index(drop=True)
dfe = df[df.d==d_easy].reset_index(drop=True)

dfh_0 = dfh[dfh['ix']==ix0].reset_index(drop=True)
dfe_0 = dfe[dfe['ix']==ix0].reset_index(drop=True)
dfh_1 = dfh[dfh['ix']==ix1].reset_index(drop=True)
dfe_1 = dfe[dfe['ix']==ix1].reset_index(drop=True)

dfh_0 = dfh_0[dfh_0.iz==iz].reset_index(drop=True)
dfe_0 = dfe_0[dfe_0.iz==iz].reset_index(drop=True)
dfh_1 = dfh_1[dfh_1.iz==iz].reset_index(drop=True)
dfe_1 = dfe_1[dfe_1.iz==iz].reset_index(drop=True)


#############################################################################

fig,axes=plt.subplots(1,3,figsize=(7,3.0),sharex=True,sharey=False)

cols=['mktpen','cost','cost_profit_ratio']
transform = [lambda x:x,lambda x:np.log(x),lambda x:x]
titles=['(a) Market penetration rate','(b) Marketing cost (log)','(c) Marketing cost/profits']

for i in range(3):
        c=cols[i]
        t=transform[i]
        axes[i].plot(dfh_0[c][0:8].apply(t),
                     color=colors[0],
                     marker='o',
                     alpha=0.75,
                     markeredgewidth=0,
                     label=r'Hard dest, low $z$')

        axes[i].plot(dfe_0[c][0:8].apply(t),
                     color=colors[2],
                     marker='o',
                     alpha=0.75,
                     markeredgewidth=0,
                     label=r'Easy dest, low $z$')

        axes[i].plot(dfh_1[c][0:8].apply(t),
                     color=colors[1],
                     marker='s',
                     alpha=0.75,
                     markeredgewidth=0,
                     label=r'Hard dest, high $z$')

        axes[i].plot(dfe_1[c][0:8].apply(t),
                     color=colors[3],
                     marker='s',
                     alpha=0.75,
                     markeredgewidth=0,
                     label=r'Easy dest, high $z$')

        axes[i].set_title(titles[i],y=1.03)
        axes[i].set_xlabel('Years since entry')

axes[0].set_ylim(0,1)
axes[0].set_xlim(-0.5,4.5)
axes[0].annotate(xy=(50,146),xytext=(0,0),xycoords='axes points',textcoords='offset points',s=r"Easy dest, high $z$",size=6)
axes[0].annotate(xy=(46,103),xytext=(0,0),xycoords='axes points',textcoords='offset points',s=r"Easy dest, low $z$",size=6)
axes[0].annotate(xy=(39,63),xytext=(0,0),xycoords='axes points',textcoords='offset points',s=r"Hard dest, high $z$",size=6)
axes[0].annotate(xy=(65,28),xytext=(0,0),xycoords='axes points',textcoords='offset points',s=r"Hard dest, low $z$",size=6)

fig.subplots_adjust(hspace=0.2,wspace=0.2)
plt.savefig('output/export_cost_lf_dyn.pdf',bbox='tight')
plt.close('all')
