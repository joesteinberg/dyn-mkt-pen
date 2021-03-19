import numpy as np
import pandas as pd
import sys

from statsmodels.api import OLS
from statsmodels.formula.api import ols
import patsy


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
colors=['#377eb8','#e41a1c','#4daf4a','#984ea3','#ff7f00','#ffff33']
fmts = ['o','s','D','X','P']


#############################################################################

# df = pd.read_csv('../c/output/lf_dyn.csv',sep=',')

# d_easy='USA'
# d_hard='JPN'
# ix0=35
# ix1=40
# iz=90

# dfh = df[df.d==d_hard].reset_index(drop=True)
# dfe = df[df.d==d_easy].reset_index(drop=True)

# dfh_0 = dfh[dfh['ix']==ix0].reset_index(drop=True)
# dfe_0 = dfe[dfe['ix']==ix0].reset_index(drop=True)
# dfh_1 = dfh[dfh['ix']==ix1].reset_index(drop=True)
# dfe_1 = dfe[dfe['ix']==ix1].reset_index(drop=True)

# dfh_0 = dfh_0[dfh_0.iz==iz].reset_index(drop=True)
# dfe_0 = dfe_0[dfe_0.iz==iz].reset_index(drop=True)
# dfh_1 = dfh_1[dfh_1.iz==iz].reset_index(drop=True)
# dfe_1 = dfe_1[dfe_1.iz==iz].reset_index(drop=True)


#############################################################################

# fig,axes=plt.subplots(1,3,figsize=(7,3.0),sharex=True,sharey=False)

# cols=['mktpen','cost','cost_profit_ratio']
# transform = [lambda x:x,lambda x:np.log(x),lambda x:x]
# titles=['(a) Market penetration rate','(b) Marketing cost (log)','(c) Marketing cost/profits']

# for i in range(3):
#         c=cols[i]
#         t=transform[i]
#         axes[i].plot(dfh_0[c][0:8].apply(t),
#                      color=colors[0],
#                      marker='o',
#                      alpha=0.75,
#                      markeredgewidth=0,
#                      label=r'Hard dest, low $z$')

#         axes[i].plot(dfe_0[c][0:8].apply(t),
#                      color=colors[2],
#                      marker='o',
#                      alpha=0.75,
#                      markeredgewidth=0,
#                      label=r'Easy dest, low $z$')

#         axes[i].plot(dfh_1[c][0:8].apply(t),
#                      color=colors[1],
#                      marker='s',
#                      alpha=0.75,
#                      markeredgewidth=0,
#                      label=r'Hard dest, high $z$')

#         axes[i].plot(dfe_1[c][0:8].apply(t),
#                      color=colors[3],
#                      marker='s',
#                      alpha=0.75,
#                      markeredgewidth=0,
#                      label=r'Easy dest, high $z$')

#         axes[i].set_title(titles[i],y=1.03)
#         axes[i].set_xlabel('Years since entry')

# axes[0].set_ylim(0,1)
# axes[0].set_xlim(-0.5,4.5)
# axes[0].annotate(xy=(50,146),xytext=(0,0),xycoords='axes points',textcoords='offset points',s=r"Easy dest, high $z$",size=6)
# axes[0].annotate(xy=(46,103),xytext=(0,0),xycoords='axes points',textcoords='offset points',s=r"Easy dest, low $z$",size=6)
# axes[0].annotate(xy=(39,63),xytext=(0,0),xycoords='axes points',textcoords='offset points',s=r"Hard dest, high $z$",size=6)
# axes[0].annotate(xy=(65,28),xytext=(0,0),xycoords='axes points',textcoords='offset points',s=r"Hard dest, low $z$",size=6)

# fig.subplots_adjust(hspace=0.2,wspace=0.2)
# plt.savefig('output/export_cost_lf_dyn.pdf',bbox='tight')
# plt.close('all')


###########################################################

print('Loading the simulated data...')

max_tenure_scalar=5

df2 = pd.read_pickle('output/model_microdata_processed.pik')
df2['nf'] = df2.groupby(['d','y'])['f'].transform(lambda x: x.nunique())
tmp2 = df2.groupby('d')['nf'].mean().reset_index()
p50 = tmp2.nf.quantile(0.5)
p90 = tmp2.nf.quantile(0.90)               
df2['grp'] = np.nan
df2.loc[df2.nf<p50,'grp']=0
df2.loc[df2.nf>p90,'grp']=1
df2=df2[df2.tenure.notnull()].reset_index(drop=True)
df2.loc[df2.tenure>max_tenure_scalar,'tenure']=max_tenure_scalar
df2.loc[df2.max_tenure>max_tenure_scalar,'max_tenure']=max_tenure_scalar
df2s = df2[df2.max_tenure>=max_tenure_scalar].reset_index(drop=True)

df2['drank'] = df2.groupby(['f','y'])['v']\
                  .transform(lambda x: x.rank(ascending=False))\
                  .astype(int)

df2.loc[df2.drank>=10,'drank']=10
df2.loc[df2.drank.isin(range(5,10)),'drank'] = 6

# ###########################################################
# print('Plotting export costs by difficulty...')

# icols = ['d','y','nf']
# vcols = ['cost','cost2']
# agged = df2.groupby(icols)[vcols].mean().reset_index()
# agged_e = df2[df2.entry==1].groupby(icols)[vcols].mean().reset_index().rename(columns = {'cost':'cost_e','cost2':'cost2_e'})
# agged_i = df2[df2.incumbent==1].groupby(icols)[vcols].mean().reset_index().rename(columns = {'cost':'cost_i','cost2':'cost2_i'})
# agged = pd.merge(left=agged,right=agged_e,how='left',on=icols)
# agged = pd.merge(left=agged,right=agged_i,how='left',on=icols)\

# agged['cost_erel'] = agged['cost_e']/agged['cost_i']
# agged['cost2_erel'] = agged['cost2_e']/agged['cost2_i']

# agged2 = agged.groupby('d').mean().reset_index()

# fig, axes = plt.subplots(2,2,figsize=(6.5,6.5),sharex=True,sharey=False)

# axes[0,0].scatter(np.log(agged2.nf),agged2.cost,color=colors[0],alpha=0.75)
# axes[0,1].scatter(np.log(agged2.nf),agged2.cost2,color=colors[0],alpha=0.75)
# axes[1,0].scatter(np.log(agged2.nf),agged2.cost_erel,color=colors[0],alpha=0.75)
# axes[1,1].scatter(np.log(agged2.nf),agged2.cost2_erel,color=colors[0],alpha=0.75)

# axes[0,0].set_title('(a) Export cost')
# axes[0,1].set_title('(b) Export cost/profits')
# axes[1,0].set_title('(c) Rel. entrant export cost')
# axes[1,1].set_title('(d) Rel entrant export cost/profits')
# axes[1,0].set_xlabel('Number of exporters (log)')
# axes[1,1].set_xlabel('Number of exporters (log)')
# fig.subplots_adjust(hspace=0.2,wspace=0.2)
# plt.savefig('output/export_cost_by_d.pdf',bbox_inches='tight')


# sys.exit()

###########################################################


print('Estimating duration-tenure effect regressions on simulated data...')

f1 = 'np.log(cost) ~ C(d) + C(tenure):C(max_tenure)-1'
y,X = patsy.dmatrices(f1, df2[df2.grp==0], return_type='dataframe')
for c in X.columns:
        if(X[c].sum()<1.0e-10):
                X.drop(c,axis=1,inplace=True)
sreg1_a = OLS(y,X).fit(cov_type='HC0')

y,X = patsy.dmatrices(f1, df2[df2.grp==1], return_type='dataframe')
for c in X.columns:
        if(X[c].sum()<1.0e-10):
                X.drop(c,axis=1,inplace=True)
sreg1_b = OLS(y,X).fit(cov_type='HC0')


f1 = 'cost2 ~ C(d) + C(tenure):C(max_tenure)-1'
y,X = patsy.dmatrices(f1, df2[df2.grp==0], return_type='dataframe')
for c in X.columns:
        if(X[c].sum()<1.0e-10):
                X.drop(c,axis=1,inplace=True)
sreg2_a = OLS(y,X).fit(cov_type='HC0')

y,X = patsy.dmatrices(f1, df2[df2.grp==1], return_type='dataframe')
for c in X.columns:
        if(X[c].sum()<1.0e-10):
                X.drop(c,axis=1,inplace=True)
sreg2_b = OLS(y,X).fit(cov_type='HC0')



seffect1_a = np.zeros((max_tenure_scalar+1,max_tenure_scalar))
seffect1_b = np.zeros((max_tenure_scalar+1,max_tenure_scalar))
seffect2_a = np.zeros((max_tenure_scalar+1,max_tenure_scalar))
seffect2_b = np.zeros((max_tenure_scalar+1,max_tenure_scalar))

for k in range(1,max_tenure_scalar+1):
        for j in range(k):

                seffect1_a[k,j] = sreg1_a.params["C(max_tenure)[T.%d]"%(k)]
                seffect1_b[k,j] = sreg1_b.params["C(max_tenure)[T.%d]"%(k)]
                seffect2_a[k,j] = sreg2_a.params["C(max_tenure)[T.%d]"%(k)]
                seffect2_b[k,j] = sreg2_b.params["C(max_tenure)[T.%d]"%(k)]

                if(j>0):
                        seffect1_a[k,j] += sreg1_a.params["C(tenure)[T.%d]:C(max_tenure)[%d]"%(j,k)]
                        seffect1_b[k,j] += sreg1_b.params["C(tenure)[T.%d]:C(max_tenure)[%d]"%(j,k)]
                        seffect2_a[k,j] += sreg2_a.params["C(tenure)[T.%d]:C(max_tenure)[%d]"%(j,k)]
                        seffect2_b[k,j] += sreg2_b.params["C(tenure)[T.%d]:C(max_tenure)[%d]"%(j,k)]




fig1,axes1=plt.subplots(2,2,figsize=(6.5,6.5),sharex=True,sharey='row')
               
axes1[0,0].set_title('(a) Log export cost, hard dests.',y=1.0)
axes1[0,1].set_title('(b) Log export cost, easy dests.',y=1.0)
axes1[1,0].set_title('(c) Export cost/profits, hard dests.',y=1.0)
axes1[1,1].set_title('(d) Export cost/profits, easy dests.',y=1.0)

for k in range(1,max_tenure_scalar+1):

        tenure = [x+1 for x in range(k)]
        
        axes1[0,0].plot(tenure,seffect1_a[k,:k],
                        color=colors[k-1],
                        alpha=0.5,
                        marker='o',
                        linewidth=1,
                        markersize=3,
                        #capsize=3,
                        linestyle='-',
                        label='Duration = %d'%(k+1))

        axes1[0,1].plot(tenure,seffect1_b[k,:k],
                        color=colors[k-1],
                        marker='o',
                        alpha=0.5,
                        markersize=3,
                        #capsize=3,
                        linewidth=1,
                        linestyle='-',
                        label='Duration = %d'%(k+1))
        
        axes1[1,0].plot(tenure,seffect2_a[k,:k],
                        color=colors[k-1],
                        alpha=0.5,
                        marker='o',
                        linewidth=1,
                        markersize=3,
                        #capsize=3,
                        linestyle='-',
                        label='Duration = %d'%(k+1))

        axes1[1,1].plot(tenure,seffect2_b[k,:k],
                        color=colors[k-1],
                        marker='o',
                        alpha=0.5,
                        markersize=3,
                        #capsize=3,
                        linewidth=1,
                        linestyle='-',
                        label='Duration = %d'%(k+1))


#axes1[0,0].set_ylim(0,3)
#axes1[1].set_ylim(0,3)
axes1[0,0].legend(loc='best',prop={'size':6})
axes1[1,0].set_xticks(range(1,max_tenure_scalar+1))
axes1[1,1].set_xticks(range(1,max_tenure_scalar+1))
axes1[1,0].set_xlabel('Years in market')
axes1[1,1].set_xlabel('Years in market')
#axes1[0,1].set_yticks([])
#axes1[1,1].set_yticks([])
axes1[0,0].set_ylabel('log export cost (relative to duration = 0)')
axes1[1,0].set_ylabel('export cost/profits (relative to duration = 0)')

fig1.subplots_adjust(hspace=0.2,wspace=0.1)

plt.sca(axes1[0,0])
plt.savefig('output/life_cycle_dyn_export_costs.pdf',bbox_inches='tight')

plt.close('all')



##############################################################################################
print('Estimating costs by nd_group and drank...')

xl=['1','2','3','4','5-9','10+']

f1 = 'np.log(cost) ~ C(d) + C(drank):C(nd_group) -1'
f2 = 'cost2 ~ C(d) + C(drank):C(nd_group) -1'

y,X = patsy.dmatrices(f1, df2, return_type='dataframe')
for c in X.columns:
        if(X[c].sum()<1.0e-10):
                X.drop(c,axis=1,inplace=True)
sreg_v = OLS(y,X).fit(cov_type='HC0')

y,X = patsy.dmatrices(f2, df2, return_type='dataframe')
for c in X.columns:
        if(X[c].sum()<1.0e-10):
                X.drop(c,axis=1,inplace=True)
sreg_x = OLS(y,X).fit(cov_type='HC0')

ngrps = len(df2.nd_group.unique())
x=df2.drank.unique()
x.sort()
drank = list(x)


seffect_v = np.zeros((ngrps,ngrps))
seffect_x = np.zeros((ngrps,ngrps))

for k in range(1,ngrps):
        for j in range(k+1):

                seffect_v[k,j] = sreg_v.params["C(nd_group)[T.%d]"%(drank[k])]
                seffect_x[k,j] = sreg_x.params["C(nd_group)[T.%d]"%(drank[k])]
                
                if(j>0):
                        seffect_v[k,j] += sreg_v.params["C(drank)[T.%d]:C(nd_group)[%d]"%(drank[j],drank[k])]
                        seffect_x[k,j] += sreg_x.params["C(drank)[T.%d]:C(nd_group)[%d]"%(drank[j],drank[k])]

                
fig1,axes1=plt.subplots(1,2,figsize=(7,3),sharex=True,sharey=False)
               
axes1[0].set_title('(a) Export cost  (log)',y=1.025)
axes1[1].set_title('(b) Export cost/profits',y=1.025)

for k in range(1,ngrps):

        drank = [x+1 for x in range(k+1)]
    
        axes1[0].plot(drank,seffect_v[k,:k+1],
                      color=colors[k-1],
                      alpha=0.5,
                      marker='o',
                      linewidth=1,
                      markersize=3,
                      #capsize=3,
                      linestyle='-',
                      label='Num. dests. = %s'%(xl[k]))
        
        axes1[1].plot(drank,seffect_x[k,:k+1],
                      color=colors[k-1],
                      marker='o',
                      alpha=0.5,
                      markersize=3,
                      #capsize=3,
                      linewidth=1,
                      linestyle='-',
                      label='Num. dests. = %s'%(xl[k]))


#axes1[0].set_ylim(0,3)
#axes1[1].set_ylim(0,3)
axes1[0].legend(loc='upper right',prop={'size':6})
#axes1[0].set_xticks(range(1,ngrps+2))
#axes1[1].set_xticks(range(1,ngrps+2)
axes1[0].set_xticks(range(1,ngrps+1))
axes1[1].set_xticks(range(1,ngrps+1))
axes1[0].set_xticklabels(xl)
axes1[1].set_xticklabels(xl)
axes1[0].set_xlabel('Destination rank')
axes1[1].set_xlabel('Destination rank')
#axes1[1].set_yticks([])
axes1[0].set_ylabel('Single-dest exporters = 0')

fig1.subplots_adjust(hspace=0.2,wspace=0.2)

plt.sca(axes1[0])
plt.savefig('output/cost_by_nd_drank.pdf',bbox_inches='tight')

plt.close('all')
