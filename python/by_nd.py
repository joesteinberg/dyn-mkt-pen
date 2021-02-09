import numpy as np
import pandas as pd
import sys
from statsmodels.formula.api import ols

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

alpha=0.75

colors=['#377eb8','#e41a1c','#4daf4a','#984ea3']


##############################################################################################3

print('Loading the processed microdata...')

# load the preprocessed data
df = pd.read_pickle('output/bra_microdata_processed.pik')
df['c']='BRA'

dfs = pd.read_pickle('output/model_microdata_processed.pik')
dfs['c']='BRA'
dfs['drank'] = dfs.groupby(['f','y'])['v']\
                  .transform(lambda x: x.rank(ascending=False))\
                  .astype(int)

pref='_'
altlab=''
dfs2=None
if len(sys.argv)>1 and sys.argv[1]=='sunk':
        pref='_sunkcost'
        altlab='Sunk cost 1'
        dfs2 = pd.read_pickle('output/sunkcost_microdata_processed.pik')
        dfs2['c']='BRA'
        dfs2['drank'] = dfs2.groupby(['f','y'])['v']\
                            .transform(lambda x: x.rank(ascending=False))\
                            .astype(int)

elif len(sys.argv)>1 and sys.argv[1]=='sunk2':
        pref='_sunkcost2'
        altlab='Sunk cost 2'
        dfs2 = pd.read_pickle('output/sunkcost2_microdata_processed.pik')
        dfs2['c']='BRA'
        dfs2['drank'] = dfs2.groupby(['f','y'])['v']\
                            .transform(lambda x: x.rank(ascending=False))\
                            .astype(int)

elif len(sys.argv)>1 and sys.argv[1]=='acr':
        pref='_acr'
        altlab='Exog. entrant dyn.'
        dfs2 = pd.read_pickle('output/sunkcost2_microdata_processed.pik')
        dfs2['c']='BRA'
        dfs2['drank'] = dfs2.groupby(['f','y'])['v']\
                            .transform(lambda x: x.rank(ascending=False))\
                            .astype(int)




##############################################################################################3

print('Distribution of exports/exporters by number of destinations...')

# data
d_agg_by_f = df.groupby(['c','f','y','nd_group'])['v'].sum().reset_index()

d_exporters_by_nd = d_agg_by_f.groupby(['c','y','nd_group']).size().reset_index()
d_exporters_by_nd = d_exporters_by_nd.rename(columns={0:'nf'})
d_exporters_by_nd['frac']=d_exporters_by_nd.groupby(['c','y'])['nf'].transform(lambda x:x/x.sum())
d_exporters_by_nd = d_exporters_by_nd.sort_values(by=['c','y','nd_group']).reset_index()
d_exporters_by_nd = d_exporters_by_nd.groupby(['c','nd_group'])['frac'].mean().reset_index()

d_exports_by_nd = d_agg_by_f.groupby(['c','y','nd_group'])['v'].agg([np.sum,np.mean]).reset_index()
d_exports_by_nd = d_exports_by_nd.rename(columns={'sum':'tot_exports','mean':'avg_exports'})
d_exports_by_nd['frac'] = d_exports_by_nd.groupby(['c','y'])['tot_exports'].transform(lambda x:x/x.sum())
d_exports_by_nd['avg_norm'] = d_exports_by_nd.groupby(['c','y'])['avg_exports'].transform(lambda x:x/x.iloc[0])
d_exports_by_nd = d_exports_by_nd.groupby(['c','nd_group'])[['frac','avg_norm']].mean().reset_index()


# model
d_agg_by_f_s = dfs.groupby(['c','f','y','nd_group'])['v'].sum().reset_index()

d_exporters_by_nd_s = d_agg_by_f_s.groupby(['c','y','nd_group']).size().reset_index()
d_exporters_by_nd_s = d_exporters_by_nd_s.rename(columns={0:'nf'})
d_exporters_by_nd_s['frac']=d_exporters_by_nd_s.groupby(['c','y'])['nf'].transform(lambda x:x/x.sum())
d_exporters_by_nd_s = d_exporters_by_nd_s.sort_values(by=['c','y','nd_group']).reset_index()
d_exporters_by_nd_s = d_exporters_by_nd_s.groupby(['c','nd_group'])['frac'].mean().reset_index()

d_exports_by_nd_s = d_agg_by_f_s.groupby(['c','y','nd_group'])['v'].agg([np.sum,np.mean]).reset_index()
d_exports_by_nd_s= d_exports_by_nd_s.rename(columns={'sum':'tot_exports','mean':'avg_exports'})
d_exports_by_nd_s['frac'] = d_exports_by_nd_s.groupby(['c','y'])['tot_exports'].transform(lambda x:x/x.sum())
d_exports_by_nd_s['avg_norm'] = d_exports_by_nd_s.groupby(['c','y'])['avg_exports'].transform(lambda x:x/x.iloc[0])
d_exports_by_nd_s = d_exports_by_nd_s.groupby(['c','nd_group'])[['frac','avg_norm']].mean().reset_index()

if dfs2 is not None:
    d_agg_by_f_s2 = dfs2.groupby(['c','f','y','nd_group'])['v'].sum().reset_index()

    d_exporters_by_nd_s2 = d_agg_by_f_s2.groupby(['c','y','nd_group']).size().reset_index()
    d_exporters_by_nd_s2 = d_exporters_by_nd_s2.rename(columns={0:'nf'})
    d_exporters_by_nd_s2['frac']=d_exporters_by_nd_s2.groupby(['c','y'])['nf'].transform(lambda x:x/x.sum())
    d_exporters_by_nd_s2 = d_exporters_by_nd_s2.sort_values(by=['c','y','nd_group']).reset_index()
    d_exporters_by_nd_s2 = d_exporters_by_nd_s2.groupby(['c','nd_group'])['frac'].mean().reset_index()

    d_exports_by_nd_s2 = d_agg_by_f_s2.groupby(['c','y','nd_group'])['v'].agg([np.sum,np.mean]).reset_index()
    d_exports_by_nd_s2= d_exports_by_nd_s2.rename(columns={'sum':'tot_exports','mean':'avg_exports'})
    d_exports_by_nd_s2['frac'] = d_exports_by_nd_s2.groupby(['c','y'])['tot_exports'].transform(lambda x:x/x.sum())
    d_exports_by_nd_s2['avg_norm'] = d_exports_by_nd_s2.groupby(['c','y'])['avg_exports'].transform(lambda x:x/x.iloc[0])
    d_exports_by_nd_s2 = d_exports_by_nd_s2.groupby(['c','nd_group'])[['frac','avg_norm']].mean().reset_index()


x=[i+1 for i in range(len(d_exporters_by_nd.nd_group.unique()))]
xl=['1','2','3','4','5-9','10+']
x1=[xx-0.2 for xx in x]
x2 = [xx+.2 for xx in x]

fig,axes=plt.subplots(2,1,figsize=(3,5),sharex=True,sharey=True)

axes[0].bar(x,d_exporters_by_nd.frac,align='center',color=colors[0])
axes[0].set_title('(a) Exporters')
axes[0].set_xticks(x)
axes[0].set_xticklabels(xl)
axes[0].set_xlim(0,len(xl)+1)
#axes[0].set_xlabel('Number of destinations')

axes[1].bar(x,d_exports_by_nd.frac,align='center',color=colors[0])
axes[1].set_title('(b) Exports')
axes[1].set_xticks(x)
axes[1].set_xticklabels(xl)
axes[1].set_xlim(0,len(xl)+1)
axes[1].set_xlabel('Number of destinations')


fig.subplots_adjust(hspace=0.2,wspace=0.0)
plt.savefig('output/dist_by_nd_data_only.pdf',bbox='tight')
plt.clf()
plt.close()


fig,axes=plt.subplots(2,1,figsize=(3,5),sharex=True,sharey=True)

axes[0].bar(x,d_exporters_by_nd.frac,align='edge',width=-0.4,color=colors[0],label='Data')
axes[0].bar(x,d_exporters_by_nd_s.frac,align='edge',width=0.4,color=colors[1],label='Model')
axes[0].set_title('(a) Exporters')
axes[0].set_xticks(x)
axes[0].set_xticklabels(xl)
axes[0].set_xlim(0,len(xl)+1)
#axes[0].set_xlabel('Number of destinations')
axes[0].legend(loc='best',prop={'size':6})

axes[1].bar(x,d_exports_by_nd.frac,align='edge', width=-0.4,color=colors[0])
axes[1].bar(x,d_exports_by_nd_s.frac,align='edge',width=0.4,color=colors[1])
axes[1].set_title('(b) Exports')
axes[1].set_xticks(x)
axes[1].set_xticklabels(xl)
axes[1].set_xlim(0,len(xl)+1)
axes[1].set_xlabel('Number of destinations')

fig.subplots_adjust(hspace=0.2,wspace=0.0)
plt.savefig('output/dist_by_nd_model_vs_data.pdf',bbox='tight')
plt.clf()
plt.close()


if dfs2 is not None:
    fig,axes=plt.subplots(2,1,figsize=(3,6.5),sharex=True,sharey=True)

    axes[0].bar(x1,d_exporters_by_nd.frac,align='center',width=0.2,color=colors[0],label='Data')
    axes[0].bar(x,d_exporters_by_nd_s.frac,align='center',width=0.2,color=colors[1],label='Model')
    axes[0].bar(x2,d_exporters_by_nd_s2.frac,align='center',width=0.2,color=colors[2],label=altlab)
 
    axes[0].set_title('(a) Exporters')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(xl)
    axes[0].set_xlim(0,len(xl)+1)
#    axes[0].set_xlabel('Number of destinations')
    axes[0].legend(loc='best',prop={'size':6})
    
    axes[1].bar(x1,d_exports_by_nd.frac,align='center', width=0.2,color=colors[0])
    axes[1].bar(x,d_exports_by_nd_s.frac,align='center',width=0.2,color=colors[1])
    axes[1].bar(x2,d_exports_by_nd_s2.frac,align='center',width=0.2,color=colors[2])
    axes[1].set_title('(b) Exports')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(xl)
    axes[1].set_xlim(0,len(xl)+1)
    axes[1].set_xlabel('Number of destinations')

    fig.subplots_adjust(hspace=0.2,wspace=0.0)
    plt.savefig('output/dist_by_nd_model_vs_data'+pref+'.pdf',bbox='tight')
    plt.clf()
    plt.close()




##############################################################################################3

print('Exit rates by nd group-drank...')

d_exit_by_nd_drank = df.groupby(['c','y','nd_group','drank'])['exit'].agg(lambda x: ((float)(x.sum()))/x.count()).reset_index()

d_exit_by_nd_drank = d_exit_by_nd_drank.groupby(['c','nd_group','drank']).mean().reset_index()


d_exit_by_nd_drank_s = dfs.groupby(['c','y','nd_group','drank'])['exit'].agg(lambda x: ((float)(x.sum()))/x.count()).reset_index()

d_exit_by_nd_drank_s = d_exit_by_nd_drank_s.groupby(['c','nd_group','drank']).mean().reset_index()

d_exit_by_nd_drank_s2=None
if dfs2 is not None:
    d_exit_by_nd_drank_s2 = dfs2.groupby(['c','y','nd_group','drank'])['exit'].agg(lambda x: ((float)(x.sum()))/x.count()).reset_index()

    d_exit_by_nd_drank_s2 = d_exit_by_nd_drank_s2.groupby(['c','nd_group','drank']).mean().reset_index()


nd=[int(n) for n in d_exit_by_nd_drank.nd_group.unique().tolist()]
nds = ['1','2','3','4','5-9','10+']
cc = list(d_exit_by_nd_drank.c.unique())

# latex output
file = open('output/exit_by_nd_drank'+pref+'.tex','w')

# header
file.write('\\begin{table}[p]\n')
file.write('\\footnotesize\n')
file.write('\\renewcommand{\\arraystretch}{1.2}\n')
file.write('\\begin{center}\n')
file.write('\\begin{threeparttable}')
file.write('\\caption{Exit rates by num. dest. and dest. rank}\n')
file.write('\\label{tab:exit_by_nd_drank}\n')
file.write('\\begin{tabular}{l')
for i in nd:
    file.write('c')
file.write('}\n')
file.write('\\toprule\n')
        

file.write('&\\multicolumn{%d}{c}{Destination rank}\\\\\n'%(len(nd)))
file.write('\\cmidrule(rl){2-%d}\n'%(len(nd)+1))
        
file.write('Num. dest.')
for s in nds:
    file.write('&'+s)
file.write('\\\\\n')
file.write('\\midrule\n')

file.write('\\multicolumn{%d}{l}{\\textit{(a) Data}}\\\\\n'%(len(nd)+1))
for i in range(len(nd)):
    n=nd[i]
    file.write(nds[i])

    tmp=d_exit_by_nd_drank[d_exit_by_nd_drank.c=='BRA']

    for j in range(len(nd)):
        m = nd[j]
        tmps='-'
        if(m<=n):
            tmpv = tmp['exit'][np.logical_and(tmp.nd_group==n,
                                              tmp.drank==m)].values[0]
            tmps = '%0.2f' % tmpv
        file.write('& '+ tmps)
            
    file.write('\\\\\n')

file.write('\\\\\n\\multicolumn{%d}{l}{\\textit{(b) Model}}\\\\\n'%(len(nd)+1))

for i in range(len(nd)):
    n=nd[i]
    file.write(nds[i])

    tmp=d_exit_by_nd_drank_s[d_exit_by_nd_drank_s.c=='BRA']
    
    for j in range(len(nd)):
        m = nd[j]
        tmps='-'
        if(m<=n):
            tmpv = tmp['exit'][np.logical_and(tmp.nd_group==n,
                                              tmp.drank==m)].values[0]
            tmps = '%0.2f' % tmpv
        file.write('& '+ tmps)
            
    file.write('\\\\\n')

if dfs2 is not None:
    file.write('\\\\\n\\multicolumn{%d}{l}{\\textit{(c) %s}}\\\\\n'%((len(nd)+1),altlab))

    for i in range(len(nd)):
        n=nd[i]
        file.write(nds[i])
        
        tmp=d_exit_by_nd_drank_s2[d_exit_by_nd_drank_s2.c=='BRA']
    
        for j in range(len(nd)):
            m = nd[j]
            tmps='-'
            if(m<=n):
                tmpv = tmp['exit'][np.logical_and(tmp.nd_group==n,
                                                  tmp.drank==m)].values[0]
                tmps = '%0.2f' % tmpv
                file.write('& '+ tmps)
            
        file.write('\\\\\n')

    
#file.write('\\\\\n')

# footer
file.write('\\bottomrule\n')
file.write('\\end{tabular}\n')
file.write('\\begin{tablenotes}\n')
file.write("\\item Source: SECEX and author's calculations.")
file.write('\\end{tablenotes}\n')
file.write('\\end{threeparttable}\n')
file.write('\\end{center}\n')
file.write('\\normalsize\n')
file.write('\\end{table}\n')
