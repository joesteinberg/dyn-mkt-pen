import os
import numpy as np
import pandas as pd
import sys
import patsy
from statsmodels.api import OLS
import statsmodels.api as sm
from statsmodels.formula.api import ols, glm

import locale
locale.setlocale(locale.LC_ALL,'en_US.utf8')

import matplotlib as mpl
import matplotlib.legend as mlegend
from matplotlib.patches import Rectangle
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator

mpl.rc('text', usetex=True)
mpl.rc('savefig',format='pdf')
mpl.rc('font',**{'family':'serif','serif':['Palatino'],'size':8})
mpl.rc('font',size=8)
mpl.rc('lines',linewidth=1)

alpha=0.75

colors=['#377eb8','#e41a1c','#4daf4a','#984ea3','#ff7f00','#ffff33']

def tablelegend(ax, col_labels=None, row_labels=None, title_label="", *args, **kwargs):
    """
    Place a table legend on the axes.
    
    Creates a legend where the labels are not directly placed with the artists, 
    but are used as row and column headers, looking like this:
    
    title_label   | col_labels[1] | col_labels[2] | col_labels[3]
    -------------------------------------------------------------
    row_labels[1] |
    row_labels[2] |              <artists go there>
    row_labels[3] |
    
    
    Parameters
    ----------
    
    ax : `matplotlib.axes.Axes`
        The artist that contains the legend table, i.e. current axes instant.
        
    col_labels : list of str, optional
        A list of labels to be used as column headers in the legend table.
        `len(col_labels)` needs to match `ncol`.
        
    row_labels : list of str, optional
        A list of labels to be used as row headers in the legend table.
        `len(row_labels)` needs to match `len(handles) // ncol`.
        
    title_label : str, optional
        Label for the top left corner in the legend table.
        
    ncol : int
        Number of columns.
        

    Other Parameters
    ----------------
    
    Refer to `matplotlib.legend.Legend` for other parameters.
    
    """
    #################### same as `matplotlib.axes.Axes.legend` #####################
    handles, labels, extra_args, kwargs = mlegend._parse_legend_args([ax], *args, **kwargs)
    if len(extra_args):
        raise TypeError('legend only accepts two non-keyword arguments')
    
    if col_labels is None and row_labels is None:
        ax.legend_ = mlegend.Legend(ax, handles, labels, **kwargs)
        ax.legend_._remove_method = ax._remove_legend
        return ax.legend_
    #################### modifications for table legend ############################
    else:
        ncol = kwargs.pop('ncol')
        handletextpad = kwargs.pop('handletextpad', 0 if col_labels is None else -2)
        title_label = [title_label]
        
        # blank rectangle handle
        extra = [Rectangle((0, 0), 1, 1, fc="w", fill=False, edgecolor='none', linewidth=0)]
        
        # empty label
        empty = [""]
        
        # number of rows infered from number of handles and desired number of columns
        nrow = len(handles) // ncol
        
        # organise the list of handles and labels for table construction
        if col_labels is None:
            assert nrow == len(row_labels), "nrow = len(handles) // ncol = %s, but should be equal to len(row_labels) = %s." % (nrow, len(row_labels))
            leg_handles = extra * nrow
            leg_labels  = row_labels
        elif row_labels is None:
            assert ncol == len(col_labels), "ncol = %s, but should be equal to len(col_labels) = %s." % (ncol, len(col_labels))
            leg_handles = []
            leg_labels  = []
        else:
            assert nrow == len(row_labels), "nrow = len(handles) // ncol = %s, but should be equal to len(row_labels) = %s." % (nrow, len(row_labels))
            assert ncol == len(col_labels), "ncol = %s, but should be equal to len(col_labels) = %s." % (ncol, len(col_labels))
            leg_handles = extra + extra * nrow
            leg_labels  = title_label + row_labels
        for col in range(ncol):
            if col_labels is not None:
                leg_handles += extra
                leg_labels  += [col_labels[col]]
            leg_handles += handles[col*nrow:(col+1)*nrow]
            leg_labels  += empty * nrow
        
        # Create legend
        ax.legend_ = mlegend.Legend(ax, leg_handles, leg_labels, ncol=ncol+int(row_labels is not None), handletextpad=handletextpad, handlelength=2.25, columnspacing=0.8, prop={'size':6},**kwargs)
        ax.legend_._remove_method = ax._remove_legend
        return ax.legend_


##############################################################################################3

print('Loading the processed microdata...')

# load the preprocessed data
df = pd.read_pickle('output/wbedd_microdata_processed.pik')
df_m = df.loc[df.c=='MEX',:].reset_index(drop=True)
df_p = df.loc[df.c=='PER',:].reset_index(drop=True)
df_m.rename(columns={'exit':'xit'}).to_stata('output/mex_microdata_processed.dta')
df_p.rename(columns={'exit':'xit'}).to_stata('output/per_microdata_processed.dta')

##############################################################################################3
print('Computing average drank...')

icols = ['c','d','y','popt','gdppc','tau']
agg_by_d_m = df_m.groupby(icols)['drank'].mean().reset_index()
agg_by_d_p = df_p.groupby(icols)['drank'].mean().reset_index()

f='drank ~ np.log(gdppc) + np.log(popt) + np.log(tau) + C(y)'
dregs_m = [ols(formula=f,data=agg_by_d_m).fit(cov_type='HC0')]
dregs_p = [ols(formula=f,data=agg_by_d_p).fit(cov_type='HC0')]


def signf(p):
        if p<0.001:
                return '^\\S'
        elif p<0.01:
                return '^\\ddagger'
        elif p<0.05:
                return '^\\dagger'
        elif p<0.1:
                return '^*'
        else:
                return ''

vars = ['drank']
vnames = ['Avg. rank']
file = open('output/drank_regs_wbedd.tex','w')

# header
#file.write('\\begin{landscape}\n')
file.write('\\begin{table}[p]\n')
file.write('\\footnotesize\n')
#file.write('\\renewcommand{\\arraystretch}{1.2}\n')
file.write('\\begin{center}\n')
file.write("\\caption{Associations between destination characteristics and average rank within exporters' portfolios (Mexico and Peru)}\n")
file.write('\\label{tab:regs}\n')
file.write('\\begin{tabular}{l')
for i in range(len(vars)):
        file.write('c')
file.write('}')
file.write('\\toprule\n')
        
# column names
for vname in vnames:
        file.write('& \\multicolumn{1}{b{1.6cm}}{\\centering '+vname+'}')
file.write('\\\\\n')
file.write('\\midrule\n')        

dregs = dregs_m
file.write('\\multicolumn{%d}{l}{\\textit{(a) Mexico}}\\\\[4pt]\n'%(len(vars)+1))

file.write('log GDPpc')
for r in dregs:
        file.write('& %0.3f' % r.params['np.log(gdppc)'])
file.write('\\\\\n')
for r in dregs:
        file.write('& $(%0.3f)%s$' % (r.HC0_se['np.log(gdppc)'],signf(r.pvalues['np.log(gdppc)'])))
file.write('\\\\[4pt]\n')

file.write('log population')
for r in dregs:
        file.write('& %0.3f' % r.params['np.log(popt)'])
file.write('\\\\\n')
for r in dregs:
        file.write('& $(%0.3f)%s$' % (r.HC0_se['np.log(popt)'],signf(r.pvalues['np.log(popt)'])))
file.write('\\\\[4pt]\n')

file.write('log trade barrier')
for r in dregs:
        file.write('& %0.3f' % r.params['np.log(tau)'])
file.write('\\\\\n')
for r in dregs:
        file.write('& $(%0.3f)%s$' % (r.HC0_se['np.log(tau)'],signf(r.pvalues['np.log(tau)'])))
file.write('\\\\[4pt]\n')

file.write('Num. observations')
for r in dregs:
        file.write('& %s' % "{:,d}".format(int(r.nobs)))
file.write('\\\\\n')

file.write('$R^2$')
for r in dregs:
        file.write('& %0.2f' % r.rsquared)
file.write('\\\\\n\\midrule\n')

dregs = dregs_p
file.write('\\multicolumn{%d}{l}{\\textit{(b) Peru}}\\\\[4pt]\n'%(len(vars)+1))

file.write('log GDPpc')
for r in dregs:
        file.write('& %0.3f' % r.params['np.log(gdppc)'])
file.write('\\\\\n')
for r in dregs:
        file.write('& $(%0.3f)%s$' % (r.HC0_se['np.log(gdppc)'],signf(r.pvalues['np.log(gdppc)'])))
file.write('\\\\[4pt]\n')

file.write('log population')
for r in dregs:
        file.write('& %0.3f' % r.params['np.log(popt)'])
file.write('\\\\\n')
for r in dregs:
        file.write('& $(%0.3f)%s$' % (r.HC0_se['np.log(popt)'],signf(r.pvalues['np.log(popt)'])))
file.write('\\\\[4pt]\n')

file.write('log trade barrier')
for r in dregs:
        file.write('& %0.3f' % r.params['np.log(tau)'])
file.write('\\\\\n')
for r in dregs:
        file.write('& $(%0.3f)%s$' % (r.HC0_se['np.log(tau)'],signf(r.pvalues['np.log(tau)'])))
file.write('\\\\[4pt]\n')

file.write('Num. observations')
for r in dregs:
        file.write('& %s' % "{:,d}".format(int(r.nobs)))
file.write('\\\\\n')

file.write('$R^2$')
for r in dregs:
        file.write('& %0.2f' % r.rsquared)
file.write('\\\\\n')

        
# footer
file.write('\\bottomrule\n')
file.write('\\end{tabular}\n')
file.write('\\end{center}\n')
file.write('\\normalsize\n')
file.write('\\end{table}\n')
#file.write('\\end{landscape}\n')

file.close()

############################################################################################

df.loc[df.drank>=10,'drank']=10
df.loc[df.drank.isin(range(5,10)),'drank'] = 6

df_m = df.loc[df.c=='MEX',:].reset_index(drop=True)
df_p = df.loc[df.c=='PER',:].reset_index(drop=True)

df_m.rename(columns={'exit':'xit'}).to_stata('output/mex_microdata_processed.dta')
df_p.rename(columns={'exit':'xit'}).to_stata('output/per_microdata_processed.dta')
     

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




x=[i+1 for i in range(len(d_exporters_by_nd.nd_group.unique()))]
xl=['1','2','3','4','5-9','10+']
x1=[xx-0.2 for xx in x]
x2 = [xx+.2 for xx in x]


fig,axes=plt.subplots(1,2,figsize=(6.5,3.5),sharex=True,sharey=True)

axes[0].bar(x,d_exporters_by_nd.loc[d_exporters_by_nd.c=='MEX','frac'],align='edge',width=-0.4,color=colors[0],label='MEX')
axes[0].bar(x,d_exporters_by_nd.loc[d_exporters_by_nd.c=='PER','frac'],align='edge',width=0.4,color=colors[1],label='PER')
axes[0].set_title('(a) Exporters')
axes[0].set_xticks(x)
axes[0].set_xticklabels(xl)
axes[0].set_xlim(0,len(xl)+1)
#axes[0].set_xlabel('Number of destinations')
axes[0].legend(loc='best',prop={'size':6})

axes[1].bar(x,d_exports_by_nd.loc[d_exports_by_nd.c=='MEX','frac'],align='edge', width=-0.4,color=colors[0])
axes[1].bar(x,d_exports_by_nd.loc[d_exports_by_nd.c=='PER','frac'],align='edge',width=0.4,color=colors[1])
axes[1].set_title('(b) Exports')
axes[1].set_xticks(x)
axes[1].set_xticklabels(xl)
axes[1].set_xlim(0,len(xl)+1)
axes[0].set_xlabel('Number of destinations')
axes[1].set_xlabel('Number of destinations')

fig.subplots_adjust(hspace=0.2,wspace=0.15)
plt.savefig('output/dist_by_nd_wbedd.pdf',bbox_inches='tight')
plt.clf()
plt.close()

##############################################################################################3

print('Exit rates by nd group-drank...')

d_exit_by_nd_drank = df.groupby(['c','y','nd_group','drank'])['exit'].agg(lambda x: ((float)(x.sum()))/x.count()).reset_index()

d_exit_by_nd_drank = d_exit_by_nd_drank.groupby(['c','nd_group','drank']).mean().reset_index()


nd=[int(n) for n in d_exit_by_nd_drank.nd_group.unique().tolist()]
nds = ['1','2','3','4','5-9','10+']
cc = list(d_exit_by_nd_drank.c.unique())

# latex output
file = open('output/exit_by_nd_drank_wbedd.tex','w')

# header
file.write('\\begin{table}[p]\n')
file.write('\\footnotesize\n')
file.write('\\renewcommand{\\arraystretch}{1.2}\n')
file.write('\\begin{center}\n')
file.write('\\caption{Exit rates by num. dest. and dest. rank (Mexico and Peru)}\n')
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

file.write('\\multicolumn{%d}{l}{\\textit{(a) Mexico}}\\\\\n'%(len(nd)+1))
for i in range(len(nd)):
    n=nd[i]
    file.write(nds[i])

    tmp=d_exit_by_nd_drank[d_exit_by_nd_drank.c=='MEX']

    for j in range(len(nd)):
        m = nd[j]
        tmps='-'
        if(m<=n):
            tmpv = tmp['exit'][np.logical_and(tmp.nd_group==n,
                                              tmp.drank==m)].values[0]
            tmps = '%0.2f' % tmpv
        file.write('& '+ tmps)
            
    file.write('\\\\\n')

file.write('\\\\\n\\multicolumn{%d}{l}{\\textit{(b) Peru}}\\\\\n'%(len(nd)+1))

for i in range(len(nd)):
    n=nd[i]
    file.write(nds[i])

    tmp=d_exit_by_nd_drank[d_exit_by_nd_drank.c=='PER']
    
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
file.write('\\end{center}\n')
file.write('\\normalsize\n')
file.write('\\end{table}\n')

file.close()

##############################################################################################
print('Estimating sales by nd_group and drank...')

ngrps = len(df.nd_group.unique())
grps = df.nd_group.unique()
x=df.drank.unique()
x.sort()
drank = list(x)

deffect_v_m = np.zeros((ngrps,ngrps))
deffect_x_m = np.zeros((ngrps,ngrps))
deffect_v_p = np.zeros((ngrps,ngrps))
deffect_x_p = np.zeros((ngrps,ngrps))

f1 = 'np.log(v) ~ C(d) + C(cohort) + C(drank):C(nd_group)'
f2 = 'exit ~ C(d) + C(cohort) + C(drank):C(nd_group)'

y,X = patsy.dmatrices(f1, df_m, return_type='dataframe')
for c in X.columns:
        if(X[c].sum()<1.0e-10):
                X.drop(c,axis=1,inplace=True)
dreg_v_m = OLS(y,X).fit(cov_type='HC0')

y,X = patsy.dmatrices(f2, df_m, return_type='dataframe')
for c in X.columns:
        if(X[c].sum()<1.0e-10):
                X.drop(c,axis=1,inplace=True)
dreg_x_m = OLS(y,X).fit(cov_type='HC0')



y,X = patsy.dmatrices(f1, df_p, return_type='dataframe')
for c in X.columns:
        if(X[c].sum()<1.0e-10):
                X.drop(c,axis=1,inplace=True)
dreg_v_p = OLS(y,X).fit(cov_type='HC0')

y,X = patsy.dmatrices(f2, df_p, return_type='dataframe')
for c in X.columns:
        if(X[c].sum()<1.0e-10):
                X.drop(c,axis=1,inplace=True)
dreg_x_p = OLS(y,X).fit(cov_type='HC0')

for k in range(1,ngrps):
        for j in range(k+1):

                deffect_v_m[k,j] = dreg_v_m.params["C(nd_group)[T.%d]"%(drank[k])]
                deffect_x_m[k,j] = dreg_x_m.params["C(nd_group)[T.%d]"%(drank[k])]
                
                deffect_v_p[k,j] = dreg_v_p.params["C(nd_group)[T.%d]"%(drank[k])]
                deffect_x_p[k,j] = dreg_x_p.params["C(nd_group)[T.%d]"%(drank[k])]
                
                if(j>0):
                        deffect_v_m[k,j] += dreg_v_m.params["C(drank)[T.%d]:C(nd_group)[%d]"%(drank[j],drank[k])]
                        deffect_x_m[k,j] += dreg_x_m.params["C(drank)[T.%d]:C(nd_group)[%d]"%(drank[j],drank[k])]
                        
                        deffect_v_p[k,j] += dreg_v_p.params["C(drank)[T.%d]:C(nd_group)[%d]"%(drank[j],drank[k])]
                        deffect_x_p[k,j] += dreg_x_p.params["C(drank)[T.%d]:C(nd_group)[%d]"%(drank[j],drank[k])]

print('\tMaking plots...')



fig1,axes1=plt.subplots(1,2,figsize=(7,3),sharex=True,sharey=False)
               
axes1[0].set_title('(a) Sales (log)',y=1.025)
axes1[1].set_title('(b) Exit rate',y=1.025)

lns=[]
for k in range(1,ngrps):
        
        drank = [x+1 for x in range(k+1)]
    
        l=axes1[0].plot(drank,deffect_v_m[k,:k+1],
                        color=colors[k-1],
                        alpha=0.5,
                        marker='o',
                        linewidth=1,
                        markersize=3,
                        #capsize=3,
                        linestyle='-',
                        label='Num. dests. = %s (data)'%(xl[k]))
        lns.append(l)

        l=axes1[1].plot(drank,deffect_x_m[k,:k+1],
                        color=colors[k-1],
                        marker='o',
                        alpha=0.5,
                        markersize=3,
                        #capsize=3,
                        linewidth=1,
                        linestyle='-',
                        label='Num. dests. = %s (data)'%(xl[k]))
        lns.append(l)
        
for k in range(1,ngrps):
                
        drank = [x+1 for x in range(k+1)]
    
        l=axes1[0].plot(drank,deffect_v_p[k,:k+1],
                        color=colors[k-1],
                        alpha=0.5,
                        marker='s',
                        linewidth=1,
                        markersize=3,
                        #capsize=3,
                        linestyle='--',
                        label='Num dests. = %s (model)'%(xl[k]))
        lns.append(l)


        l=axes1[1].plot(drank,deffect_x_p[k,:k+1],
                        color=colors[k-1],
                        marker='s',
                        alpha=0.5,
                        markersize=3,
                        #capsize=3,
                        linewidth=1,
                        linestyle='--',
                        label='Num dests. = %s (model)'%(xl[k]))
        lns.append(l)
        

tablelegend(axes1[0], ncol=2, loc='upper right',#bbox_to_anchor=(0.063,0.62), 
            row_labels=['2', '3', '4', '6-9', '10+'], 
            col_labels=['MEX','PER'], 
            title_label='Num. dests.')


axes1[0].set_xlabel('Destination rank')
axes1[1].set_xlabel('Destination rank')
#axes1[0].set_ylim(-0.5,3)
#axes1[1].set_ylim(-0.5,3)
axes1[0].set_ylabel('Single-dest. exporters = 0')
axes1[0].set_xticks(range(1,ngrps+1))
axes1[1].set_xticks(range(1,ngrps+1))
axes1[0].set_xticklabels(xl)
axes1[1].set_xticklabels(xl)

fig1.subplots_adjust(hspace=0.1,wspace=0.2)

plt.sca(axes1[0])
plt.savefig('output/by_nd_drank_wbedd.pdf',bbox_inches='tight')

plt.close('all')
