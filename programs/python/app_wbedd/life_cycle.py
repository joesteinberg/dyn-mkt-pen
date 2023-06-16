import os
import numpy as np
import pandas as pd
import sys
from statsmodels.api import OLS
from statsmodels.formula.api import ols
import patsy
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
mpl.rc('lines',linewidth=0.5)
mpl.rcParams.update({'errorbar.capsize': 2})
alpha=0.8
colors=['#377eb8','#e41a1c','#4daf4a','#984ea3','#ff7f00','#ffff33']
fmts = ['o','s','D','X','P']

dopath = '/home/linux/Documents/Research/dyn_mkt_pen/v3/programs/python/wbedd/'
outpath = '/home/linux/Documents/Research/dyn_mkt_pen/v3/programs/python/wbedd/output/'
figpath = '/home/linux/Documents/Research/dyn_mkt_pen/v3/programs/python/wbedd/output/'

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


max_tenure_scalar=4

#############################################################################

print('\tLoading the processed microdata...')

# load the preprocessed data
df = pd.read_pickle(outpath + 'wbedd_microdata_processed.pik')
#df=df.loc[~((df.c=='MEX') & (df.d=='USA')),:].reset_index(drop=True)

df['nf'] = df.groupby(['c','d','y'])['f'].transform(lambda x: x.nunique())
tmp2 = df.groupby(['c','d'])['nf'].mean().reset_index()
lo = tmp2.groupby('c')['nf'].agg(lambda x: x.quantile(0.5)).reset_index().rename(columns={'nf':'lo'})
hi = tmp2.groupby('c')['nf'].agg(lambda x: x.quantile(0.5)).reset_index().rename(columns={'nf':'hi'})
df = pd.merge(left=df,right=lo,how='left',on='c')
df = pd.merge(left=df,right=hi,how='left',on='c')

df['grp'] = np.nan
df.loc[df.nf<df.lo,'grp']=0
df.loc[df.nf>df.hi,'grp']=1
df=df[df.tenure.notnull()].reset_index(drop=True)
df.loc[df.tenure>max_tenure_scalar,'tenure']=max_tenure_scalar
df.loc[df.max_tenure>max_tenure_scalar,'max_tenure']=max_tenure_scalar

df_m = df.loc[df.c=='MEX',:].reset_index(drop=True)
df_p = df.loc[df.c=='PER',:].reset_index(drop=True)

df_m.rename(columns={'exit':'xit'}).to_stata(outpath + 'mex_microdata_processed.dta')
df_p.rename(columns={'exit':'xit'}).to_stata(outpath + 'per_microdata_processed.dta')

#############################################################################

print('\tEstimating tenure effect regressions on actual data...')

os.system('stata -b ' + dopath + 'life_cycle_data.do')


#############################################################################
print('Processing exit regression results...')

mreg_x_all = pd.read_stata(outpath + 'mreg_x_all.dta').set_index('var')
mreg_x_2way = pd.read_stata(outpath + 'mreg_x_2way.dta').set_index('var')

preg_x_all = pd.read_stata(outpath + 'preg_x_all.dta').set_index('var')
preg_x_2way = pd.read_stata(outpath + 'preg_x_2way.dta').set_index('var')

grps = ['0b.grp','1.grp']

meffect_x_all = np.zeros((max_tenure_scalar+1,2))
peffect_x_all = np.zeros((max_tenure_scalar+1,2))
for j in range(0,max_tenure_scalar+1):
    strx = ''
    if j==0:
        strx = '0b.tenure'
    else:
        strx = str(j)+'.tenure'

    meffect_x_all[j,0] = mreg_x_all.coef[strx]
    meffect_x_all[j,1] = mreg_x_all.ci_upper[strx] - mreg_x_all.ci_lower[strx]
    peffect_x_all[j,0] = preg_x_all.coef[strx]
    peffect_x_all[j,1] = preg_x_all.ci_upper[strx] - preg_x_all.ci_lower[strx]

meffect_x_2way = np.zeros((2,max_tenure_scalar+1,2))
peffect_x_2way = np.zeros((2,max_tenure_scalar+1,2))
for i in range(2):
    for j in range(0,max_tenure_scalar+1):

        strx = ''
        if j==0:
            strx = '0b.tenure#'+grps[i]
        else:
            strx = str(j)+'.tenure#'+grps[i]
              
        meffect_x_2way[i,j,0] = mreg_x_2way.coef[strx]
        meffect_x_2way[i,j,1] = mreg_x_2way.ci_upper[strx] - mreg_x_2way.ci_lower[strx]
        peffect_x_2way[i,j,0] = preg_x_2way.coef[strx]
        peffect_x_2way[i,j,1] = preg_x_2way.ci_upper[strx] - preg_x_2way.ci_lower[strx]
  

#############################################################################
        
print('\tMaking plots of tenure effects on exit...')

sz=(6,3)
tenure = range(1,max_tenure_scalar+2)

fig,axes=plt.subplots(1,2,figsize=sz,sharex=True,sharey=False)

ax = axes[0]
ax.set_title('(a) All markets',y=1.025)
ax.plot(tenure,meffect_x_all[:,0],
        color=colors[0],
        alpha=0.5,
        marker='o',
        linewidth=1,
        markersize=3,
        linestyle='-',
        label='MEX')
ax.plot(tenure,peffect_x_all[:,0],
        color=colors[1],
        alpha=0.5,
        marker='s',
        linewidth=1,
        markersize=3,
        linestyle='-',
        label='MEX')
ax.set_xticks(range(1,max_tenure_scalar+2))
ax.set_xlabel('Years in market')

ax=axes[1]
ax.set_title('(b) Hard vs. easy markets',y=1.025)
ax.plot(tenure,meffect_x_2way[0,:,0],
        color=colors[0],
        alpha=0.5,
        marker='o',
        linewidth=1,
        markersize=3,
        linestyle='-',
        label='Hard markets (MEX)')
ax.plot(tenure,meffect_x_2way[1,:,0],
        color=colors[2],
        marker='x',
        alpha=0.5,
        markersize=3,
        linewidth=1,
        linestyle='--',
        label='Easy markets (MEX)')
ax.plot(tenure,peffect_x_2way[0,:,0],
        color=colors[1],
        alpha=0.5,
        marker='s',
        linewidth=1,
        markersize=3,
        linestyle='--',
        label='Hard markets (PER)')
ax.plot(tenure,peffect_x_2way[1,:,0],
        color=colors[3],
        marker='+',
        alpha=0.5,
        markersize=3,
        linewidth=1,
        linestyle='--',
        label='Easy markets (PER)')
ax.set_xticks(range(1,max_tenure_scalar+2))
ax.set_xlabel('Years in market')      
ax.legend(loc='upper right',prop={'size':6})

axes[0].set_ylabel('Conditional exit rate (0 at entry)')
fig.subplots_adjust(hspace=0.2,wspace=0.2)
plt.savefig(figpath + 'life_cycle_dyn_x_wbedd.pdf',bbox_inches='tight')

plt.close('all')

################################################################################
print('Processing sales regression results...')

mreg_v_all = pd.read_stata(outpath + 'mreg_v_all.dta').set_index('var')
mreg_v_3way = pd.read_stata(outpath + 'mreg_v_3way.dta').set_index('var')
preg_v_all = pd.read_stata(outpath + 'preg_v_all.dta').set_index('var')
preg_v_3way = pd.read_stata(outpath + 'preg_v_3way.dta').set_index('var')

meffect_v_all = np.zeros((max_tenure_scalar+1,max_tenure_scalar+1,2))
peffect_v_all = np.zeros((max_tenure_scalar+1,max_tenure_scalar+1,2))
for k in range(1,max_tenure_scalar+1):
    for j in range(k+1):

        strx=''
        if j==0:
            strx = '0b.tenure#'+str(k)+'.max_tenure'
        else:
            strx = str(j)+'.tenure#'+str(k)+'.max_tenure'
            
        meffect_v_all[k,j,0] = mreg_v_all.coef[strx]
        meffect_v_all[k,j,1] = mreg_v_all.ci_upper[strx] - mreg_v_all.ci_lower[strx]
        peffect_v_all[k,j,0] = preg_v_all.coef[strx]
        peffect_v_all[k,j,1] = preg_v_all.ci_upper[strx] - preg_v_all.ci_lower[strx]
 

meffect_v_3way = np.zeros((2,max_tenure_scalar+1,max_tenure_scalar+1,2))
peffect_v_3way = np.zeros((2,max_tenure_scalar+1,max_tenure_scalar+1,2))
for i in range(2):
    for k in range(1,max_tenure_scalar+1):
        for j in range(k+1):

            strx=''
            if j==0:
                strx = '0b.tenure#'+str(k)+'.max_tenure#'+grps[i]
            else:
                strx = str(j)+'.tenure#'+str(k)+'.max_tenure#'+grps[i]
             
            meffect_v_3way[i,k,j,0] = mreg_v_3way.coef[strx]
            meffect_v_3way[i,k,j,1] = mreg_v_3way.ci_upper[strx] - mreg_v_3way.ci_lower[strx]
            peffect_v_3way[i,k,j,0] = preg_v_3way.coef[strx]
            peffect_v_3way[i,k,j,1] = preg_v_3way.ci_upper[strx] - preg_v_3way.ci_lower[strx]
 

#############################################################################
        
print('\tMaking plots of tenure#max_tenure effects on sales...')

sz = (7.5,3)

fig,axes=plt.subplots(1,3,figsize=sz,sharex=True,sharey=True)

for k in range(1,max_tenure_scalar+1):

    tenure = [x+1 for x in range(k+1)]

    axes[0].plot(tenure,meffect_v_all[k,:k+1,0],
                 color=colors[k-1],
                 alpha=0.5,
                 marker='o',
                 linewidth=1,
                 markersize=3,
                 #capsize=3,
                 linestyle='-',
                 label='Duration = %d (MEX)'%(k+1))

    axes[1].plot(tenure,meffect_v_3way[0,k,:k+1,0],
                 color=colors[k-1],
                 marker='o',
                 alpha=0.5,
                 markersize=3,
                 #capsize=3,
                 linewidth=1,
                 linestyle='-',
                 label='Duration = %d (MEX)'%(k+1))

    axes[2].plot(tenure,meffect_v_3way[1,k,:k+1,0],
                 color=colors[k-1],
                 marker='o',
                 alpha=0.5,
                 markersize=3,
                 #capsize=3,
                 linewidth=1,
                 linestyle='-',
                 label='Duration = %d (MEX)'%(k+1))

for k in range(1,max_tenure_scalar+1):

    tenure = [x+1 for x in range(k+1)]
    
    axes[0].plot(tenure,peffect_v_all[k,:k+1,0],
                 color=colors[k-1],
                 alpha=0.5,
                 marker='s',
                 linewidth=1,
                 markersize=3,
                 #capsize=3,
                 linestyle='--',
                 label='Duration = %d (PER)'%(k+1))

    axes[1].plot(tenure,peffect_v_3way[0,k,:k+1,0],
                 color=colors[k-1],
                 marker='s',
                 alpha=0.5,
                 markersize=3,
                 #capsize=3,
                 linewidth=1,
                 linestyle='--',
                 label='Duration = %d (PER)'%(k+1))    

    axes[2].plot(tenure,peffect_v_3way[1,k,:k+1,0],
                 color=colors[k-1],
                 marker='s',
                 alpha=0.5,
                 markersize=3,
                 #capsize=3,
                 linewidth=1,
                 linestyle='--',
                 label='Duration = %d (PER)'%(k+1))


tablelegend(axes[0], ncol=2, loc='upper left',#bbox_to_anchor=(0.063,0.62), 
            row_labels=['2', '3', '4', '5'], 
            col_labels=['MEX','PER'], 
            title_label='Dur.')
    

    
#axes1[0].set_ylim(-0.25,3)
#axes1[1].set_ylim(-0.25,3)
#axes[0].legend(loc='lower right',prop={'size':6})
axes[0].set_xticks(range(1,max_tenure_scalar+2))
axes[1].set_xticks(range(1,max_tenure_scalar+2))
axes[2].set_xticks(range(1,max_tenure_scalar+2))
axes[0].set_xlabel('Years in market')
axes[1].set_xlabel('Years in market')
axes[2].set_xlabel('Years in market')
axes[0].set_title('(a) All markets',y=1.025)
axes[1].set_title('(b) Hard markets',y=1.025)
axes[2].set_title('(b) Easy markets',y=1.025)

#axes1[1].set_yticks([])
axes[0].set_ylabel('log exports (relative to duration = 1)')


fig.subplots_adjust(hspace=0.15,wspace=0.1)
plt.sca(axes[0])
plt.savefig(figpath + 'life_cycle_dyn_v_wbedd.pdf',bbox_inches='tight')
plt.close('all')
