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
mpl.rc('lines',linewidth=1)


alpha=0.8
colors=['#377eb8','#e41a1c','#4daf4a','#984ea3','#ff7f00','#ffff33']
fmts = ['o','s','D','X','P']

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
df = pd.read_pickle('output/wbedd_microdata_processed.pik')
#df=df.loc[~((df.c=='MEX') & (df.d=='USA')),:].reset_index(drop=True)

df['nf'] = df.groupby(['c','d','y'])['f'].transform(lambda x: x.nunique())
tmp2 = df.groupby(['c','d'])['nf'].mean().reset_index()
lo = tmp2.groupby('c')['nf'].agg(lambda x: x.quantile(0.75)).reset_index().rename(columns={'nf':'lo'})
hi = tmp2.groupby('c')['nf'].agg(lambda x: x.quantile(0.95)).reset_index().rename(columns={'nf':'hi'})
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

#############################################################################

print('\tEstimating tenure effect regressions on actual data...')

f2 = 'exit ~ C(d) + C(cohort) + C(tenure)'
dreg_x_m_a = ols(formula=f2,data=df_m[df_m.grp==0]).fit(cov_type='HC0')
dreg_x_m_b = ols(formula=f2,data=df_m[df_m.grp==1]).fit(cov_type='HC0')
dreg_x_p_a = ols(formula=f2,data=df_p[df_p.grp==0]).fit(cov_type='HC0')
dreg_x_p_b = ols(formula=f2,data=df_p[df_p.grp==1]).fit(cov_type='HC0')

#############################################################################
        
print('\tMaking plots of tenure effects...')
labs=['Conditional exit rate (0 at entry)']

# conditional exit
deffect_m_a = np.zeros(max_tenure_scalar+1)
deffect_m_b = np.zeros(max_tenure_scalar+1)
derr_m_a = np.zeros(max_tenure_scalar+1)
derr_m_b = np.zeros(max_tenure_scalar+1)

deffect_p_a = np.zeros(max_tenure_scalar+1)
deffect_p_b = np.zeros(max_tenure_scalar+1)
derr_p_a = np.zeros(max_tenure_scalar+1)
derr_p_b = np.zeros(max_tenure_scalar+1)

for j in range(1,max_tenure_scalar+1):
    dcoeff_m_a = dreg_x_m_a.params["C(tenure)[T.%d.0]"%j]
    dcoeff_m_b = dreg_x_m_b.params["C(tenure)[T.%d.0]"%j]
    deffect_m_a[j] = dcoeff_m_a
    deffect_m_b[j] = dcoeff_m_b

    derr_m_a[j] = dreg_x_m_a.conf_int(alpha=0.05)[1]["C(tenure)[T.%d.0]"%j]-deffect_m_a[j]
    derr_m_b[j] = dreg_x_m_b.conf_int(alpha=0.05)[1]["C(tenure)[T.%d.0]"%j]-deffect_m_b[j]

    dcoeff_p_a = dreg_x_p_a.params["C(tenure)[T.%d.0]"%j]
    dcoeff_p_b = dreg_x_p_b.params["C(tenure)[T.%d.0]"%j]
    deffect_p_a[j] = dcoeff_p_a
    deffect_p_b[j] = dcoeff_p_b

    derr_p_a[j] = dreg_x_p_a.conf_int(alpha=0.05)[1]["C(tenure)[T.%d.0]"%j]-deffect_m_a[j]
    derr_p_b[j] = dreg_x_p_b.conf_int(alpha=0.05)[1]["C(tenure)[T.%d.0]"%j]-deffect_m_b[j]

fig1,axes1=plt.subplots(1,1,figsize=(3.5,3.5),sharex=True,sharey=False)
    
tenure = range(1,max_tenure_scalar+2)

axes1.plot(tenure,deffect_m_a,
           color=colors[0],
           alpha=0.5,
           marker='o',
           linewidth=1,
           markersize=3,
           #capsize=3,
           linestyle='-',
           label='Hard (MEX)')

axes1.plot(tenure,deffect_m_b,
           color=colors[1],
           marker='s',
           alpha=0.5,
           markersize=3,
           #capsize=3,
           linewidth=1,
           linestyle='-',
           label='Easy (MEX)')

axes1.plot(tenure,deffect_p_a,
           color=colors[2],
           alpha=0.5,
           marker='D',
           linewidth=1,
           markersize=3,
           #markeredgecolor=colors[0],
           #capsize=3,
           linestyle='-',
           label='Hard (PER)')

axes1.plot(tenure,deffect_p_b,
           color=colors[3],
           marker='^',
           alpha=0.5,
           markersize=3,
           #markeredgecolor=colors[1],
           #capsize=3,
           linewidth=1,
           linestyle='-',
           label='Easy (PER)')

        
axes1.legend(loc='upper right',prop={'size':6})
axes1.set_xticks(range(1,max_tenure_scalar+2))
axes1.set_xlabel('Years in market')
axes1.set_xlabel('Years in market')

fig1.subplots_adjust(hspace=0.2,wspace=0.2)
plt.savefig('output/life_cycle_dyn_wbedd.pdf',bbox_inches='tight')

plt.close('all')

#############################################################################

print('\tEstimating duration-tenure effect regressions on actual data...')

f1 = 'np.log(v) ~ C(d) + C(cohort) + C(tenure):C(max_tenure)'

y,X = patsy.dmatrices(f1, df_m[df_m.grp==0], return_type='dataframe')
for c in X.columns:
        if(X[c].sum()<1.0e-10):
                X.drop(c,axis=1,inplace=True)
dreg_v_m_a = OLS(y,X).fit(cov_type='HC0')

y,X = patsy.dmatrices(f1, df_m[df_m.grp==1], return_type='dataframe')
for c in X.columns:
        if(X[c].sum()<1.0e-10):
                X.drop(c,axis=1,inplace=True)
dreg_v_m_b = OLS(y,X).fit(cov_type='HC0')

y,X = patsy.dmatrices(f1, df_p[df_p.grp==0], return_type='dataframe')
for c in X.columns:
        if(X[c].sum()<1.0e-10):
                X.drop(c,axis=1,inplace=True)
dreg_v_p_a = OLS(y,X).fit(cov_type='HC0')

y,X = patsy.dmatrices(f1, df_p[df_p.grp==1], return_type='dataframe')
for c in X.columns:
        if(X[c].sum()<1.0e-10):
                X.drop(c,axis=1,inplace=True)
dreg_v_p_b = OLS(y,X).fit(cov_type='HC0')


deffect_m_a = np.zeros((max_tenure_scalar+1,max_tenure_scalar+1))
deffect_m_b = np.zeros((max_tenure_scalar+1,max_tenure_scalar+1))
derr_m_a = np.zeros((max_tenure_scalar+1,max_tenure_scalar+1))
derr_m_b = np.zeros((max_tenure_scalar+1,max_tenure_scalar+1))

deffect_p_a = np.zeros((max_tenure_scalar+1,max_tenure_scalar+1))
deffect_p_b = np.zeros((max_tenure_scalar+1,max_tenure_scalar+1))
derr_p_a = np.zeros((max_tenure_scalar+1,max_tenure_scalar+1))
derr_p_b = np.zeros((max_tenure_scalar+1,max_tenure_scalar+1))


for k in range(1,max_tenure_scalar+1):
    for j in range(k+1):

        deffect_m_a[k,j] = dreg_v_m_a.params["C(max_tenure)[T.%d.0]"%(k)]
        deffect_m_b[k,j] = dreg_v_m_b.params["C(max_tenure)[T.%d.0]"%(k)]
        derr_m_a[k,j] = dreg_v_m_a.conf_int(alpha=0.05)[1]["C(max_tenure)[T.%d.0]"%(k)]
        derr_m_b[k,j] = dreg_v_m_b.conf_int(alpha=0.05)[1]["C(max_tenure)[T.%d.0]"%(k)]

        deffect_p_a[k,j] = dreg_v_p_a.params["C(max_tenure)[T.%d.0]"%(k)]
        deffect_p_b[k,j] = dreg_v_p_b.params["C(max_tenure)[T.%d.0]"%(k)]
        derr_p_a[k,j] = dreg_v_p_a.conf_int(alpha=0.05)[1]["C(max_tenure)[T.%d.0]"%(k)]
        derr_p_b[k,j] = dreg_v_p_b.conf_int(alpha=0.05)[1]["C(max_tenure)[T.%d.0]"%(k)]

        if(j>0):
            deffect_m_a[k,j] += dreg_v_m_a.params["C(tenure)[T.%d.0]:C(max_tenure)[%d.0]"%(j,k)]
            deffect_m_b[k,j] += dreg_v_m_b.params["C(tenure)[T.%d.0]:C(max_tenure)[%d.0]"%(j,k)]
            derr_m_a[k,j] += dreg_v_m_a.conf_int(alpha=0.05)[1]["C(tenure)[T.%d.0]:C(max_tenure)[%d.0]"%(j,k)]
            derr_m_b[k,j] += dreg_v_m_b.conf_int(alpha=0.05)[1]["C(tenure)[T.%d.0]:C(max_tenure)[%d.0]"%(j,k)]

            deffect_p_a[k,j] += dreg_v_p_a.params["C(tenure)[T.%d.0]:C(max_tenure)[%d.0]"%(j,k)]
            deffect_p_b[k,j] += dreg_v_p_b.params["C(tenure)[T.%d.0]:C(max_tenure)[%d.0]"%(j,k)]
            derr_p_a[k,j] += dreg_v_p_a.conf_int(alpha=0.05)[1]["C(tenure)[T.%d.0]:C(max_tenure)[%d.0]"%(j,k)]
            derr_p_b[k,j] += dreg_v_p_b.conf_int(alpha=0.05)[1]["C(tenure)[T.%d.0]:C(max_tenure)[%d.0]"%(j,k)]

                       
        derr_m_a[k,j] -= deffect_m_a[k,j]
        derr_m_b[k,j] -= deffect_m_b[k,j]

        derr_p_a[k,j] -= deffect_p_a[k,j]
        derr_p_b[k,j] -= deffect_p_b[k,j]
        

                
print('\tMaking plots...')


fig1,axes1=plt.subplots(1,2,figsize=(7,3),sharex=True,sharey=True)
               
axes1[0].set_title('(a) Hard destinations',y=1.025)
axes1[1].set_title('(b) Easy destinations',y=1.025)

lns=[]
for k in range(1,max_tenure_scalar+1):
        
    tenure = [x+1 for x in range(k+1)]
    
    l=axes1[0].plot(tenure,deffect_m_a[k,:k+1],
                    color=colors[k-1],
                    alpha=0.5,
                    marker='o',
                    linewidth=1,
                    markersize=3,
                    #capsize=3,
                    linestyle='-',
                    label='Duration = %d (MEX)'%(k+1))
    lns.append(l)

    l=axes1[1].plot(tenure,deffect_m_b[k,:k+1],
                    color=colors[k-1],
                    marker='o',
                    alpha=0.5,
                    markersize=3,
                    #capsize=3,
                    linewidth=1,
                    linestyle='-',
                    label='Duration = %d (MEX)'%(k+1))
    lns.append(l)
        
for k in range(1,max_tenure_scalar+1):

    tenure = [x+1 for x in range(k+1)]
    
    l=axes1[0].plot(tenure,deffect_p_a[k,:k+1],
                    color=colors[k-1],
                    alpha=0.5,
                    marker='s',
                    linewidth=1,
                    markersize=3,
                    #capsize=3,
                    linestyle='--',
                    label='Duration = %d (PER)'%(k+1))
    lns.append(l)


    l=axes1[1].plot(tenure,deffect_p_b[k,:k+1],
                    color=colors[k-1],
                    marker='s',
                    alpha=0.5,
                    markersize=3,
                    #capsize=3,
                    linewidth=1,
                    linestyle='--',
                    label='Duration = %d (PER)'%(k+1))
    lns.append(l)
        

tablelegend(axes1[0], ncol=2, loc='upper left',#bbox_to_anchor=(0.063,0.62), 
            row_labels=['2', '3', '4', '5'], 
            col_labels=['MEX','PER'], 
            title_label='Dur.')


axes1[0].set_xticks(range(1,max_tenure_scalar+2))
axes1[1].set_xticks(range(1,max_tenure_scalar+2))
axes1[0].set_xlabel('Years in market')
axes1[1].set_xlabel('Years in market')
axes1[0].set_ylim(-0.5,3)
axes1[1].set_ylim(-0.5,3)
#axes1[1].set_yticks([])
axes1[0].set_ylabel('log exports (relative to duration = 1)')


fig1.subplots_adjust(hspace=0.1,wspace=0.1)

plt.sca(axes1[0])
plt.savefig('output/life_cycle_dyn2_wbedd.pdf',bbox_inches='tight')

plt.close('all')
