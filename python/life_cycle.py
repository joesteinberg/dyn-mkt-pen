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

outpath='/home/joseph/Research/ongoing_projects/dyn_mkt_pen/v2/programs/python/output/'

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


max_tenure_scalar=5

#############################################################################

print('\tLoading the processed microdata...')

# load the preprocessed data
df = pd.read_pickle(outpath + 'bra_microdata_processed.pik')

df['nf'] = df.groupby(['d','y'])['f'].transform(lambda x: x.nunique())
tmp2 = df.groupby('d')['nf'].mean().reset_index()
p50 = tmp2.nf.quantile(0.5)
p90 = tmp2.nf.quantile(0.90)               
df['grp'] = np.nan
df.loc[df.nf<p50,'grp']=0
df.loc[df.nf>p90,'grp']=1
df=df[df.tenure.notnull()].reset_index(drop=True)
df.loc[df.tenure>max_tenure_scalar,'tenure']=max_tenure_scalar
df.loc[df.max_tenure>max_tenure_scalar,'max_tenure']=max_tenure_scalar
dfs = df[df.max_tenure>=max_tenure_scalar].reset_index(drop=True)

print('\tLoading the simulated data...')

df2 = pd.read_pickle(outpath + 'model_microdata_processed.pik')
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

df2_alt = None
df2s_alt = None
pref=''
altlab=''

if len(sys.argv)>1 and sys.argv[1]=='sunk':
        pref='_sunkcost'
        altlab='Sunk cost'
        df2_alt = pd.read_pickle('output/sunkcost_microdata_processed.pik')

elif len(sys.argv)>1 and sys.argv[1]=='acr':
        pref='_acr'
        altlab='Exog. entrant dyn.'
        df2_alt = pd.read_pickle('output/acr_microdata_processed.pik')

elif len(sys.argv)>1 and sys.argv[1]=='smp':
        pref='_smp'
        altlab='Static mkt. pen.'
        df2_alt = pd.read_pickle('output/smp_microdata_processed.pik')

if pref!='':
    df2_alt['nf'] = df2_alt.groupby(['d','y'])['f'].transform(lambda x: x.nunique())
    tmp2 = df2_alt.groupby('d')['nf'].mean().reset_index()
    p50 = tmp2.nf.quantile(0.5)
    p90 = tmp2.nf.quantile(0.90)               
    df2_alt['grp'] = np.nan
    df2_alt.loc[df2_alt.nf<p50,'grp']=0
    df2_alt.loc[df2_alt.nf>p90,'grp']=1
    df2_alt=df2_alt[df2_alt.tenure.notnull()].reset_index(drop=True)
    df2_alt.loc[df2_alt.tenure>max_tenure_scalar,'tenure']=max_tenure_scalar
    df2_alt.loc[df2_alt.max_tenure>max_tenure_scalar,'max_tenure']=max_tenure_scalar
    df2s_alt = df2_alt[df2_alt.max_tenure>=max_tenure_scalar].reset_index(drop=True)
    

#############################################################################

print('\tEstimating tenure effect regressions on actual data...')

#f1 = 'np.log(v) ~ C(d) + C(cohort) + C(tenure)'
#dreg_v_a = ols(formula=f1,data=dfs[dfs.grp==0]).fit(cov_type='HC0')
#dreg_v_b= ols(formula=f1,data=dfs[dfs.grp==1]).fit(cov_type='HC0')

f2 = 'exit ~ C(d) + C(cohort) + C(tenure)'
dreg_x_a = ols(formula=f2,data=df[df.grp==0]).fit(cov_type='HC0')
dreg_x_b = ols(formula=f2,data=df[df.grp==1]).fit(cov_type='HC0')
dreg_x = ols(formula=f2,data=df).fit(cov_type='HC0')


print('\tEstimating tenure effect regressions on simulated data...')

#f1 = 'np.log(v) ~ C(d) + C(tenure)'
#sreg_v_a = ols(formula=f1,data=df2s[df2s.grp==0]).fit(cov_type='HC0')
#sreg_v_b = ols(formula=f1,data=df2s[df2s.grp==1]).fit(cov_type='HC0')

f2 = 'exit ~ C(d) + C(tenure)'
sreg_x_a = ols(formula=f2,data=df2[df2.grp==0]).fit(cov_type='HC0')
sreg_x_b = ols(formula=f2,data=df2[df2.grp==1]).fit(cov_type='HC0')
sreg_x = ols(formula=f2,data=df2).fit(cov_type='HC0')

sreg2_x_a = None
sreg2_x_b = None
sreg_x = None

if pref!='':
    sreg2_x_a = ols(formula=f2,data=df2_alt[df2_alt.grp==0]).fit(cov_type='HC0')
    sreg2_x_b = ols(formula=f2,data=df2_alt[df2_alt.grp==1]).fit(cov_type='HC0')
    sreg2_x = ols(formula=f2,data=df2_alt).fit(cov_type='HC0')
    

caldata = np.genfromtxt(outpath + "calibration_data.txt",delimiter=" ")
assert (caldata.shape == (3,3+10)), 'Error! Calibration data file wrong size! Run sumstats first!'
    
caldata2 = np.zeros((3,10+40))

caldata2[0][0] = dreg_x_a.params['C(tenure)[T.1.0]']
caldata2[0][1] = dreg_x_a.params['C(tenure)[T.2.0]']
caldata2[0][2] = dreg_x_a.params['C(tenure)[T.3.0]']
caldata2[0][3] = dreg_x_a.params['C(tenure)[T.4.0]']
caldata2[0][4] = dreg_x_a.params['C(tenure)[T.5.0]']

caldata2[0][5] = dreg_x_b.params['C(tenure)[T.1.0]']
caldata2[0][6] = dreg_x_b.params['C(tenure)[T.2.0]']
caldata2[0][7] = dreg_x_b.params['C(tenure)[T.3.0]']
caldata2[0][8] = dreg_x_b.params['C(tenure)[T.4.0]']
caldata2[0][9] = dreg_x_b.params['C(tenure)[T.5.0]']

caldata2[2][0] = dreg_x_a.bse['C(tenure)[T.1.0]']
caldata2[2][1] = dreg_x_a.bse['C(tenure)[T.2.0]']
caldata2[2][2] = dreg_x_a.bse['C(tenure)[T.3.0]']
caldata2[2][3] = dreg_x_a.bse['C(tenure)[T.4.0]']
caldata2[2][4] = dreg_x_a.bse['C(tenure)[T.5.0]']

caldata2[2][5] = dreg_x_b.bse['C(tenure)[T.1.0]']
caldata2[2][6] = dreg_x_b.bse['C(tenure)[T.2.0]']
caldata2[2][7] = dreg_x_b.bse['C(tenure)[T.3.0]']
caldata2[2][8] = dreg_x_b.bse['C(tenure)[T.4.0]']
caldata2[2][9] = dreg_x_b.bse['C(tenure)[T.5.0]']

caldata2[1][0] = sreg_x_a.params['C(tenure)[T.1]']
caldata2[1][1] = sreg_x_a.params['C(tenure)[T.2]']
caldata2[1][2] = sreg_x_a.params['C(tenure)[T.3]']
caldata2[1][3] = sreg_x_a.params['C(tenure)[T.4]']
caldata2[1][4] = sreg_x_a.params['C(tenure)[T.5]']

caldata2[1][5] = sreg_x_b.params['C(tenure)[T.1]']
caldata2[1][6] = sreg_x_b.params['C(tenure)[T.2]']
caldata2[1][7] = sreg_x_b.params['C(tenure)[T.3]']
caldata2[1][8] = sreg_x_b.params['C(tenure)[T.4]']
caldata2[1][9] = sreg_x_b.params['C(tenure)[T.5]']

caldata2[2][0:10] = 10000*caldata2[2][0:10]

#############################################################################
        
print('\tMaking plots of tenure effects...')
labs=['Conditional exit rate (0 at entry)']

fig1,axes1=plt.subplots(1,1,figsize=(3.5,3.5),sharex=True,sharey=False)

# conditional exit
deffect_a = np.zeros(max_tenure_scalar+1)
deffect_b = np.zeros(max_tenure_scalar+1)
derr_a = np.zeros(max_tenure_scalar+1)
derr_b = np.zeros(max_tenure_scalar+1)

for j in range(1,max_tenure_scalar+1):
        dcoeff_a = dreg_x_a.params["C(tenure)[T.%d.0]"%j]
        dcoeff_b = dreg_x_b.params["C(tenure)[T.%d.0]"%j]
        
        deffect_a[j] = dcoeff_a
        deffect_b[j] = dcoeff_b

        derr_a[j] = dreg_x_a.conf_int(alpha=0.05)[1]["C(tenure)[T.%d.0]"%j]-deffect_a[j]
        derr_b[j] = dreg_x_b.conf_int(alpha=0.05)[1]["C(tenure)[T.%d.0]"%j]-deffect_b[j]
                        
axes1.set_title(labs[0],y=1.025)

tenure = range(1,max_tenure_scalar+2)

axes1.plot(tenure,deffect_a,
           color=colors[0],
           alpha=0.5,
           marker='o',
           linewidth=1,
           markersize=3,
           #capsize=3,
           linestyle='-',
           label='Hard destinations')

axes1.plot(tenure,deffect_b,
           color=colors[1],
           marker='s',
           alpha=0.5,
           markersize=3,
           #capsize=3,
           linewidth=1,
           linestyle='-',
           label='Easy destinations')

#for cap in caps:
#        cap.set_markeredgewidth(1)

        
axes1.legend(loc='upper right',prop={'size':6})
axes1.set_xticks(range(1,max_tenure_scalar+2))
axes1.set_xlabel('Years in market')
axes1.set_xlabel('Years in market')

fig1.subplots_adjust(hspace=0.2,wspace=0.2)
plt.savefig(outpath + 'life_cycle_dyn_data_only.pdf',bbox_inches='tight')

plt.close('all')


####-----------

deffect_a = np.zeros(max_tenure_scalar+1)
deffect_b = np.zeros(max_tenure_scalar+1)
deffect_c = np.zeros(max_tenure_scalar+1)
derr_a = np.zeros(max_tenure_scalar+1)
derr_b = np.zeros(max_tenure_scalar+1)
derr_c = np.zeros(max_tenure_scalar+1)

seffect_a = np.zeros(max_tenure_scalar+1)
seffect_b = np.zeros(max_tenure_scalar+1)
seffect_c = np.zeros(max_tenure_scalar+1)
serr_a = np.zeros(max_tenure_scalar+1)
serr_b = np.zeros(max_tenure_scalar+1)
serr_c = np.zeros(max_tenure_scalar+1)

seffect2_a = np.zeros(max_tenure_scalar+1)
seffect2_b = np.zeros(max_tenure_scalar+1)
seffect2_c = np.zeros(max_tenure_scalar+1)
serr2_a = np.zeros(max_tenure_scalar+1)
serr2_b = np.zeros(max_tenure_scalar+1)
serr2_c = np.zeros(max_tenure_scalar+1)

# conditional exit
for j in range(1,max_tenure_scalar+1):
        dcoeff_a = dreg_x_a.params["C(tenure)[T.%d.0]"%j]
        dcoeff_b = dreg_x_b.params["C(tenure)[T.%d.0]"%j]

        deffect_a[j] = dcoeff_a
        deffect_b[j] = dcoeff_b

        derr_a[j] = dreg_x_a.conf_int(alpha=0.05)[1]["C(tenure)[T.%d.0]"%j]-deffect_a[j]
        derr_b[j] = dreg_x_b.conf_int(alpha=0.05)[1]["C(tenure)[T.%d.0]"%j]-deffect_b[j]


        scoeff_a = sreg_x_a.params["C(tenure)[T.%d]"%j]
        scoeff_b = sreg_x_b.params["C(tenure)[T.%d]"%j]

        seffect_a[j] = scoeff_a
        seffect_b[j] = scoeff_b

        serr_a[j] = sreg_x_a.conf_int(alpha=0.05)[1]["C(tenure)[T.%d]"%j]-seffect_a[j]
        serr_b[j] = sreg_x_b.conf_int(alpha=0.05)[1]["C(tenure)[T.%d]"%j]-seffect_b[j]

        if pref!='':
            scoeff2_a = sreg2_x_a.params["C(tenure)[T.%d]"%j]
            scoeff2_b = sreg2_x_b.params["C(tenure)[T.%d]"%j]

            seffect2_a[j] = scoeff2_a
            seffect2_b[j] = scoeff2_b

            serr2_a[j] = sreg2_x_a.conf_int(alpha=0.05)[1]["C(tenure)[T.%d]"%j]-seffect_a[j]
            serr2_b[j] = sreg2_x_b.conf_int(alpha=0.05)[1]["C(tenure)[T.%d]"%j]-seffect_b[j]

            
                        
fig1,axes1=plt.subplots(1,1,figsize=(3.5,3.5),sharex=True,sharey=False)
lns=[]

ln=axes1.plot(tenure,deffect_a,
              color=colors[0],
              alpha=0.5,
              marker='o',
              linewidth=1,
              markersize=3,
              #markeredgecolor=colors[0],
              #capsize=3,
              linestyle='-',
              label='Hard destinations (data)')
lns.append(ln)

ln=axes1.plot(tenure,deffect_b,
              color=colors[1],
              marker='s',
              alpha=0.5,
              markersize=3,
              #markeredgecolor=colors[1],
              #capsize=3,
              linewidth=1,
              linestyle='-',
              label='Easy destinations (data)')

lns.append(ln)

ln=axes1.plot(tenure,seffect_a,
              color=colors[2],
              alpha=0.5,
              marker='D',
              linewidth=1,
              markersize=3,
              #markeredgecolor=colors[0],
              #capsize=3,
              linestyle='-',
              label='Hard destinations (model)')
lns.append(ln)

ln=axes1.plot(tenure,seffect_b,
              color=colors[3],
              marker='^',
              alpha=0.5,
              markersize=3,
              #markeredgecolor=colors[1],
              #capsize=3,
              linewidth=1,
              linestyle='-',
              label='Easy destinations (model)')

lns.append(ln)
        
#labs = [l.get_label() for l in lns]
axes1.legend(loc='upper right',prop={'size':6})

axes1.set_xticks(range(1,max_tenure_scalar+2))
axes1.set_ylabel(labs[0])
axes1.set_xlabel('Years in market')

fig1.subplots_adjust(hspace=0.2,wspace=0.2)

#plt.sca(axes1[0])
plt.savefig(outpath + 'life_cycle_dyn_model_vs_data.pdf',bbox_inches='tight')

plt.close('all')





if pref!='':
    fig1,axes1=plt.subplots(1,1,figsize=(3.5,3.5),sharex=True,sharey=False)
    lns=[]

    ln=axes1.plot(tenure,seffect_a,
                  color=colors[0],
                  alpha=0.5,
                  marker='o',
                  linewidth=1,
                  markersize=3,
                  #markeredgecolor=colors[0],
                  #capsize=3,
                  linestyle='-',
                  label='Hard destinations (baseline)')
    lns.append(ln)

    ln=axes1.plot(tenure,seffect_b,
                  color=colors[1],
                  marker='s',
                  alpha=0.5,
                  markersize=3,
                  #markeredgecolor=colors[1],
                  #capsize=3,
                  linewidth=1,
                  linestyle='-',
                  label='Easy destinations (baseline)')

    lns.append(ln)

    ln=axes1.plot(tenure,seffect2_a,
                  color=colors[2],
                  alpha=0.5,
                  marker='D',
                  linewidth=1,
                  markersize=3,
                  #markeredgecolor=colors[0],
                  #capsize=3,
                  linestyle='-',
                  label='Hard destinations (%s)'%altlab)
    lns.append(ln)

    ln=axes1.plot(tenure,seffect2_b,
                  color=colors[3],
                  marker='^',
                  alpha=0.5,
                  markersize=3,
                  #markeredgecolor=colors[1],
                  #capsize=3,
                  linewidth=1,
                  linestyle='-',
                  label='Easy destinations (%s)'%altlab)

    lns.append(ln)
        
    #labs = [l.get_label() for l in lns]
    axes1.legend(loc='upper right',prop={'size':6})
    
    axes1.set_xticks(range(1,max_tenure_scalar+2))
    axes1.set_ylabel(labs[0])
    axes1.set_xlabel('Years in market')

    fig1.subplots_adjust(hspace=0.2,wspace=0.2)

    #plt.sca(axes1[0])
    plt.savefig(outpath + 'life_cycle_dyn'+pref+'.pdf',bbox_inches='tight')

plt.close('all')





#############################################################################

print('\tEstimating duration-tenure effect regressions on actual data...')

f1 = 'np.log(v) ~ C(d) + C(cohort) + C(tenure):C(max_tenure)'
y,X = patsy.dmatrices(f1, df[df.grp==0], return_type='dataframe')
for c in X.columns:
        if(X[c].sum()<1.0e-10):
                X.drop(c,axis=1,inplace=True)
dreg_v_a = OLS(y,X).fit(cov_type='HC0')

y,X = patsy.dmatrices(f1, df[df.grp==1], return_type='dataframe')
for c in X.columns:
        if(X[c].sum()<1.0e-10):
                X.drop(c,axis=1,inplace=True)
dreg_v_b = OLS(y,X).fit(cov_type='HC0')


print('\tEstimating duration-tenure effect regressions on simulated data...')

f1 = 'np.log(v) ~ C(d) + C(tenure):C(max_tenure)-1'
y,X = patsy.dmatrices(f1, df2[df2.grp==0], return_type='dataframe')
for c in X.columns:
        if(X[c].sum()<1.0e-10):
                X.drop(c,axis=1,inplace=True)
sreg_v_a = OLS(y,X).fit(cov_type='HC0')

y,X = patsy.dmatrices(f1, df2[df2.grp==1], return_type='dataframe')
for c in X.columns:
        if(X[c].sum()<1.0e-10):
                X.drop(c,axis=1,inplace=True)
sreg_v_b = OLS(y,X).fit(cov_type='HC0')

sreg2_v_a = None
sreg2_v_b = None

if pref!='':
    y,X = patsy.dmatrices(f1, df2_alt[df2_alt.grp==0], return_type='dataframe')
    for c in X.columns:
        if(X[c].sum()<1.0e-10):
            X.drop(c,axis=1,inplace=True)
    sreg2_v_a = OLS(y,X).fit(cov_type='HC0')

    y,X = patsy.dmatrices(f1, df2_alt[df2_alt.grp==1], return_type='dataframe')
    for c in X.columns:
        if(X[c].sum()<1.0e-10):
            X.drop(c,axis=1,inplace=True)
    sreg2_v_b = OLS(y,X).fit(cov_type='HC0')


deffect_a = np.zeros((max_tenure_scalar+1,max_tenure_scalar+1))
deffect_b = np.zeros((max_tenure_scalar+1,max_tenure_scalar+1))
derr_a = np.zeros((max_tenure_scalar+1,max_tenure_scalar+1))
derr_b = np.zeros((max_tenure_scalar+1,max_tenure_scalar+1))
dse_a = np.zeros((max_tenure_scalar+1,max_tenure_scalar+1))
dse_b = np.zeros((max_tenure_scalar+1,max_tenure_scalar+1))
seffect_a = np.zeros((max_tenure_scalar+1,max_tenure_scalar+1))
seffect_b = np.zeros((max_tenure_scalar+1,max_tenure_scalar+1))
serr_a = np.zeros((max_tenure_scalar+1,max_tenure_scalar+1))
serr_b = np.zeros((max_tenure_scalar+1,max_tenure_scalar+1))
seffect2_a = np.zeros((max_tenure_scalar+1,max_tenure_scalar+1))
seffect2_b = np.zeros((max_tenure_scalar+1,max_tenure_scalar+1))
serr2_a = np.zeros((max_tenure_scalar+1,max_tenure_scalar+1))
serr2_b = np.zeros((max_tenure_scalar+1,max_tenure_scalar+1))


for k in range(1,max_tenure_scalar+1):
        for j in range(k+1):

                deffect_a[k,j] = dreg_v_a.params["C(max_tenure)[T.%d.0]"%(k)]
                deffect_b[k,j] = dreg_v_b.params["C(max_tenure)[T.%d.0]"%(k)]
                
                derr_a[k,j] = dreg_v_a.conf_int(alpha=0.05)[1]["C(max_tenure)[T.%d.0]"%(k)]
                derr_b[k,j] = dreg_v_b.conf_int(alpha=0.05)[1]["C(max_tenure)[T.%d.0]"%(k)]

                dse_a[k,j] = dreg_v_a.bse["C(max_tenure)[T.%d.0]"%(k)]
                dse_b[k,j] = dreg_v_b.bse["C(max_tenure)[T.%d.0]"%(k)]

                seffect_a[k,j] = sreg_v_a.params["C(max_tenure)[T.%d]"%(k)]
                seffect_b[k,j] = sreg_v_b.params["C(max_tenure)[T.%d]"%(k)]
                
                serr_a[k,j] = sreg_v_a.conf_int(alpha=0.05)[1]["C(max_tenure)[T.%d]"%(k)]
                serr_b[k,j] = sreg_v_b.conf_int(alpha=0.05)[1]["C(max_tenure)[T.%d]"%(k)]


                if pref!='':
                    seffect2_a[k,j] = sreg2_v_a.params["C(max_tenure)[T.%d]"%(k)]
                    seffect2_b[k,j] = sreg2_v_b.params["C(max_tenure)[T.%d]"%(k)]
                
                    serr2_a[k,j] = sreg2_v_a.conf_int(alpha=0.05)[1]["C(max_tenure)[T.%d]"%(k)]
                    serr2_b[k,j] = sreg2_v_b.conf_int(alpha=0.05)[1]["C(max_tenure)[T.%d]"%(k)]

                    
                if(j>0):
                        deffect_a[k,j] += dreg_v_a.params["C(tenure)[T.%d.0]:C(max_tenure)[%d.0]"%(j,k)]
                        deffect_b[k,j] += dreg_v_b.params["C(tenure)[T.%d.0]:C(max_tenure)[%d.0]"%(j,k)]
                
                        derr_a[k,j] += dreg_v_a.conf_int(alpha=0.05)[1]["C(tenure)[T.%d.0]:C(max_tenure)[%d.0]"%(j,k)]
                        derr_b[k,j] += dreg_v_b.conf_int(alpha=0.05)[1]["C(tenure)[T.%d.0]:C(max_tenure)[%d.0]"%(j,k)]

                        dse_a[k,j] += dreg_v_a.bse["C(tenure)[T.%d.0]:C(max_tenure)[%d.0]"%(j,k)]
                        dse_b[k,j] += dreg_v_b.bse["C(tenure)[T.%d.0]:C(max_tenure)[%d.0]"%(j,k)]

                        seffect_a[k,j] += sreg_v_a.params["C(tenure)[T.%d]:C(max_tenure)[%d]"%(j,k)]
                        seffect_b[k,j] += sreg_v_b.params["C(tenure)[T.%d]:C(max_tenure)[%d]"%(j,k)]
                
                        serr_a[k,j] += sreg_v_a.conf_int(alpha=0.05)[1]["C(tenure)[T.%d]:C(max_tenure)[%d]"%(j,k)]
                        serr_b[k,j] += sreg_v_b.conf_int(alpha=0.05)[1]["C(tenure)[T.%d]:C(max_tenure)[%d]"%(j,k)]


                        if pref!='':
                            seffect2_a[k,j] += sreg2_v_a.params["C(tenure)[T.%d]:C(max_tenure)[%d]"%(j,k)]
                            seffect2_b[k,j] += sreg2_v_b.params["C(tenure)[T.%d]:C(max_tenure)[%d]"%(j,k)]
                
                            serr2_a[k,j] += sreg2_v_a.conf_int(alpha=0.05)[1]["C(tenure)[T.%d]:C(max_tenure)[%d]"%(j,k)]
                            serr2_b[k,j] += sreg2_v_b.conf_int(alpha=0.05)[1]["C(tenure)[T.%d]:C(max_tenure)[%d]"%(j,k)]
                            

                        
                derr_a[k,j] -= deffect_a[k,j]
                derr_b[k,j] -= deffect_b[k,j]
                
                serr_a[k,j] -= seffect_a[k,j]
                serr_b[k,j] -= seffect_b[k,j]

                if pref!='':
                    serr2_a[k,j] -= seffect2_a[k,j]
                    serr2_b[k,j] -= seffect2_b[k,j]
                    



calcol=10
for k in range(1,max_tenure_scalar+1):
        for j in range(k+1):
            caldata2[0][calcol] = deffect_a[k,j]
            caldata2[1][calcol] = seffect_a[k,j]
            caldata2[2][calcol] = dse_a[k,j]
            
            if(not(j==0 or j==k)):
                    caldata2[2][calcol] = caldata2[2][calcol]*10000
            else:
                    caldata2[2][calcol] = caldata2[2][calcol]/1

            calcol = calcol+1

for k in range(1,max_tenure_scalar+1):
        for j in range(k+1):
            caldata2[0][calcol] = deffect_b[k,j]
            caldata2[1][calcol] = seffect_b[k,j]
            caldata2[2][calcol] = dse_b[k,j]
            
            if(not(j==0 or j==k)):
                    caldata2[2][calcol] = caldata2[2][calcol]*10000
            else:
                    caldata2[2][calcol] = caldata2[2][calcol]/2.5

            calcol = calcol+1
                
print('\tMaking plots...')

fig1,axes1=plt.subplots(1,2,figsize=(7,3),sharex=True,sharey=True)
               
axes1[0].set_title('(a) Hard destinations',y=1.025)
axes1[1].set_title('(b) Easy destinations',y=1.025)

for k in range(1,max_tenure_scalar+1):

    tenure = [x+1 for x in range(k+1)]
    
    axes1[0].plot(tenure,deffect_a[k,:k+1],
                  color=colors[k-1],
                  alpha=0.5,
                  marker='o',
                  linewidth=1,
                  markersize=3,
                  #capsize=3,
                  linestyle='-',
                  label='Duration = %d'%(k+1))

    axes1[1].plot(tenure,deffect_b[k,:k+1],
                  color=colors[k-1],
                  marker='o',
                  alpha=0.5,
                  markersize=3,
                  #capsize=3,
                  linewidth=1,
                  linestyle='-',
                  label='Duration = %d'%(k+1))

        #for cap in caps:
        #        cap.set_markeredgewidth(1)

axes1[0].set_ylim(0,3)
axes1[1].set_ylim(0,3)
axes1[0].legend(loc='upper left',prop={'size':6})
axes1[0].set_xticks(range(1,max_tenure_scalar+2))
axes1[1].set_xticks(range(1,max_tenure_scalar+2))
axes1[0].set_xlabel('Years in market')
axes1[1].set_xlabel('Years in market')
#axes1[1].set_yticks([])
axes1[0].set_ylabel('log exports (relative to duration = 1)')

fig1.subplots_adjust(hspace=0.15,wspace=0.1)

plt.sca(axes1[0])
plt.savefig(outpath + 'life_cycle_dyn2_data_only.pdf',bbox_inches='tight')

plt.close('all')






fig1,axes1=plt.subplots(2,1,figsize=(4,6),sharex=True,sharey=True)
               
axes1[0].set_title('(a) Hard destinations',y=1.025)
axes1[1].set_title('(b) Easy destinations',y=1.025)

for k in range(1,max_tenure_scalar+1):

    tenure = [x+1 for x in range(k+1)]
    
    axes1[0].plot(tenure,deffect_a[k,:k+1],
                  color=colors[k-1],
                  alpha=0.5,
                  marker='o',
                  linewidth=1,
                  markersize=3,
                  #capsize=3,
                  linestyle='-',
                  label='Duration = %d'%(k+1))

    axes1[1].plot(tenure,deffect_b[k,:k+1],
                  color=colors[k-1],
                  marker='o',
                  alpha=0.5,
                  markersize=3,
                  #capsize=3,
                  linewidth=1,
                  linestyle='-',
                  label='Duration = %d'%(k+1))

        #for cap in caps:
        #        cap.set_markeredgewidth(1)

axes1[0].set_ylim(0,3)
axes1[1].set_ylim(0,3)
axes1[0].legend(loc='upper left',prop={'size':6})
axes1[0].set_xticks(range(1,max_tenure_scalar+2))
axes1[1].set_xticks(range(1,max_tenure_scalar+2))
#axes1[0].set_xlabel('Years in market')
axes1[1].set_xlabel('Years in market')
#axes1[1].set_yticks([])
axes1[1].set_ylabel('log exports (relative to duration = 1)')
axes1[0].set_ylabel('log exports (relative to duration = 1)')

fig1.subplots_adjust(hspace=0.2,wspace=0.1)

plt.sca(axes1[0])
plt.savefig(outpath + 'life_cycle_dyn2_data_only_tall.pdf',bbox_inches='tight')

plt.close('all')








fig1,axes1=plt.subplots(1,2,figsize=(7,3),sharex=True,sharey=True)
               
axes1[0].set_title('(a) Hard destinations',y=1.025)
axes1[1].set_title('(b) Easy destinations',y=1.025)

lns=[]
for k in range(1,max_tenure_scalar+1):
        
    tenure = [x+1 for x in range(k+1)]
    
    l=axes1[0].plot(tenure,deffect_a[k,:k+1],
                    color=colors[k-1],
                    alpha=0.5,
                    marker='o',
                    linewidth=1,
                    markersize=3,
                    #capsize=3,
                    linestyle='-',
                    label='Duration = %d (data)'%(k+1))
    lns.append(l)

    l=axes1[1].plot(tenure,deffect_b[k,:k+1],
                    color=colors[k-1],
                    marker='o',
                    alpha=0.5,
                    markersize=3,
                    #capsize=3,
                    linewidth=1,
                    linestyle='-',
                    label='Duration = %d (data)'%(k+1))
    lns.append(l)
        
for k in range(1,max_tenure_scalar+1):

    tenure = [x+1 for x in range(k+1)]
    
    l=axes1[0].plot(tenure,seffect_a[k,:k+1],
                    color=colors[k-1],
                    alpha=0.5,
                    marker='s',
                    linewidth=1,
                    markersize=3,
                    #capsize=3,
                    linestyle='--',
                    label='Duration = %d (model)'%(k+1))
    lns.append(l)


    l=axes1[1].plot(tenure,seffect_b[k,:k+1],
                    color=colors[k-1],
                    marker='s',
                    alpha=0.5,
                    markersize=3,
                    #capsize=3,
                    linewidth=1,
                    linestyle='--',
                    label='Duration = %d (model)'%(k+1))
    lns.append(l)
        

tablelegend(axes1[0], ncol=2, loc='upper left',#bbox_to_anchor=(0.063,0.62), 
            row_labels=['2', '3', '4', '5', '6'], 
            col_labels=['Data','Model'], 
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
plt.savefig(outpath + 'life_cycle_dyn2_model_vs_data.pdf',bbox_inches='tight')

plt.close('all')





if pref!='':
    fig1,axes1=plt.subplots(1,2,figsize=(7,3),sharex=True,sharey=True)
               
    axes1[0].set_title('(a) Hard destinations',y=1.025)
    axes1[1].set_title('(b) Easy destinations',y=1.025)

    lns=[]
    for k in range(1,max_tenure_scalar+1):
        
        tenure = [x+1 for x in range(k+1)]
    
        l=axes1[0].plot(tenure,seffect_a[k,:k+1],
                        color=colors[k-1],
                        alpha=0.5,
                        marker='o',
                        linewidth=1,
                        markersize=3,
                        #capsize=3,
                        linestyle='-',
                        label='Duration = %d (baseline)'%(k+1))
        lns.append(l)

        l=axes1[1].plot(tenure,seffect_b[k,:k+1],
                        color=colors[k-1],
                        marker='o',
                        alpha=0.5,
                        markersize=3,
                        #capsize=3,
                        linewidth=1,
                        linestyle='-',
                        label='Duration = %d (baseline)'%(k+1))
    lns.append(l)
        
    for k in range(1,max_tenure_scalar+1):

        tenure = [x+1 for x in range(k+1)]
    
        l=axes1[0].plot(tenure,seffect2_a[k,:k+1],
                        color=colors[k-1],
                        alpha=0.5,
                        marker='s',
                        linewidth=1,
                        markersize=3,
                        #capsize=3,
                        linestyle='--',
                        label='Duration = %d (%s)'%(k+1,altlab))
        lns.append(l)


        l=axes1[1].plot(tenure,seffect2_b[k,:k+1],
                        color=colors[k-1],
                        marker='s',
                        alpha=0.5,
                        markersize=3,
                        #capsize=3,
                        linewidth=1,
                        linestyle='--',
                        label='Duration = %d (%s)'%(k+1,altlab))
        lns.append(l)
        

    tablelegend(axes1[0], ncol=2, loc='upper left',#bbox_to_anchor=(0.063,0.62), 
                row_labels=['2', '3', '4', '5', '6'], 
                col_labels=['Baseline',altlab], 
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
    plt.savefig(outpath + 'life_cycle_dyn2'+pref+'.pdf',bbox_inches='tight')

plt.close('all')







caldata3 = np.hstack((caldata,caldata2))
np.savetxt(outpath + "calibration_data2.txt",caldata3,delimiter=" ")









fig1,axes1=plt.subplots(2,1,figsize=(4,6),sharex=True,sharey=True)
               
axes1[0].set_title('(a) Hard destinations',y=1.025)
axes1[1].set_title('(b) Easy destinations',y=1.025)

lns=[]
for k in range(1,max_tenure_scalar+1):
        
    tenure = [x+1 for x in range(k+1)]
    
    l=axes1[0].plot(tenure,deffect_a[k,:k+1],
                    color=colors[k-1],
                    alpha=0.5,
                    marker='o',
                    linewidth=1,
                    markersize=3,
                    #capsize=3,
                    linestyle='-',
                    label='Duration = %d (data)'%(k+1))
    lns.append(l)

    l=axes1[1].plot(tenure,deffect_b[k,:k+1],
                    color=colors[k-1],
                    marker='o',
                    alpha=0.5,
                    markersize=3,
                    #capsize=3,
                    linewidth=1,
                    linestyle='-',
                    label='Duration = %d (data)'%(k+1))
    lns.append(l)
        
for k in range(1,max_tenure_scalar+1):

    tenure = [x+1 for x in range(k+1)]
    
    l=axes1[0].plot(tenure,seffect_a[k,:k+1],
                    color=colors[k-1],
                    alpha=0.5,
                    marker='s',
                    linewidth=1,
                    markersize=3,
                    #capsize=3,
                    linestyle='--',
                    label='Duration = %d (model)'%(k+1))
    lns.append(l)


    l=axes1[1].plot(tenure,seffect_b[k,:k+1],
                    color=colors[k-1],
                    marker='s',
                    alpha=0.5,
                    markersize=3,
                    #capsize=3,
                    linewidth=1,
                    linestyle='--',
                    label='Duration = %d (model)'%(k+1))
    lns.append(l)
        

tablelegend(axes1[0], ncol=2, loc='upper left',#bbox_to_anchor=(0.063,0.62), 
            row_labels=['2', '3', '4', '5', '6'], 
            col_labels=['Data','Model'], 
            title_label='Dur.')


axes1[0].set_xticks(range(1,max_tenure_scalar+2))
axes1[1].set_xticks(range(1,max_tenure_scalar+2))
#axes1[0].set_xlabel('Years in market')
axes1[1].set_xlabel('Years in market')
axes1[0].set_ylim(-0.5,3)
axes1[1].set_ylim(-0.5,3)
#axes1[1].set_yticks([])
axes1[0].set_ylabel('log exports (relative to duration = 1)')
axes1[1].set_ylabel('log exports (relative to duration = 1)')


fig1.subplots_adjust(hspace=0.2,wspace=0.1)

plt.sca(axes1[0])
plt.savefig(outpath + 'life_cycle_dyn2_model_vs_data_tall.pdf',bbox_inches='tight')

plt.close('all')
