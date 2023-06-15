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
mpl.rcParams['savefig.pad_inches'] = 0
alpha=0.8
colors=['#377eb8','#e41a1c','#4daf4a','#984ea3','#ff7f00','#ffff33']
fmts = ['o','s','D','X','P']

dopath = '/home/linux/Documents/Research/dyn_mkt_pen/v3/programs/python/'
outpath = '/home/linux/Documents/Research/dyn_mkt_pen/v3/programs/python/output/'
figpath = '/home/linux/Documents/Research/dyn_mkt_pen/v3/programs/python/output/life_cycle_figs/'

alt_suffs=['smp','sunkcost','acr']
model_labs=['Baseline','Static MP','Sunk cost','Exog NED']

alt_models=False
if len(sys.argv)>1 and sys.argv[1]=='alt-models':
    alt_models=True

sensitivity=False
if len(sys.argv)>1 and sys.argv[1]=='sensitivity':
    alt_models=True
    sensitivity=True
    alt_suffs = alt_suffs + ['abn1','abn2','abo1','abo2','a0']
    model_labs = model_labs + ['$\\alpha_n=\\beta_n$','$\\beta_n=\\alpha_n$','$\\alpha_o=\\beta_o$',
                               '$\\beta_o=\\alpha_o$','$\\alpha_n=\\alpha_o=1$']

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
print('Processing exit regression results...')

dreg_x_all = pd.read_stata(outpath + 'stata/dreg_x_all.dta').set_index('var')
dreg_x_2way = pd.read_stata(outpath + 'stata/dreg_x_2way.dta').set_index('var')

sreg_x_all = [pd.read_stata(outpath + 'stata/sreg_x_all.dta').set_index('var')]
sreg_x_2way = [pd.read_stata(outpath + 'stata/sreg_x_2way.dta').set_index('var')]

if(alt_models==True):
    sreg_x_all = sreg_x_all + [pd.read_stata(outpath + 'stata/sreg_x_all_'+s+'.dta').set_index('var') for s in alt_suffs[0:3]]
    sreg_x_2way = sreg_x_2way + [pd.read_stata(outpath + 'stata/sreg_x_2way_'+s+'.dta').set_index('var') for s in alt_suffs[0:3]]

if(sensitivity==True):
    sreg_x_all = sreg_x_all + [pd.read_stata(outpath + 'stata/sreg_x_all_'+s+'.dta').set_index('var') for s in alt_suffs[3:]]
    sreg_x_2way = sreg_x_2way + [pd.read_stata(outpath + 'stata/sreg_x_2way_'+s+'.dta').set_index('var') for s in alt_suffs[3:]]


grps = ['0b.grp','1.grp']

caldata = np.genfromtxt(outpath + "calibration/calibration_data.txt",delimiter=" ")
#assert (caldata.shape == (3,3+10)), 'Error! Calibration data file wrong size! Run sumstats first!'
assert (caldata.shape == (3,4*6)), 'Error! Calibration data file wrong size! Run sumstats first!'
    
caldata2 = np.zeros((3,10+40))

for j in range(max_tenure_scalar):
    
    strx = str(j+1)+'.tenure#'+grps[0]
    stry = str(j+1)+'.tenure'
    caldata2[0][j] = dreg_x_2way.coef[strx]
    #caldata2[1][j] = sreg_x_2way[0].coef[stry]
    caldata2[1][j] = sreg_x_2way[0].coef[strx]
    caldata2[2][j] = dreg_x_2way.stderr[strx]

for j in range(max_tenure_scalar):

    strx = str(j+1)+'.tenure#'+grps[1]
    stry = str(j+1)+'.tenure'
    caldata2[0][j+5] = dreg_x_2way.coef[strx]
    #caldata2[1][j+5] = sreg_x_2way[1].coef[stry]
    caldata2[1][j+5] = sreg_x_2way[0].coef[strx]
    caldata2[2][j+5] = dreg_x_2way.stderr[strx]

caldata2[2][0:10] = 1e8


deffect_x_all = np.zeros((max_tenure_scalar+1,2))
seffect_x_all = [np.zeros(max_tenure_scalar+1) for x in sreg_x_all]
for j in range(0,max_tenure_scalar+1):
    strx = ''
    if j==0:
        strx = '0b.tenure'
    else:
        strx = str(j)+'.tenure'

    deffect_x_all[j,0] = dreg_x_all.coef[strx]
    deffect_x_all[j,1] = dreg_x_all.ci_upper[strx] - dreg_x_all.ci_lower[strx]

    for k in range(len(seffect_x_all)):
        seffect_x_all[k][j] = sreg_x_all[k].coef[strx]

deffect_x_2way = np.zeros((2,max_tenure_scalar+1,2))
seffect_x_2way = [np.zeros((2,max_tenure_scalar+1)) for x in sreg_x_2way]
for i in range(2):
    for j in range(0,max_tenure_scalar+1):

        strx = ''
        stry = ''
        if j==0:
            strx = '0b.tenure#'+grps[i]
            stry = '0b.tenure'
        else:
            strx = str(j)+'.tenure#'+grps[i]
            stry = str(j)+'.tenure'
            
        deffect_x_2way[i,j,0] = dreg_x_2way.coef[strx]
        deffect_x_2way[i,j,1] = dreg_x_2way.ci_upper[strx] - dreg_x_2way.ci_lower[strx]
        
        for k in range(len(seffect_x_2way)):
            seffect_x_2way[k][i,j] = sreg_x_2way[k].coef[strx]

#############################################################################
        
print('Making plots of tenure effects on exit...')

#--------------------------------------------------
# Data figure for paper

sz=(6,3)
tenure = range(1,max_tenure_scalar+2)

fig,axes=plt.subplots(1,2,figsize=sz,sharex=True,sharey=False)

ax = axes[0]
ax.set_title('(a) All markets',y=1.025)
ax.errorbar(tenure,deffect_x_all[:,0],yerr=deffect_x_all[:,1],
            color=colors[0],
            alpha=0.5,
            marker='o',
            linewidth=1,
            markersize=3,
            linestyle='-')
ax.set_xticks(range(1,max_tenure_scalar+2))
ax.set_xlabel('Years in market')

ax=axes[1]
ax.set_title('(b) Hard vs. easy markets',y=1.025)
ax.errorbar(tenure,deffect_x_2way[0,:,0],yerr=deffect_x_2way[0,:,1],
               color=colors[1],
               alpha=0.5,
               marker='s',
               linewidth=1,
               markersize=3,
               linestyle='-',
               label='Hard markets')

ax.errorbar(tenure,deffect_x_2way[1,:,0],yerr=deffect_x_2way[1,:,1],
               color=colors[2],
               marker='x',
               alpha=0.5,
               markersize=3,
               linewidth=1,
               linestyle='--',
               label='Easy markets')
ax.set_xticks(range(1,max_tenure_scalar+2))
ax.set_xlabel('Years in market')      
ax.legend(loc='upper right',prop={'size':6})

axes[0].set_ylabel('Conditional exit rate (0 at entry)')
fig.subplots_adjust(hspace=0.2,wspace=0.2)
plt.savefig(figpath + 'fig2_life_cycle_dyn_x_data.pdf',bbox_inches='tight')

plt.close('all')


#--------------------------------------------------
# Model figure for paper

sz=(6,3)
N=1
if(alt_models):
    sz=(6,9)
    N=4
    
fig,axes=plt.subplots(N,2,figsize=sz,sharex=True,sharey=False)
panels=['(a)','(b)','(c)','(d)','(e)','(f)','(g)','(h)']

cnt=0
for i in range(N):
    
    ax2 = None
    if(alt_models==True):
        ax2 = axes[i]
    else:
        ax2=axes
        
    ax = ax2[0]
    ax.set_title('%s %s: All markets'%(panels[cnt],model_labs[i]),y=1.025)
    cnt += 1

    ax.fill_between(tenure,deffect_x_all[:,0]-deffect_x_all[:,1],deffect_x_all[:,0]+deffect_x_all[:,1],
                    color=colors[0],
                    alpha=0.1,
                    lw=0)

    ax.plot(tenure,seffect_x_all[i],
            color=colors[0],
            alpha=0.5,
            marker='o',
            linewidth=1,
            markersize=3,
            linestyle='-')

    ax.set_xticks(range(1,max_tenure_scalar+2))
    if(i==N-1):
        ax.set_xlabel('Years in market')
    else:
        ax.set_xlabel('')        

    ax=ax2[1]
    ax.set_title('%s %s: Hard vs. easy' %(panels[cnt],model_labs[i]),y=1.025)
    cnt+=1

    ax.fill_between(tenure,deffect_x_2way[0,:,0]-deffect_x_2way[0,:,1],deffect_x_2way[0,:,0]+deffect_x_2way[0,:,1],
                    color=colors[1],
                    alpha=0.1,
                    lw=0)

    ax.plot(tenure,seffect_x_2way[i][0,:],
            color=colors[1],
            alpha=0.5,
            marker='s',
            linewidth=1,
            markersize=3,
            linestyle='-',
            label='Hard markets')
    
    ax.fill_between(tenure,deffect_x_2way[1,:,0]-deffect_x_2way[1,:,1],deffect_x_2way[1,:,0]+deffect_x_2way[1,:,1],
                    color=colors[2],
                    alpha=0.1,
                    lw=0)

    ax.plot(tenure,seffect_x_2way[i][1,:],
            color=colors[2],
            marker='x',
            alpha=0.5,
            markersize=3,
            linewidth=1,
            linestyle='--',
            label='Easy markets')

    ax.set_xticks(range(1,max_tenure_scalar+2))
    if(i==N-1):
        ax.set_xlabel('Years in market')
        ax.legend(loc='upper right',prop={'size':6})
    else:
        ax.set_xlabel('')        

fig.subplots_adjust(hspace=0.3,wspace=0.2)
plt.savefig(figpath + 'fig5_life_cycle_dyn_x_model.pdf',bbox_inches='tight')

plt.close('all')


#--------------------------------------------------
# Sensitivity analysis figure for paper

if(sensitivity==True):
    sz=(6,9)
    
    fig,axes=plt.subplots(5,2,figsize=sz,sharex=True,sharey=False)
    panels=['(a)','(b)','(c)','(d)','(e)','(f)','(g)','(h)','(i)','(j)']

    cnt=0
    for i in range(5):
        j=i+4
        ax2 = axes[i]
        
        ax = ax2[0]
        ax.set_title('%s %s: All markets'%(panels[cnt],model_labs[j]),y=1.025)
        cnt += 1

        ax.fill_between(tenure,deffect_x_all[:,0]-deffect_x_all[:,1],deffect_x_all[:,0]+deffect_x_all[:,1],
                        color=colors[0],
                        alpha=0.1,
                        lw=0)

        ax.plot(tenure,seffect_x_all[j],
                color=colors[0],
                alpha=0.5,
                marker='o',
                linewidth=1,
                markersize=3,
                linestyle='-')

        ax.set_xticks(range(1,max_tenure_scalar+2))
        if(i==4):
            ax.set_xlabel('Years in market')
        else:
            ax.set_xlabel('')        

        ax=ax2[1]
        ax.set_title('%s %s: Hard vs. easy' %(panels[cnt],model_labs[j]),y=1.025)
        cnt+=1

        ax.fill_between(tenure,deffect_x_2way[0,:,0]-deffect_x_2way[0,:,1],deffect_x_2way[0,:,0]+deffect_x_2way[0,:,1],
                        color=colors[1],
                        alpha=0.1,
                        lw=0)

        ax.plot(tenure,seffect_x_2way[j][0,:],
                color=colors[1],
                alpha=0.5,
                marker='s',
                linewidth=1,
                markersize=3,
                linestyle='-',
                label='Hard markets')
        
        ax.fill_between(tenure,deffect_x_2way[1,:,0]-deffect_x_2way[1,:,1],deffect_x_2way[1,:,0]+deffect_x_2way[1,:,1],
                        color=colors[2],
                        alpha=0.1,
                        lw=0)
        
        ax.plot(tenure,seffect_x_2way[j][1,:],
                color=colors[2],
                marker='x',
                alpha=0.5,
                markersize=3,
                linewidth=1,
                linestyle='--',
                label='Easy markets')

        ax.set_xticks(range(1,max_tenure_scalar+2))
        if(i==4):
            ax.set_xlabel('Years in market')
            ax.legend(loc='upper right',prop={'size':6})
        else:
            ax.set_xlabel('')        

    fig.subplots_adjust(hspace=0.3,wspace=0.2)
    plt.savefig(figpath + 'figA2_life_cycle_dyn_x_alpha_beta.pdf',bbox_inches='tight')
    
    plt.close('all')



################################################################################
print('Processing sales regression results...')

dreg_v_all = pd.read_stata(outpath + 'stata/dreg_v_all.dta').set_index('var')
dreg_v_3way = pd.read_stata(outpath + 'stata/dreg_v_3way.dta').set_index('var')
dreg_v_tests = pd.read_csv(outpath + 'stata/dreg_v_tests.txt',sep='\t')

sreg_v_all = [pd.read_stata(outpath + 'stata/sreg_v_all.dta').set_index('var')]
sreg_v_3way = [pd.read_stata(outpath + 'stata/sreg_v_3way.dta').set_index('var')]

if(alt_models==True):
    sreg_v_all = sreg_v_all + [pd.read_stata(outpath + 'stata/sreg_v_all_'+s+'.dta').set_index('var') for s in alt_suffs]
    sreg_v_3way = sreg_v_3way + [pd.read_stata(outpath + 'stata/sreg_v_3way_'+s+'.dta').set_index('var') for s in alt_suffs]

deffect_v_all = np.zeros((max_tenure_scalar+1,max_tenure_scalar+1,2))
seffect_v_all = [np.zeros((max_tenure_scalar+1,max_tenure_scalar+1)) for x in sreg_v_all]
for k in range(1,max_tenure_scalar+1):
    for j in range(k+1):

        strx=''
        if j==0:
            strx = '0b.tenure#'+str(k)+'.max_tenure'
        else:
            strx = str(j)+'.tenure#'+str(k)+'.max_tenure'
            
        deffect_v_all[k,j,0] = dreg_v_all.coef[strx]
        deffect_v_all[k,j,1] = dreg_v_all.ci_upper[strx] - dreg_v_all.ci_lower[strx]

        for l in range(len(sreg_v_all)):
            seffect_v_all[l][k,j] = sreg_v_all[l].coef[strx]

deffect_v_3way = np.zeros((2,max_tenure_scalar+1,max_tenure_scalar+1,2))
seffect_v_3way = [np.zeros((2,max_tenure_scalar+1,max_tenure_scalar+1)) for x in sreg_v_3way]
for i in range(2):
    for k in range(1,max_tenure_scalar+1):
        for j in range(k+1):

            strx=''
            stry=''
            if j==0:
                strx = '0b.tenure#'+str(k)+'.max_tenure#'+grps[i]
                stry = '0b.tenure#'+str(k)+'.max_tenure'
            else:
                strx = str(j)+'.tenure#'+str(k)+'.max_tenure#'+grps[i]
                stry = str(j)+'.tenure#'+str(k)+'.max_tenure'
            
            deffect_v_3way[i,k,j,0] = dreg_v_3way.coef[strx]
            deffect_v_3way[i,k,j,1] = dreg_v_3way.ci_upper[strx] - dreg_v_3way.ci_lower[strx]
            #seffect_v_3way[i,k,j] = sreg_v_3way[i].coef[stry]

            for l in range(len(sreg_v_3way)):
                seffect_v_3way[l][i,k,j] = sreg_v_3way[l].coef[strx]
            

calcol=10
for k in range(1,max_tenure_scalar+1):
        for j in range(k+1):
            caldata2[0][calcol] = deffect_v_3way[0,k,j,0]
            caldata2[1][calcol] = seffect_v_3way[0][0,k,j]
            caldata2[2][calcol] = deffect_v_3way[0,k,j,1]

            if(j==0):
                caldata2[2][calcol] = caldata2[2][calcol]/1
            elif(k==max_tenure_scalar and j==k):
                caldata2[2][calcol] = caldata2[2][calcol]/1
            else:
                caldata2[2][calcol] = caldata2[2][calcol]*10

            calcol = calcol+1

for k in range(1,max_tenure_scalar+1):
        for j in range(k+1):
            caldata2[0][calcol] = deffect_v_3way[1,k,j,0]
            caldata2[1][calcol] = seffect_v_3way[0][1,k,j]
            caldata2[2][calcol] = deffect_v_3way[1,k,j,1]

            if(j==0):
                caldata2[2][calcol] = caldata2[2][calcol]/1
            elif(k==max_tenure_scalar and j==k):
                caldata2[2][calcol] = caldata2[2][calcol]/1
            else:
                caldata2[2][calcol] = caldata2[2][calcol]*10

            calcol = calcol+1

caldata3 = np.hstack((caldata,caldata2))
np.savetxt(outpath + "calibration/calibration_data2.txt",caldata3,delimiter=" ")
            
#############################################################################
        
print('Making plots of tenure#max_tenure effects on sales...')

#--------------------------------------------------
# Data figure

sz = (7.5,3)

fig,axes=plt.subplots(1,3,figsize=sz,sharex=True,sharey=True)

for k in range(1,max_tenure_scalar+1):

    tenure = [x+1 for x in range(k+1)]

    axes[0].errorbar(tenure,deffect_v_all[k,:k+1,0],yerr=deffect_v_all[k,:k+1,1],
                     color=colors[k-1],
                     alpha=0.5,
                     marker='o',
                     linewidth=1,
                     markersize=3,
                     #capsize=3,
                     linestyle='-',
                     label='Spell length = %d'%(k+1))
    
    axes[1].errorbar(tenure,deffect_v_3way[0,k,:k+1,0],yerr=deffect_v_3way[0,k,:k+1,1],
                     color=colors[k-1],
                     marker='o',
                     alpha=0.5,
                     markersize=3,
                     #capsize=3,
                     linewidth=1,
                     linestyle='-',
                     label='Spell length = %d'%(k+1))

    axes[2].errorbar(tenure,deffect_v_3way[1,k,:k+1,0],yerr=deffect_v_3way[1,k,:k+1,1],
                     color=colors[k-1],
                     marker='o',
                     alpha=0.5,
                     markersize=3,
                     #capsize=3,
                     linewidth=1,
                     linestyle='-',
                     label='Spell length = %d'%(k+1))
    
#axes1[0].set_ylim(-0.25,3)
#axes1[1].set_ylim(-0.25,3)
axes[0].legend(loc='lower right',prop={'size':6})
axes[0].set_xticks(range(1,max_tenure_scalar+2))
axes[1].set_xticks(range(1,max_tenure_scalar+2))
axes[2].set_xticks(range(1,max_tenure_scalar+2))
axes[0].set_xlabel('Years in market')
axes[1].set_xlabel('Years in market')
axes[2].set_xlabel('Years in market')
axes[0].set_title('(a) All markets',y=1.025)
axes[1].set_title('(b) Hard markets',y=1.025)
axes[2].set_title('(c) Easy markets',y=1.025)

#axes1[1].set_yticks([])
axes[0].set_ylim(0,2.5)
axes[1].set_ylim(0,2.5)
axes[2].set_ylim(0,2.5)
axes[0].set_ylabel('log exports (1-yr spell = 0)')

fig.subplots_adjust(hspace=0.15,wspace=0.1)
plt.sca(axes[0])
plt.savefig(figpath + 'fig1_life_cycle_dyn_v_data.pdf',bbox_inches='tight')
plt.close('all')

#------------------------------------------------------
# Model figure

sz=(7.5,3)
N=1
if(alt_models):
    sz=(7.5,9)
    N=4

fig,axes=plt.subplots(N,3,figsize=sz,sharex=True,sharey=False)
panels=['(a)','(b)','(c)','(d)','(e)','(f)','(g)','(h)','(i)','(j)','(k)','(l)']

cnt=0
for i in range(N):
    
    ax = None
    if(alt_models==True):
        ax = axes[i]
    else:
        ax=axes
        
    ax[0].set_title('%s %s: All markets'%(panels[cnt],model_labs[i]),y=1.025)
    cnt += 1
    
    ax[1].set_title('%s %s: Hard markets'%(panels[cnt],model_labs[i]),y=1.025)
    cnt += 1    

    ax[2].set_title('%s %s: Easy markets'%(panels[cnt],model_labs[i]),y=1.025)
    cnt += 1    

    for k in range(1,max_tenure_scalar+1):
        tenure = [x+1 for x in range(k+1)]

        ax[0].fill_between(tenure,deffect_v_all[k,:k+1,0]-deffect_v_all[k,:k+1,1],deffect_v_all[k,:k+1,0]+deffect_v_all[k,:k+1,1],
                            color=colors[k-1],
                            alpha=0.25,
                            linewidth=0)

        ax[1].fill_between(tenure,deffect_v_3way[0,k,:k+1,0]-deffect_v_3way[0,k,:k+1,1],deffect_v_3way[0,k,:k+1,0]+deffect_v_3way[0,k,:k+1,1],
                          color=colors[k-1],
                          alpha=0.25,
                          linewidth=0)
        
        ax[2].fill_between(tenure,deffect_v_3way[1,k,:k+1,0]-deffect_v_3way[1,k,:k+1,1],deffect_v_3way[1,k,:k+1,0]+deffect_v_3way[1,k,:k+1,1],
                          color=colors[k-1],
                          alpha=0.25,
                          linewidth=0)
    
        ax[0].plot(tenure,seffect_v_all[i][k,:k+1],
                     color=colors[k-1],
                     alpha=0.5,
                     marker='s',
                     linewidth=1,
                     markersize=3,
                     #capsize=3,
                     linestyle='--',
                     label='Spell length = %d'%(k+1))
        
        ax[1].plot(tenure,seffect_v_3way[i][0,k,:k+1],
                     color=colors[k-1],
                     marker='s',
                     alpha=0.5,
                     markersize=3,
                     #capsize=3,
                     linewidth=1,
                     linestyle='--',
                     label='Spell length = %d'%(k+1))

        ax[2].plot(tenure,seffect_v_3way[i][1,k,:k+1],
                     color=colors[k-1],
                     marker='s',
                     alpha=0.5,
                     markersize=3,
                     #capsize=3,
                     linewidth=1,
                     linestyle='--',
                     label='Spell length = %d'%(k+1))

    ax[0].set_xticks(range(1,max_tenure_scalar+2))
    ax[1].set_xticks(range(1,max_tenure_scalar+2))
    ax[2].set_xticks(range(1,max_tenure_scalar+2))
    ax[0].set_ylim(-1,2.5)
    ax[1].set_ylim(-1,2.5)
    ax[2].set_ylim(-1,2.5)

    if(i==N-1):
        ax[0].set_xlabel('Years in market')
        ax[1].set_xlabel('Years in market')
        ax[2].set_xlabel('Years in market')
    else:
        ax[0].set_xlabel('')
        ax[1].set_xlabel('')
        ax[2].set_xlabel('')
            
    ax[0].set_ylabel('log exports (1-yr spell = 0)')

if(N>1):
    axes[1][2].set_ylim(-1,5)
    axes[2][1].set_ylim(-3,2.5)
    axes[2][2].set_ylim(-3,2.5)

fig.subplots_adjust(hspace=0.3,wspace=0.2)

if(N>1):
    plt.sca(axes[0][0])
    
plt.savefig(figpath + 'fig4_life_cycle_dyn_v_model.pdf',bbox_inches='tight')
plt.close('all')

#------------------------------------------------------
# Sensitivity analysis for appendix

if(sensitivity==True):
    sz=(7.5,9)
    N=5

    fig,axes=plt.subplots(N,3,figsize=sz,sharex=True,sharey=False)
    panels=['(a)','(b)','(c)','(d)','(e)','(f)','(g)','(h)','(i)','(j)','(k)','(l)','(m)','(n)','(o)']

    cnt=0
    for i in range(5):

        j=i+4
        
        ax = axes[i]
        
        ax[0].set_title('%s %s: All markets'%(panels[cnt],model_labs[j]),y=1.025)
        cnt += 1
        
        ax[1].set_title('%s %s: Hard markets'%(panels[cnt],model_labs[j]),y=1.025)
        cnt += 1    
        
        ax[2].set_title('%s %s: Easy markets'%(panels[cnt],model_labs[j]),y=1.025)
        cnt += 1    

        for k in range(1,max_tenure_scalar+1):
            tenure = [x+1 for x in range(k+1)]

            ax[0].fill_between(tenure,deffect_v_all[k,:k+1,0]-deffect_v_all[k,:k+1,1],deffect_v_all[k,:k+1,0]+deffect_v_all[k,:k+1,1],
                               color=colors[k-1],
                               alpha=0.25,
                               linewidth=0)

            ax[1].fill_between(tenure,deffect_v_3way[0,k,:k+1,0]-deffect_v_3way[0,k,:k+1,1],deffect_v_3way[0,k,:k+1,0]+deffect_v_3way[0,k,:k+1,1],
                               color=colors[k-1],
                               alpha=0.25,
                               linewidth=0)
        
            ax[2].fill_between(tenure,deffect_v_3way[1,k,:k+1,0]-deffect_v_3way[1,k,:k+1,1],deffect_v_3way[1,k,:k+1,0]+deffect_v_3way[1,k,:k+1,1],
                               color=colors[k-1],
                               alpha=0.25,
                               linewidth=0)
    
            ax[0].plot(tenure,seffect_v_all[j][k,:k+1],
                       color=colors[k-1],
                       alpha=0.5,
                       marker='s',
                       linewidth=1,
                       markersize=3,
                       #capsize=3,
                       linestyle='--',
                       label='Spell length = %d'%(k+1))
        
            ax[1].plot(tenure,seffect_v_3way[j][0,k,:k+1],
                       color=colors[k-1],
                       marker='s',
                       alpha=0.5,
                       markersize=3,
                       #capsize=3,
                       linewidth=1,
                       linestyle='--',
                       label='Spell length = %d'%(k+1))

            ax[2].plot(tenure,seffect_v_3way[j][1,k,:k+1],
                       color=colors[k-1],
                       marker='s',
                       alpha=0.5,
                       markersize=3,
                       #capsize=3,
                       linewidth=1,
                       linestyle='--',
                       label='Spell length = %d'%(k+1))

        ax[0].set_xticks(range(1,max_tenure_scalar+2))
        ax[1].set_xticks(range(1,max_tenure_scalar+2))
        ax[2].set_xticks(range(1,max_tenure_scalar+2))
        #ax[0].set_ylim(-1,2.5)
        #ax[1].set_ylim(-1,2.5)
        #ax[2].set_ylim(-1,2.5)

        if(i==4):
            ax[0].set_xlabel('Years in market')
            ax[1].set_xlabel('Years in market')
            ax[2].set_xlabel('Years in market')
        else:
            ax[0].set_xlabel('')
            ax[1].set_xlabel('')
            ax[2].set_xlabel('')
            
        ax[0].set_ylabel('log exports (1-yr spell = 0)')

    #axes[1][2].set_ylim(-1,5)
    #axes[2][1].set_ylim(-3,2.5)
    #axes[2][2].set_ylim(-3,2.5)

    fig.subplots_adjust(hspace=0.3,wspace=0.2)
    plt.sca(axes[0][0])
    plt.savefig(figpath + 'figA1_life_cycle_dyn_v_alpha_beta.pdf',bbox_inches='tight')
    plt.close('all')


#------------------------------------------------------
# Statistical significance table for appendix

file = open(figpath + 'life_cycle_dyn_v_data_tests.tex','w')
file.write('\\begin{table}[p]\n')
file.write('\\footnotesize\n')
file.write('\\renewcommand{\\arraystretch}{1.2}\n')
file.write('\\begin{center}\n')
file.write("\\caption{Effects of tenure $\times$ spell length in hard vs. easy destinations}\n")
file.write('\\label{tab:sumstats}\n')
file.write('\\begin{tabular}{cccccccccccc}')
file.write('\\toprule\n')
file.write('& & \\multicolumn{2}{c}{Effect on log sales} & \\multicolumn{2}{c}{H0 test: Hard > Easy} & \\multicolumn{6}{c}{Linear combo: Hard - Easy}\\\\\n')
file.write('\\cmidrule(rl){3-4}\\cmidrule(rl){5-6}\\cmidrule(rl){7-12}\n')
file.write('Spell & Tenure & Hard & Easy & F-stat & p-value & Coeff. & SE & t-stat & p-value & CI (lower) & CI (upper)\\\\\n')
file.write('\\midrule\n')

for k in range(1,max_tenure_scalar+1):
    for j in range(k+1):
        file.write('%d & %d' % (k+1,j+1))
        file.write('&%0.3f' % deffect_v_3way[0,k,j,0])
        file.write('&%0.3f' % deffect_v_3way[1,k,j,0])
        file.write('&%0.3g' % dreg_v_tests.loc[(dreg_v_tests.tenure==j) & (dreg_v_tests.max_tenure==k),"H0_Fstat"].values[0])
        file.write('&%0.3g' % dreg_v_tests.loc[(dreg_v_tests.tenure==j) & (dreg_v_tests.max_tenure==k),"H0_pval_1sided"].values[0])
        file.write('&%0.3g' % dreg_v_tests.loc[(dreg_v_tests.tenure==j) & (dreg_v_tests.max_tenure==k),"diff"].values[0])
        file.write('&%0.3g' % dreg_v_tests.loc[(dreg_v_tests.tenure==j) & (dreg_v_tests.max_tenure==k),"diff_se"].values[0])
        file.write('&%0.3g' % dreg_v_tests.loc[(dreg_v_tests.tenure==j) & (dreg_v_tests.max_tenure==k),"diff_t"].values[0])
        file.write('&%0.3g' % dreg_v_tests.loc[(dreg_v_tests.tenure==j) & (dreg_v_tests.max_tenure==k),"diff_p"].values[0])
        file.write('&%0.3g' % dreg_v_tests.loc[(dreg_v_tests.tenure==j) & (dreg_v_tests.max_tenure==k),"diff_lb"].values[0])
        file.write('&%0.3g' % dreg_v_tests.loc[(dreg_v_tests.tenure==j) & (dreg_v_tests.max_tenure==k),"diff_ub"].values[0])
        file.write('\\\\\n')

file.write('\\bottomrule\n')
file.write('\\end{tabular}\n')
file.write('\\end{center}\n')
file.write('\\normalsize\n')
file.write('\\end{table}\n')

file.close()


################################################################################
print('Processing export cost regression results...')

sreg_c_all = pd.read_stata(outpath + 'stata/sreg_c_all.dta').set_index('var')
sreg_c_3way = pd.read_stata(outpath + 'stata/sreg_c_3way.dta').set_index('var')

sreg_c2_all = pd.read_stata(outpath + 'stata/sreg_c2_all.dta').set_index('var')
sreg_c2_3way = pd.read_stata(outpath + 'stata/sreg_c2_3way.dta').set_index('var')

seffect_c_all = np.zeros((max_tenure_scalar+1,max_tenure_scalar+1))
seffect_c2_all = np.zeros((max_tenure_scalar+1,max_tenure_scalar+1))
for k in range(1,max_tenure_scalar+1):
    for j in range(k):

        strx=''
        if j==0:
            strx = '0b.tenure#'+str(k)+'.max_tenure'
        else:
            strx = str(j)+'.tenure#'+str(k)+'.max_tenure'
            
        seffect_c_all[k,j] = sreg_c_all.coef[strx]
        seffect_c2_all[k,j] = sreg_c2_all.coef[strx]

seffect_c_3way = np.zeros((2,max_tenure_scalar+1,max_tenure_scalar+1))
seffect_c2_3way = np.zeros((2,max_tenure_scalar+1,max_tenure_scalar+1))
for i in range(2):
    for k in range(1,max_tenure_scalar+1):
        for j in range(k):

            strx=''
            stry=''
            if j==0:
                strx = '0b.tenure#'+str(k)+'.max_tenure#'+grps[i]
                stry = '0b.tenure#'+str(k)+'.max_tenure'
            else:
                strx = str(j)+'.tenure#'+str(k)+'.max_tenure#'+grps[i]
                stry = str(j)+'.tenure#'+str(k)+'.max_tenure'
            
            seffect_c_3way[i,k,j] = sreg_c_3way.coef[strx]
            seffect_c2_3way[i,k,j] = sreg_c2_3way.coef[strx]


print('Making plots of tenure#max_tenure effects on export costs...')

sz = (7.5,5)
fig,axes=plt.subplots(2,3,figsize=sz,sharex=True,sharey=False)

for k in range(1,max_tenure_scalar+1):

    tenure = [x+1 for x in range(k)]

    axes[0][0].plot(tenure,seffect_c_all[k,:k],
                     color=colors[k-1],
                     alpha=0.5,
                     marker='o',
                     linewidth=1,
                     markersize=3,
                     #capsize=3,
                     linestyle='-',
                     label='Spell length = %d'%(k+1))
    
    axes[0][1].plot(tenure,seffect_c_3way[0,k,:k],
                     color=colors[k-1],
                     marker='o',
                     alpha=0.5,
                     markersize=3,
                     #capsize=3,
                     linewidth=1,
                     linestyle='-',
                     label='Spell length = %d'%(k+1))

    axes[0][2].plot(tenure,seffect_c_3way[1,k,:k],
                     color=colors[k-1],
                     marker='o',
                     alpha=0.5,
                     markersize=3,
                     #capsize=3,
                     linewidth=1,
                     linestyle='-',
                     label='Spell length = %d'%(k+1))
    
    axes[1][0].plot(tenure,seffect_c2_all[k,:k],
                     color=colors[k-1],
                     alpha=0.5,
                     marker='o',
                     linewidth=1,
                     markersize=3,
                     #capsize=3,
                     linestyle='-',
                     label='Spell length = %d'%(k+1))
    
    axes[1][1].plot(tenure,seffect_c2_3way[0,k,:k],
                     color=colors[k-1],
                     marker='o',
                     alpha=0.5,
                     markersize=3,
                     #capsize=3,
                     linewidth=1,
                     linestyle='-',
                     label='Spell length = %d'%(k+1))

    axes[1][2].plot(tenure,seffect_c2_3way[1,k,:k],
                     color=colors[k-1],
                     marker='o',
                     alpha=0.5,
                     markersize=3,
                     #capsize=3,
                     linewidth=1,
                     linestyle='-',
                     label='Spell length = %d'%(k+1))

axes[0][2].legend(loc='lower right',prop={'size':6})
axes[0][0].set_xticks(range(1,max_tenure_scalar+2))
axes[0][1].set_xticks(range(1,max_tenure_scalar+2))
axes[0][2].set_xticks(range(1,max_tenure_scalar+2))
axes[1][0].set_xlabel('Years in market')
axes[1][1].set_xlabel('Years in market')
axes[1][2].set_xlabel('Years in market')
axes[0][0].set_title('(a) Log cost: all markets',y=1.025)
axes[0][1].set_title('(b) Log cost: hard markets',y=1.025)
axes[0][2].set_title('(c) Log cost: easy markets',y=1.025)
axes[1][0].set_title('(d) Cost/profits: all markets',y=1.025)
axes[1][1].set_title('(e) Cost/profits: hard markets',y=1.025)
axes[1][2].set_title('(f) Cost/profits: easy markets',y=1.025)

axes[0][0].set_ylim(0,2)
axes[0][1].set_ylim(0,2)
axes[0][2].set_ylim(0,2)

axes[1][0].set_ylim(-0.5,0.2)
axes[1][1].set_ylim(-0.5,0.2)
axes[1][2].set_ylim(-0.5,0.2)

axes[0][0].set_ylabel('log cost (1-yr spell = 0)')
axes[1][0].set_ylabel('cost/profits (1-yr spell = 0)')

fig.subplots_adjust(hspace=0.25,wspace=0.2)
plt.sca(axes[0][0])
plt.savefig(figpath + 'fig6_life_cycle_dyn_c_model.pdf',bbox_inches='tight')
plt.close('all')
