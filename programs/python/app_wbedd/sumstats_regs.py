import sys
import numpy as np
import pandas as pd
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
mpl.rc('savefig',format='pdf')
mpl.rc('font',**{'family':'serif','serif':['Palatino'],'size':8})
mpl.rc('font',size=8)
mpl.rc('lines',linewidth=1)

alpha=0.8
colors=['#377eb8','#e41a1c','#4daf4a','#984ea3']

# column 1: coefficient of variation (nf)
# column 2: top 5 avg
# column 3: avg num dest avg
# column 4: exit rate avg
# column 5: rel entrant size avg
# column 6: rel entrant exit rate avg
calibration_data=np.zeros((3,6+18))

##############################################################################################3

print('\tLoading the processed microdata...')

agg_by_d = pd.read_pickle('output/wbedd_microdata_agg_by_d.pik')
agg_by_d2 = pd.read_pickle('output/wbedd_microdata_agg_by_d2.pik')
countries = ['MEX','PER']

#agg_by_d_m = agg_by_d.loc[(agg_by_d.c=='MEX') & (agg_by_d.d != 'USA'),:].reset_index(drop=True)
agg_by_d_m = agg_by_d.loc[agg_by_d.c=='MEX',:].reset_index(drop=True)
agg_by_d_p = agg_by_d.loc[agg_by_d.c=='PER',:].reset_index(drop=True)

#agg_by_d2_m = agg_by_d2.loc[(agg_by_d2.c=='MEX') & (agg_by_d2.d != 'USA'),:].reset_index(drop=True)
agg_by_d2_m = agg_by_d2.loc[agg_by_d2.c=='MEX',:].reset_index(drop=True)
agg_by_d2_p = agg_by_d2.loc[agg_by_d2.c=='PER',:].reset_index(drop=True)

##############################################################################################3

print('\tWriting summary statistics to LaTeX table')

vars=['nf',
     'top5_share',
     'avg_nd',
     'exit_rate',
     'erel_size',
     'erel_exit_rate']

vnames=['Num. exporters',
        'Top-5 share',
        'Avg. num. dests.',
        'Exit rate',
        'Entrant rel. size',
        'Entrant rel. exit rate']

fmt=['%d'] + ['%0.2f' for x in vars[1:]]

panels = ['a','b','c','d','e','f','g','h']

file = open('output/sumstats_wbedd.tex','w')

# header
#file.write('\\begin{landscape}\n')
file.write('\\begin{table}[p]\n')
file.write('\\begin{center}\n')
file.write('\\footnotesize\n')
file.write('\\renewcommand{\\arraystretch}{1.2}\n')
file.write("\\caption{Summary statistics of export participation and exporter dynamics across destinations (Mexico and Peru)}\n")
file.write('\\label{tab:sumstats}\n')
file.write('\\begin{tabular}{l')
for i in range(len(vars)):
        file.write('c')
file.write('}')
file.write('\\toprule\n')

# column names
file.write('Statistic')
for vname in vnames:
        file.write('& \\multicolumn{1}{b{1.6cm}}{\\centering '+vname+'}')
file.write('\\\\\n')
file.write('\\midrule\n')

tmp = agg_by_d2_m
file.write('\\multicolumn{%d}{l}{\\textit{(a) Mexico}}\\\\\n'%(len(vars)+1))

file.write('Mean')
for v,f in zip(vars,fmt):
        file.write('& %s' % locale.format_string(f,tmp[v].mean(),grouping=True))
file.write('\\\\\nMin')
for v,f in zip(vars,fmt):
        file.write('& %s' % locale.format_string(f,tmp[v].min(),grouping=True))
                
file.write('\\\\\nMax')
for v,f in zip(vars,fmt):
        file.write('& %s' % locale.format_string(f,tmp[v].max(),grouping=True))

file.write('\\\\\nStd. dev.')
for v,f in zip(vars,fmt):
        file.write('& %s' % locale.format_string(f,tmp[v].std(),grouping=True))
        
file.write('\\\\\n')

tmp = agg_by_d2_p
file.write('\\\\\n\\multicolumn{%d}{l}{\\textit{(b) Peru}}\\\\\n'%(len(vars)+1))
        
file.write('Mean')
for v,f in zip(vars,fmt):
        file.write('& %s' % locale.format_string(f,tmp[v].mean(),grouping=True))
file.write('\\\\\nMin')
for v,f in zip(vars,fmt):
        file.write('& %s' % locale.format_string(f,tmp[v].min(),grouping=True))
                
file.write('\\\\\nMax')
for v,f in zip(vars,fmt):
        file.write('& %s' % locale.format_string(f,tmp[v].max(),grouping=True))

file.write('\\\\\nStd. dev.')
for v,f in zip(vars,fmt):
        file.write('& %s' % locale.format_string(f,tmp[v].std(),grouping=True))

file.write('\\\\\n')


# footer
file.write('\\bottomrule\n')
file.write('\\end{tabular}\n')
file.write('\\end{center}\n')
file.write('\\normalsize\n')
file.write('\\end{table}\n')

file.close()

#############################################################################

print('\tCreating scatter plots and storing calibration targets...')

fig = plt.figure(figsize=(6.5,5.5))
ax1 = fig.add_subplot(221)
ax2 = fig.add_subplot(223,sharex=ax1)
ax3 = fig.add_subplot(322,sharex=ax1)
ax4 = fig.add_subplot(324,sharex=ax1)
ax5 = fig.add_subplot(326,sharex=ax1)
axes=[ax1,ax2,ax3,ax4,ax5]

xcol = 'nf'
xlab='Number of exporters (log)'

# first just plot the data
ycols=['top5_share','avg_nd']
yfun = [lambda x: x, lambda x:x]
ylabs=['(a) Top 5\% share','(b) Avg. num. dests.']

lns=[]
cnt=0
cal_col=0

for y in range(2):
    ycol=ycols[y]
    ax=axes[cnt]
    cnt+=1

    tmp=agg_by_d2_m
    xvals=np.log(tmp[xcol])
    yvals=yfun[y](tmp[ycol])
    ax.scatter(xvals,yvals,alpha=0.75,zorder=0,label='MEX',color=colors[0])
    z,cov = np.polyfit(xvals,yvals,1,cov=True)
    p = np.poly1d(z)
    ax.plot(xvals,p(xvals),linestyle='-',color=colors[0],zorder=1,label='MEX (trend)')

    tmp=agg_by_d2_p
    xvals=np.log(tmp[xcol])
    yvals=yfun[y](tmp[ycol])
    ax.scatter(xvals,yvals,alpha=0.75,zorder=0,label='PER',color=colors[1])
    z,cov = np.polyfit(xvals,yvals,1,cov=True)
    p = np.poly1d(z)
    ax.plot(xvals,p(xvals),linestyle='-',color=colors[1],zorder=1,label='PER (trend)')
    
    ax.set_title(ylabs[y],y=1.02,size=8)

ycols=['exit_rate','erel_size','erel_exit_rate']
yfun = [lambda x:x, lambda x: x, lambda x:x]
ylabs=['(c) Exit rate','(d) Rel. entrant size','(e) Rel. entrant exit rate']

for y in range(3):
    ycol=ycols[y]
    ax=axes[cnt]
    cnt+=1

    tmp=agg_by_d2_m
    xvals=np.log(tmp[xcol])
    yvals=yfun[y](tmp[ycol])            
    tmp1=ax.scatter(xvals,yvals,alpha=0.75,zorder=0,label='MEX',color=colors[0])
    z,cov = np.polyfit(xvals,yvals,1,cov=True)
    p = np.poly1d(z)
    tmp2,=ax.plot(xvals,p(xvals),linestyle='-',color=colors[0],zorder=1,label='MEX (trend)')

    tmp=agg_by_d2_p
    xvals=np.log(tmp[xcol])
    yvals=yfun[y](tmp[ycol])            
    tmp1=ax.scatter(xvals,yvals,alpha=0.75,zorder=0,label='PER',color=colors[1])
    z,cov = np.polyfit(xvals,yvals,1,cov=True)
    p = np.poly1d(z)
    tmp2,=ax.plot(xvals,p(xvals),linestyle='-',color=colors[1],zorder=1,label='PER (trend)')

    ax.set_title(ylabs[y],y=1.02,size=8)

ax1.set_xticks([])
ax3.set_xticks([])
ax4.set_xticks([])
ax2.set_xlabel(xlab)
ax5.set_xlabel(xlab)
ax1.legend(loc='lower right',prop={'size':6})
fig.subplots_adjust(hspace=0.3,wspace=0.175)
plt.savefig('output/sumstats_by_d_wbedd.pdf',bbox_inches='tight')
       

plt.close('all')



#############################################################################

print('\tEstimating relationships between destination characteristics and firm-level variables')

lhs=['np.log(nf)',
     'top5_share',
     'avg_nd',
     'exit_rate',
     'erel_size',
     'erel_exit_rate']

vnames=['Log num. exporters',
        'Top-5 share',
        'Avg. num. dests.',
        'Exit rate',
        'Entrant rel. size',
        'Entrant rel. exit rate']


rhs='np.log(gdppc) + np.log(popt) + np.log(tau) + C(y)'

formulas=[l+'~'+rhs for l in lhs]
dregs_m = [ols(formula=f,data=agg_by_d_m).fit(cov_type='cluster',cov_kwds={'groups':agg_by_d_m['d']}) for f in formulas]
dregs_p = [ols(formula=f,data=agg_by_d_p).fit(cov_type='cluster',cov_kwds={'groups':agg_by_d_p['d']}) for f in formulas]

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

print('\tWriting LaTeX table')

file = open('output/regs_wbedd.tex','w')

# header
#file.write('\\begin{landscape}\n')
file.write('\\begin{table}[p]\n')
file.write('\\footnotesize\n')
#file.write('\\renewcommand{\\arraystretch}{1.2}\n')
file.write('\\begin{center}\n')
file.write("\\caption{Associations between destination characteristics and exporters' behavior (Mexico and Peru)}\n")
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
        file.write('& $(%0.3f)%s$' % (r.bse['np.log(gdppc)'],signf(r.pvalues['np.log(gdppc)'])))
file.write('\\\\[4pt]\n')

file.write('log population')
for r in dregs:
        file.write('& %0.3f' % r.params['np.log(popt)'])
file.write('\\\\\n')
for r in dregs:
        file.write('& $(%0.3f)%s$' % (r.bse['np.log(popt)'],signf(r.pvalues['np.log(popt)'])))
file.write('\\\\[4pt]\n')

file.write('log trade barrier')
for r in dregs:
        file.write('& %0.3f' % r.params['np.log(tau)'])
file.write('\\\\\n')
for r in dregs:
        file.write('& $(%0.3f)%s$' % (r.bse['np.log(tau)'],signf(r.pvalues['np.log(tau)'])))
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
        file.write('& $(%0.3f)%s$' % (r.bse['np.log(gdppc)'],signf(r.pvalues['np.log(gdppc)'])))
file.write('\\\\[4pt]\n')

file.write('log population')
for r in dregs:
        file.write('& %0.3f' % r.params['np.log(popt)'])
file.write('\\\\\n')
for r in dregs:
        file.write('& $(%0.3f)%s$' % (r.bse['np.log(popt)'],signf(r.pvalues['np.log(popt)'])))
file.write('\\\\[4pt]\n')

file.write('log trade barrier')
for r in dregs:
        file.write('& %0.3f' % r.params['np.log(tau)'])
file.write('\\\\\n')
for r in dregs:
        file.write('& $(%0.3f)%s$' % (r.bse['np.log(tau)'],signf(r.pvalues['np.log(tau)'])))
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

