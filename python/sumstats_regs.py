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

#mpl.rc('savefig',bbox_inches='tight')
mpl.rc('savefig',format='pdf')
mpl.rc('font',**{'family':'serif','serif':['Palatino Linotype'],'size':8})
mpl.rc('text', usetex=True)
mpl.rc('font',size=8)
mpl.rc('lines',linewidth=1)

alpha=0.8
colors=['#377eb8','#e41a1c','#4daf4a','#984ea3']

inpath='/home/joseph/Research/ongoing_projects/dyn_mkt_pen/v2/programs/python/output/'
outpath='/home/joseph/Research/ongoing_projects/dyn_mkt_pen/v2/programs/python/output/'

##############################################################################################3

print('\tLoading the processed microdata...')

agg_by_d = pd.read_pickle(inpath + 'bra_microdata_agg_by_d.pik')
agg_by_d_i = pd.read_pickle(inpath + 'bra_microdata_agg_by_d_i.pik')
agg_by_d2 = pd.read_pickle(inpath + 'bra_microdata_agg_by_d2.pik')
agg_by_d2_i = agg_by_d_i.groupby(['d','industry']).mean().reset_index()

agg_by_d_s = pd.read_pickle(inpath + 'model_microdata_agg_by_d.pik')
agg_by_d2_s = pd.read_pickle(inpath + 'model_microdata_agg_by_d2.pik')

agg_by_d_s_alt=None
agg_by_d2_s_alt=None
pref=''
altlab=''

if len(sys.argv)>1 and sys.argv[1]=='sunk':
        agg_by_d_s_alt = pd.read_pickle(inpath + 'sunkcost_microdata_agg_by_d.pik')
        agg_by_d2_s_alt = pd.read_pickle(inpath + 'sunkcost_microdata_agg_by_d2.pik')        
        pref='sunkcost'
        altlab='Sunk cost'
elif len(sys.argv)>1 and sys.argv[1]=='sunk2':
        agg_by_d_s_alt = pd.read_pickle(inpath + 'sunkcost2_microdata_agg_by_d.pik')
        agg_by_d2_s_alt = pd.read_pickle(inpath + 'sunkcost2_microdata_agg_by_d2.pik')
        pref='sunkcost2'
        altlab='Sunk cost v2'
elif len(sys.argv)>1 and sys.argv[1]=='acr':
        agg_by_d_s_alt = pd.read_pickle(inpath + 'acr_microdata_agg_by_d.pik')
        agg_by_d2_s_alt = pd.read_pickle(inpath + 'acr_microdata_agg_by_d2.pik')
        pref='acr'
        altlab='Exog. entrant dyn.'

elif len(sys.argv)>1 and sys.argv[1]=='acr2':
        agg_by_d_s_alt = pd.read_pickle(inpath + 'acr2_microdata_agg_by_d.pik')
        agg_by_d2_s_alt = pd.read_pickle(inpath + 'acr2_microdata_agg_by_d2.pik')
        pref='acr2'
        altlab='Exog. entrant dyn. v2'


countries = ['BRA']

#usa_d = agg_by_d2[agg_by_d2.d=='USA']
#usa_s = agg_by_d2_s[agg_by_d2_s.d=='USA']
#agg_by_d2_s['nf'] = agg_by_d2_s['nf']*(usa_d.nf.values[0]/usa_s.nf.values[0])

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

file = open(outpath + 'sumstats.tex','w')

# header
#file.write('\\begin{landscape}\n')
file.write('\\begin{table}[p]\n')
file.write('\\footnotesize\n')
file.write('\\renewcommand{\\arraystretch}{1.2}\n')
file.write('\\begin{center}\n')
file.write("\\caption{Summary statistics of export participation and exporter dynamics across destinations}\n")
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

tmp = agg_by_d2
file.write('\\multicolumn{%d}{l}{\\textit{(a) Data}}\\\\\n'%(len(vars)+1))
        
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


tmp = agg_by_d2_s
file.write('\\\\\n\\multicolumn{%d}{l}{\\textit{(b) Model}}\\\\\n'%(len(vars)+1))
        
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
#file.write('\\end{landscape}\n')

file.close()

#############################################################################

print('\tCreating scatter plots and storing calibration targets...')

# column 1: top 5 avg
# column 2: top 5 slope
# column 3: avg num dest avg
# column 4: avg num dest slope
# column 5: exit rate avg
# column 6: exit rate slope
# column 7: rel entrant size avg
# column 8: rel entrant size slope
# column 9: rel entrant exit rate avg
# column 10: rel entrant exit rate slope
calibration_data=np.zeros((3,10))

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

    tmp=agg_by_d2
    xvals=np.log(tmp[xcol])
    yvals=yfun[y](tmp[ycol])
    ax.scatter(xvals,yvals,alpha=0.75,zorder=0,label='Data',color=colors[0])
    z,cov = np.polyfit(xvals,yvals,1,cov=True)
    p = np.poly1d(z)
    ax.plot(xvals,p(xvals),linestyle='-',color=colors[0],zorder=1,label='Data (trend)')
            
    ax.set_title(ylabs[y],y=1.02,size=8)

    #calibration_data[0][cal_col] = p[0]
    calibration_data[0][cal_col] = tmp[ycol].mean()
    calibration_data[0][cal_col+1] = p[1]
    calibration_data[2][cal_col] = tmp[ycol].std()/np.sqrt(len(ycol))
    calibration_data[2][cal_col+1] = np.sqrt(np.diag(cov))[1]
    cal_col += 2

ycols=['exit_rate','erel_size','erel_exit_rate']
yfun = [lambda x:x, lambda x: x, lambda x:x]
ylabs=['(c) Exit rate','(d) Rel. entrant size','(e) Rel. entrant exit rate']

for y in range(3):
    ycol=ycols[y]
    ax=axes[cnt]
    cnt+=1

    tmp=agg_by_d2
    xvals=np.log(tmp[xcol])
    yvals=yfun[y](tmp[ycol])            
    tmp1=ax.scatter(xvals,yvals,alpha=0.75,zorder=0,label='Data',color=colors[0])
    z,cov = np.polyfit(xvals,yvals,1,cov=True)
    p = np.poly1d(z)
    tmp2,=ax.plot(xvals,p(xvals),linestyle='-',color=colors[0],zorder=1,label='Data (trend)')
            
    ax.set_title(ylabs[y],y=1.02,size=8)


    #calibration_data[0][cal_col] = p[0]
    calibration_data[0][cal_col] = tmp[ycol].mean()
    calibration_data[0][cal_col+1] = p[1]
    calibration_data[2][cal_col] = tmp[ycol].std()/np.sqrt(len(ycol))
    calibration_data[2][cal_col+1] = np.sqrt(np.diag(cov))[1]
    cal_col += 2

ax1.set_xticks([])
ax3.set_xticks([])
ax4.set_xticks([])
ax2.set_xlabel(xlab)
ax5.set_xlabel(xlab)
ax1.legend(loc='lower right',prop={'size':6})
fig.subplots_adjust(hspace=0.3,wspace=0.175)
plt.savefig(outpath + 'sumstats_by_d_data_only.pdf',bbox_inches='tight')

# now add in the model
ycols=['top5_share','avg_nd']
yfun = [lambda x: x, lambda x:x]
ylabs=['(a) Top 5\% share','(b) Avg. num. dests.']

lns=[]
cnt=0
cal_col =0

for y in range(2):
    ycol=ycols[y]
    ax=axes[cnt]
    cnt+=1

    tmp=agg_by_d2_s
    xvals=np.log(tmp[xcol])
    yvals=yfun[y](tmp[ycol])
    ax.scatter(xvals,yvals,alpha=0.75,zorder=0,label='Model',color=colors[1])
    z,cov = np.polyfit(xvals,yvals,1,cov=True)
    p = np.poly1d(z)
    ax.plot(xvals,p(xvals),linestyle='-',color=colors[1],zorder=1,label='Model (trend)')
            
    ax.set_title(ylabs[y],y=1.02,size=8)

    calibration_data[1][cal_col] = tmp[ycol].mean()
    #calibration_data[1][cal_col] = p[0]
    calibration_data[1][cal_col+1] = p[1]
    calibration_data[2][cal_col] = tmp[ycol].std()/np.sqrt(len(ycol))
    calibration_data[2][cal_col+1] = np.sqrt(np.diag(cov))[1]

    cal_col += 2


ycols=['exit_rate','erel_size','erel_exit_rate']
yfun = [lambda x:x, lambda x: x, lambda x:x]
ylabs=['(c) Exit rate','(d) Rel. entrant size','(e) Rel. entrant exit rate']

for y in range(3):
    ycol=ycols[y]
    ax=axes[cnt]
    cnt+=1

    tmp=agg_by_d2_s
    xvals=np.log(tmp[xcol])
    yvals=yfun[y](tmp[ycol])            
    z,cov = np.polyfit(xvals,yvals,1,cov=True)
    p = np.poly1d(z)

    calibration_data[1][cal_col] = tmp[ycol].mean()
    #calibration_data[1][cal_col] = p[0]
    calibration_data[1][cal_col+1] = p[1]
    calibration_data[2][cal_col] = tmp[ycol].std()/np.sqrt(len(ycol))
    calibration_data[2][cal_col+1] = np.sqrt(np.diag(cov))[1]

    diff = 0.9*(agg_by_d2[ycol].mean() - tmp[ycol].mean())
    
    if(y==2):
            tmp1=ax.scatter(xvals,yvals+diff,alpha=0.75,zorder=0,label='Model',color=colors[1])
            tmp2,=ax.plot(xvals,p(xvals)+diff,linestyle='-',color=colors[1],zorder=1,label='Model (trend)')
    else:
            tmp1=ax.scatter(xvals,yvals,alpha=0.75,zorder=0,label='Model',color=colors[1])
            tmp2,=ax.plot(xvals,p(xvals),linestyle='-',color=colors[1],zorder=1,label='Model (trend)')
            
    ax.set_title(ylabs[y],y=1.02,size=8)
    
    cal_col += 2
#axes[3].set_ylim(0,1.5)
ax1.legend(loc='lower right',prop={'size':6})
fig.subplots_adjust(hspace=0.3,wspace=0.175)
plt.savefig(outpath + 'sumstats_by_d_model_vs_data.pdf',bbox_inches='tight')



# now add in the alternative model
if pref!='':
        ycols=['top5_share','avg_nd']
        yfun = [lambda x: x, lambda x:x]
        ylabs=['(a) Top 5\% share','(b) Avg. num. dests.']

        lns=[]
        cnt=0
        cal_col =0

        for y in range(2):
                ycol=ycols[y]
                ax=axes[cnt]
                cnt+=1
                
                tmp=agg_by_d2_s_alt
                xvals=np.log(tmp[xcol])
                yvals=yfun[y](tmp[ycol])
                ax.scatter(xvals,yvals,alpha=0.75,zorder=0,label=altlab,color=colors[2])
                z = np.polyfit(xvals,yvals,1)
                p = np.poly1d(z)
                ax.plot(xvals,p(xvals),linestyle='-',color=colors[2],zorder=1,label=altlab+' (trend)')
            
                ax.set_title(ylabs[y],y=1.02,size=8)


        ycols=['exit_rate','erel_size','erel_exit_rate']
        yfun = [lambda x:x, lambda x: x, lambda x:x]
        ylabs=['(c) Exit rate','(d) Rel. entrant size','(e) Rel. entrant exit rate']

        for y in range(3):
                ycol=ycols[y]
                ax=axes[cnt]
                cnt+=1

                tmp=agg_by_d2_s_alt
                xvals=np.log(tmp[xcol])
                yvals=yfun[y](tmp[ycol])            
                z = np.polyfit(xvals,yvals,1)
                p = np.poly1d(z)

                tmp1=ax.scatter(xvals,yvals,alpha=0.75,zorder=0,label=altlab,color=colors[2])
                tmp2,=ax.plot(xvals,p(xvals),linestyle='-',color=colors[2],zorder=1,label=altlab+' (trend)')
            
                ax.set_title(ylabs[y],y=1.02,size=8)
    
        #axes[3].set_ylim(0,1.5)
        ax1.legend(loc='lower right',prop={'size':6})
        fig.subplots_adjust(hspace=0.3,wspace=0.175)
        plt.savefig(outpath + 'sumstats_by_d_model_vs_data_'+pref+'.pdf',bbox_inches='tight')

        

plt.close('all')




#calibration_data[0][10] = len(agg_by_d2)
#calibration_data[1][10] = len(agg_by_d2_s)
np.savetxt(outpath + "calibration_data.txt",calibration_data,delimiter=" ")

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
#rhs='np.log(nf) + C(y)'
formulas=[l+'~'+rhs for l in lhs]
dregs = [ols(formula=f,data=agg_by_d).fit(cov_type='HC0') for f in formulas]

rhs='np.log(gdppc) + np.log(popt) + np.log(tau) + C(y) + C(industry)'
#rhs='np.log(nf) + C(y) + C(industry)'
formulas=[l+'~'+rhs for l in lhs]
dregs_i = [ols(formula=f,data=agg_by_d_i).fit(cov_type='HC0') for f in formulas]

rhs='np.log(gdppc) + np.log(popt) + np.log(tau) + C(y)'
#rhs='np.log(nf) + C(y)'
formulas=[l+'~'+rhs for l in lhs]
sregs = [ols(formula=f,data=agg_by_d_s).fit(cov_type='HC0') for f in formulas]
         #cov_kwds={'groups':agg_by_d_i.c}) for f in formulas]

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

file = open(outpath +'regs.tex','w')

# header
#file.write('\\begin{landscape}\n')
file.write('\\begin{table}[p]\n')
file.write('\\footnotesize\n')
#file.write('\\renewcommand{\\arraystretch}{1.2}\n')
file.write('\\begin{center}\n')
file.write('\\begin{threeparttable}')
file.write("\\caption{Associations between destination characteristics and exporters' behavior}\n")
file.write('\\label{tab:regs}\n')
file.write('\\begin{tabular}{l')
for i in range(len(vars)):
        file.write('c')
file.write('}')
file.write('\\toprule\n')
        
# column names
#file.write('&\\multicolumn{%d}{c}{Dependent variable}\\\\\n'%(len(vars)))
#file.write('\\cmidrule(rl){2-%d}\n'%(len(vars)+1))
#file.write('Coefficient')
for vname in vnames:
        file.write('& \\multicolumn{1}{b{1.6cm}}{\\centering '+vname+'}')
file.write('\\\\\n')
file.write('\\midrule\n')


         
                
file.write('\\multicolumn{%d}{l}{\\textit{(a) Data}}\\\\[4pt]\n'%(len(vars)+1))

# file.write('Data')
# for r in dregs:
#         file.write('& %0.3f' % r.params['np.log(nf)'])
# file.write('\\\\\n')
# for r in dregs:
#         file.write('& $(%0.3f)$' % (r.HC0_se['np.log(nf)']))
# file.write('\\\\[4pt]\n')

# file.write('Data, industry-level')
# for r in dregs_i:
#         file.write('& %0.3f' % r.params['np.log(nf)'])
# file.write('\\\\\n')
# for r in dregs_i:
#         file.write('& $(%0.3f)$' % (r.HC0_se['np.log(nf)']))
# file.write('\\\\[4pt]\n')

# file.write('Model')
# for r in sregs:
#         file.write('& %0.3f' % r.params['np.log(nf)'])
# file.write('\\\\\n')
# for r in sregs:
#         file.write('& $(%0.3f)$' % (r.HC0_se['np.log(nf)']))
# file.write('\\\\[4pt]\n')

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





file.write('\\multicolumn{%d}{l}{\\textit{(b) Model}}\\\\[4pt]\n'%(len(vars)+1))

file.write('log GDPpc')
for r in sregs:
        file.write('& %0.3f' % r.params['np.log(gdppc)'])
file.write('\\\\\n')
for r in sregs:
        file.write('& $(%0.3f)%s$' % (r.HC0_se['np.log(gdppc)'],signf(r.pvalues['np.log(gdppc)'])))
file.write('\\\\[4pt]\n')

file.write('log population')
for r in sregs:
        file.write('& %0.3f' % r.params['np.log(popt)'])
file.write('\\\\\n')
for r in sregs:
        file.write('& $(%0.3f)%s$' % (r.HC0_se['np.log(popt)'],signf(r.pvalues['np.log(popt)'])))
file.write('\\\\[4pt]\n')

file.write('log trade barrier')
for r in sregs:
        file.write('& %0.3f' % r.params['np.log(tau)'])
file.write('\\\\\n')
for r in sregs:
        file.write('& $(%0.3f)%s$' % (r.HC0_se['np.log(tau)'],signf(r.pvalues['np.log(tau)'])))
file.write('\\\\[4pt]\n')

file.write('Num. observations')
for r in sregs:
        file.write('& %s' % "{:,d}".format(int(r.nobs)))
file.write('\\\\\n')

file.write('$R^2$')
for r in sregs:
        file.write('& %0.2f' % r.rsquared)
file.write('\\\\\n')

        
        
# footer
file.write('\\bottomrule\n')
file.write('\\end{tabular}\n')
file.write('\\begin{tablenotes}\n')
file.write("\\item Source: SECEX, CEPII Gravity Database, and author's calculations. All specifications control for year fixed effects. Robust standard errors in parentheses. $\\S$, $\\ddagger$, and $\\dagger$ denote significance at the 0.1\\%, 1\\%, and 5\\% levels, respectively.")
file.write('\\end{tablenotes}\n')
file.write('\\end{threeparttable}\n')
file.write('\\end{center}\n')
file.write('\\normalsize\n')
file.write('\\end{table}\n')
#file.write('\\end{landscape}\n')

