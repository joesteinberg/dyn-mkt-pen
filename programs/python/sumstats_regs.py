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

inpath='/home/linux/Documents/Research/dyn_mkt_pen/v3/programs/python/output/pik/'
outpath='/home/linux/Documents/Research/dyn_mkt_pen/v3/programs/python/output/sumstats_regs/'
calpath='/home/linux/Documents/Research/dyn_mkt_pen/v3/programs/python/output/calibration/'

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

##############################################################################################3

print('\tLoading the processed microdata...')

agg_by_d = pd.read_pickle(inpath + 'bra_microdata_agg_by_d.pik')
agg_by_d2 = pd.read_pickle(inpath + 'bra_microdata_agg_by_d2.pik')

models = ['model','smp','sunkcost','acr','abn1','abn2','abo1','abo2','a0']
agg_by_d_s = [pd.read_pickle(inpath + m+'_microdata_agg_by_d.pik') for m in models]
agg_by_d2_s = [pd.read_pickle(inpath + m+'_microdata_agg_by_d2.pik') for m in models]

countries = ['BRA']

##############################################################################################3

print('\tEstimating relationships between destination characteristics and firm-level variables')

lhs=['np.log(nf)',
     'top5_share',
     'avg_nd',
     'exit_rate',
     'erel_size',
     'erel_exit_rate']

vnames=['Log num. exporters',
        'Top-5 share',
        'Avg. num. markets',
        'Exit rate',
        'Entrant rel. size',
        'Entrant rel. exit rate']

rhs='np.log(gdppc) + np.log(popt) + np.log(tau) + C(y)'
formulas=[l+'~'+rhs for l in lhs]
dregs = [ols(formula=f,data=agg_by_d).fit(cov_type='cluster',cov_kwds={'groups':agg_by_d['d']}) for f in formulas]

rhs='np.log(gdppc) + np.log(popt) + np.log(tau) + C(y)'
formulas=[l+'~'+rhs for l in lhs]
sregs=[None for d in agg_by_d_s]
for i in range(len(sregs)):
        sregs[i] = [ols(formula=f,data=agg_by_d_s[i]).fit() for f in formulas]

##############################################################################################3

print('\tWriting LaTeX table for actual data')

vars=['nf',
      'top5_share',
      'avg_nd',
      'exit_rate',
      'erel_size',
      'erel_exit_rate']

vnames=['Num.\\\\exporters',
        'Top-5\\\\share',
        'Avg. num.\\\\dests.',
        'Exit\\\\rate',
        'Entrant\\\\rel. size',
        'Entrant rel.\\\\exit rate']

fmt=['%d'] + ['%0.2f' for x in vars[1:]]

panels = ['a','b','c','d','e','f','g','h']

file = open(outpath + 'table1_sumstats_regs.tex','w')

# header
file.write('\\begin{table}[h!]\n')
file.write('\\footnotesize\n')
file.write('\\renewcommand{\\arraystretch}{1.2}\n')
file.write('\\begin{center}\n')
file.write("\\caption{Market-level measures of exporter performance}\n")
file.write('\\label{tab:sumstats_regs}\n')
file.write('\\begin{tabular}{l')
for i in range(len(vars)):
        file.write('c')
file.write('}')
file.write('\\toprule\n')

# column names
file.write('Statistic/coefficient')
for vname in vnames:
        file.write('& \\makecell{'+vname+'}')
file.write('\\\\\n')
file.write('\\midrule\n')

tmp = agg_by_d2
file.write('\\multicolumn{%d}{l}{\\textit{(a) Summary statistics}}\\\\[4pt]\n'%(len(vars)+1))

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

file.write('\\midrule\\multicolumn{%d}{l}{\\textit{(b) Associations with market characteristics}}\\\\[4pt]\n'%(len(vars)+1))

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

# footer
file.write('\\bottomrule\n')
file.write('\\end{tabular}\n')
file.write('\\end{center}\n')
file.write('\\normalsize\n')
file.write('\\end{table}\n')

file.close()

##############################################################################################3
# calibration moments

# column 1: coefficient of variation (nf)
# column 2: top 5 avg
# column 3: avg num dest avg
# column 4: exit rate avg
# column 5: rel entrant size avg
# column 6: rel entrant exit rate avg
calibration_data=np.zeros((3,4*6))

calcol=0
n=len(agg_by_d2)
for v in vars[0:6]:
        if(v=='nf'):
                cv= agg_by_d2[v].std()/agg_by_d2[v].mean()
                calibration_data[0][calcol] = cv
                calibration_data[1][calcol] = agg_by_d2_s[0][v].std()/agg_by_d2_s[0][v].mean()
                calibration_data[2][calcol] = cv/np.sqrt(2*n) * np.sqrt(1+2*(cv/100)*(cv/100))
                calcol = calcol+1
        else:
                calibration_data[0][calcol] = agg_by_d2[v].mean()
                calibration_data[1][calcol] = agg_by_d2_s[0][v].mean()
                calibration_data[2][calcol] = agg_by_d2[v].std()/np.sqrt(n)
                calcol = calcol+1

        if(v=='avg_nd' or v=='exit_rate'):
                calibration_data[2][calcol] = calibration_data[2][calcol]/50
        elif(v=='top5_share'):
                calibration_data[2][calcol] = calibration_data[2][calcol]/300

for rd, rs in zip(dregs[0:6],sregs[0][0:6]):
        calibration_data[0][calcol] = rd.params['np.log(gdppc)']
        calibration_data[1][calcol] = rs.params['np.log(gdppc)']
        calibration_data[2][calcol] = rd.bse['np.log(gdppc)']
        calcol = calcol + 1
        
        calibration_data[0][calcol] = rd.params['np.log(popt)']
        calibration_data[1][calcol] = rs.params['np.log(popt)']
        calibration_data[2][calcol] = rd.bse['np.log(popt)']
        calcol = calcol + 1
        
        calibration_data[0][calcol] = rd.params['np.log(tau)']
        calibration_data[1][calcol] = rs.params['np.log(tau)']
        calibration_data[2][calcol] = rd.bse['np.log(tau)']
        calcol = calcol + 1


np.savetxt(calpath + "calibration_data.txt",calibration_data,delimiter=" ")

#############################################################################

print('\tWriting LaTeX table for model results')

model_names = ['Baseline model','Static mkt. pen. model','Sunk-cost model','Exog. exporter dyn. model']

file = open(outpath +'table3_model_results.tex','w')

# header
file.write('\\begin{table}[h!]\n')
file.write('\\footnotesize\n')
file.write('\\begin{center}\n')
file.write("\\caption{Exporter performance across markets in alternative models}\n")
file.write('\\label{tab:regs}\n')
file.write('\\begin{tabular}{l')
for i in range(len(vars)):
        file.write('c')
file.write('}')
file.write('\\toprule\n')

file.write('Statistic/coefficient')
for vname in vnames:
        file.write('& \\makecell{'+vname+'}')
file.write('\\\\\n')
file.write('\\midrule\n')       

panels = ['(b) Associations with destination characteristics: Baseline model',
          '(c) Associations with destination characteristics: Static mkt. pen. model',
          '(d) Associations with destination characteristics: Sunk-cost model',
          '(e) Associations with destination characteristics: Exog. exporter dyn. model']

file.write('\\multicolumn{%d}{l}{\\textit{(a) Cross-market averages}}\\\\[4pt]\n'%(len(vars)+1))
cnt=0
for tmp in agg_by_d2_s[0:4]:
        file.write(model_names[cnt])
        cnt += 1
        for v,f in zip(vars,fmt):
                file.write('& %s' % locale.format_string(f,tmp[v].mean(),grouping=True))
        file.write('\\\\\n')

for i in range(len(sregs[0:4])):
        file.write('\\midrule\\multicolumn{%d}{l}{\\textit{%s}}\\\\[4pt]\n'%((len(vars)+1),panels[i]))

        file.write('log GDPpc')
        for r in sregs[i]:
                file.write('& %0.3f' % r.params['np.log(gdppc)'])
        file.write('\\\\\n')
        
        file.write('log population')
        for r in sregs[i]:
                file.write('& %0.3f' % r.params['np.log(popt)'])
        file.write('\\\\\n')
        
        file.write('log trade barrier')
        for r in sregs[i]:
                file.write('& %0.3f' % r.params['np.log(tau)'])
        file.write('\\\\\n')        
        
# footer
file.write('\\bottomrule\n')
file.write('\\end{tabular}\n')
file.write('\\end{center}\n')
file.write('\\normalsize\n')
file.write('\\end{table}\n')

file.close()


#############################################################################

print('\tWriting LaTeX table with sensitivity analysis results')

model_names = ['$\\alpha_n=\\beta_n$','$\\beta_n=\\alpha_n$','$\\alpha_o=\\beta_o$','$\\beta_o=\\alpha_o$','$\\alpha_n=\\alpha_o=1$']

file = open(outpath +'table_A2_alpha_beta.tex','w')

# header
file.write('\\begin{table}[h!]\n')
file.write('\\footnotesize\n')
file.write('\\begin{center}\n')
file.write("\\caption{Exporter performance across markets: sensitivity analysis}\n")
file.write('\\label{tab:regs}\n')
file.write('\\begin{tabular}{l')
for i in range(len(vars)):
        file.write('c')
file.write('}')
file.write('\\toprule\n')

file.write('Statistic/coefficient')
for vname in vnames:
        file.write('& \\makecell{'+vname+'}')
file.write('\\\\\n')
file.write('\\midrule\n')       

panels = ['(b) Associations with destination characteristics: $\\alpha_n=\\beta_n$',
          '(c) Associations with destination characteristics: $\\beta_n = \\alpha_n$',
          '(d) Associations with destination characteristics: $\\alpha_o=\\beta_o$',
          '(e) Associations with destination characteristics: $\\beta_o = \\alpha_o$',
          '(f) Associations with destination characteristics: $\\alpha_n=\\alpha_o=1$']

file.write('\\multicolumn{%d}{l}{\\textit{(a) Cross-market averages}}\\\\[4pt]\n'%(len(vars)+1))
cnt=0
for tmp in agg_by_d2_s[4:]:
        file.write(model_names[cnt])
        cnt += 1
        for v,f in zip(vars,fmt):
                file.write('& %s' % locale.format_string(f,tmp[v].mean(),grouping=True))
        file.write('\\\\\n')

for i in range(len(sregs[4:])):
        file.write('\\midrule\\multicolumn{%d}{l}{\\textit{%s}}\\\\[4pt]\n'%((len(vars)+1),panels[i]))

        file.write('log GDPpc')
        for r in sregs[i]:
                file.write('& %0.3f' % r.params['np.log(gdppc)'])
        file.write('\\\\\n')
        
        file.write('log population')
        for r in sregs[i]:
                file.write('& %0.3f' % r.params['np.log(popt)'])
        file.write('\\\\\n')
        
        file.write('log trade barrier')
        for r in sregs[i]:
                file.write('& %0.3f' % r.params['np.log(tau)'])
        file.write('\\\\\n')        
        
# footer
file.write('\\bottomrule\n')
file.write('\\end{tabular}\n')
file.write('\\end{center}\n')
file.write('\\normalsize\n')
file.write('\\end{table}\n')

file.close()



##############################################################################################3

print('\tEstimating additional relationships between destination characteristics and cross-sectional moments')

lhs=['p25_norm_exports',
     'p75_norm_exports',
     'p95_norm_exports',
     'p25_norm_exports_incumbent',
     'p75_norm_exports_incumbent',
     'p95_norm_exports_incumbent',
     'p25_norm_exports_entrant',
     'p75_norm_exports_entrant',
     'p95_norm_exports_entrant']

rhs='np.log(gdppc) + np.log(popt) + np.log(tau) + C(y)'
formulas=['np.log('+l+')~'+rhs for l in lhs]
dregs = [ols(formula=f,data=agg_by_d).fit(cov_type='cluster',cov_kwds={'groups':agg_by_d['d']}) for f in formulas]

rhs='np.log(gdppc) + np.log(popt) + np.log(tau) + C(y)'
formulas=['np.log('+l+')~'+rhs for l in lhs]
sregs = [ols(formula=f,data=agg_by_d_s[0]).fit() for f in formulas]


print('\tWriting LaTeX table with actual data vs. baseline model')

vars=['p25_norm_exports',
      'p75_norm_exports',
      'p95_norm_exports',
      'p25_norm_exports_incumbent',
      'p75_norm_exports_incumbent',
      'p95_norm_exports_incumbent',
      'p25_norm_exports_entrant',
      'p75_norm_exports_entrant',
      'p95_norm_exports_entrant']

vnames=['25th','75th','95th',
        '25th','75th','95th',
        '25th','75th','95th']

groups=['All firms','Incumbents','Entrants']

fmt=['%0.3f' for x in vars]

file = open(outpath + 'tableB1_sumstats_regs_cross_sec.tex','w')

# header
file.write('\\begin{table}[h!]\n')
file.write('\\footnotesize\n')
file.write('\\renewcommand{\\arraystretch}{1.2}\n')
file.write('\\begin{center}\n')
file.write("\\caption{Percentiles of normalized exports across markets}\n")
file.write('\\label{app:tab:sumstats_regs_cross_sec}\n')
file.write('\\begin{tabular}{l')
for i in range(len(vars)):
        file.write('c')
file.write('}')
file.write('\\toprule\n')

file.write('&\\multicolumn{3}{c}{All firms}&\\multicolumn{3}{c}{Incumbents}&\\multicolumn{3}{c}{Entrants}\\\\\n')

# column names
file.write('Statistic/coefficient')
for vname in vnames:
        file.write('& \\makecell{'+vname+'}')
file.write('\\\\\n')
file.write('\\midrule\n')

tmp = agg_by_d2
file.write('\\multicolumn{%d}{l}{\\textit{(a) Summary statistics (data)}}\\\\[4pt]\n'%(len(vars)+1))

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

file.write('\\midrule\\multicolumn{%d}{l}{\\textit{(b) Associations with market characteristics (data)}}\\\\[4pt]\n'%(len(vars)+1))

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

tmp = agg_by_d2_s[0]
file.write('\\multicolumn{%d}{l}{\\textit{(c) Summary statistics (baseline model)}}\\\\[4pt]\n'%(len(vars)+1))

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

file.write('\\midrule\\multicolumn{%d}{l}{\\textit{(d) Associations with market characteristics (baseline model)}}\\\\[4pt]\n'%(len(vars)+1))

file.write('log GDPpc')
for r in sregs:
        file.write('& %0.3f' % r.params['np.log(gdppc)'])
file.write('\\\\\n')

file.write('log population')
for r in sregs:
        file.write('& %0.3f' % r.params['np.log(popt)'])
file.write('\\\\\n')

file.write('log trade barrier')
for r in sregs:
        file.write('& %0.3f' % r.params['np.log(tau)'])
file.write('\\\\\n')


# footer
file.write('\\bottomrule\n')
file.write('\\end{tabular}\n')
file.write('\\end{center}\n')
file.write('\\normalsize\n')
file.write('\\end{table}\n')

file.close()



##############################################################################################3

print('\tEstimating additional relationships between destination characteristics and export costs (model only)')

lhs=['np.log(avg_cost)',
     'erel_cost',
     'avg_cost2',
     'erel_cost2']

rhs='np.log(gdppc) + np.log(popt) + np.log(tau) + C(y)'
formulas=['np.log('+l+')~'+rhs for l in lhs]
sregs = [ols(formula=f,data=agg_by_d_s[0]).fit() for f in formulas]

print('\tWriting LaTeX table with actual data vs. baseline model')

vars=['avg_cost',
      'erel_cost',
      'avg_cost2',
      'erel_cost2']

vnames = ['Avg.\\\\cost','Entrant\\\\rel. cost','Avg.\\\\cost/profits','Entrant rel.\\\\cost/profits']

fmt=['%0.3f' for x in vars]

file = open(outpath + 'table4_sumstats_regs_costs.tex','w')

# header
file.write('\\begin{table}[h!]\n')
file.write('\\footnotesize\n')
file.write('\\renewcommand{\\arraystretch}{1.2}\n')
file.write('\\begin{center}\n')
file.write("\\caption{Calibrated export costs across markets}\n")
file.write('\\label{tab:sumstats_regs_costs}\n')
file.write('\\begin{tabular}{l')
for i in range(len(vars)):
        file.write('c')
file.write('}')
file.write('\\toprule\n')

# column names
file.write('Statistic/coefficient')
for vname in vnames:
        file.write('& \\makecell{'+vname+'}')
file.write('\\\\\n')
file.write('\\midrule\n')

tmp = agg_by_d2_s[0]
tmp['avg_cost'] = tmp.avg_cost/tmp.avg_cost.mean()

file.write('\\multicolumn{%d}{l}{\\textit{(a) Summary statistics}}\\\\[4pt]\n'%(len(vars)+1))

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

file.write('\\midrule\\multicolumn{%d}{l}{\\textit{(b) Associations with market characteristics}}\\\\[4pt]\n'%(len(vars)+1))

file.write('log GDPpc')
for r in sregs:
        file.write('& %0.3f' % r.params['np.log(gdppc)'])
file.write('\\\\\n')

file.write('log population')
for r in sregs:
        file.write('& %0.3f' % r.params['np.log(popt)'])
file.write('\\\\\n')

file.write('log trade barrier')
for r in sregs:
        file.write('& %0.3f' % r.params['np.log(tau)'])
file.write('\\\\\n')


# footer
file.write('\\bottomrule\n')
file.write('\\end{tabular}\n')
file.write('\\end{center}\n')
file.write('\\normalsize\n')
file.write('\\end{table}\n')

file.close()
