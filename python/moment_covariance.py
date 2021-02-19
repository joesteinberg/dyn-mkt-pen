import numpy as np

N = 14

plabels = ['$\\sigma_x$',
           '$\\sigma_z$',
           '$\\rho_z$',
           '$\\psi_n$',
           '$\\alpha_n$',
           '$\\gamma_n$',
           '$\\psi_o$',
           '$\\alpha_o$',
           '$\\gamma_o$',
           '$Q$',
           '$\\delta_0$',
           '$\\delta_1$',
           '$\\beta_n$',
           '$\\beta_o$']
           
mlabels = ['T5 ($\\alpha$)',
           'T5 ($\\beta$)',
           'ND ($\\alpha$)',
           'ND ($\\beta$)',
           'XR ($\\alpha$)',
           'XR ($\\beta$)',
           'eS ($\\alpha$)',
           'eS ($\\beta$)',
           'eXR ($\\alpha$)',
           'eXR ($\\beta$)',
           '5yS (hard)',
           '5yS (easy)',
           '5yX',
           'NF']

results = np.genfromtxt('../c/output/results.csv',dtype=float,delimiter=',')

params = results[:,0:N]
moments = results[:,N:N*2]
errors = results[:,N+N:N*3]


#------------------------------------------------
# covariance matrix between moments

cm = np.cov(moments,rowvar=False)
icm = np.linalg.inv(cm)
np.savetxt("output/vcov_inv.txt",icm,delimiter=" ")

cm = np.corrcoef(moments,rowvar=False)
file = open('output/corr.tex','w')

file.write('\\begin{landscape}\n')
file.write('\\begin{table}[p]\n')
file.write('\\footnotesize\n')
file.write('\\renewcommand{\\arraystretch}{1.2}\n')
file.write('\\begin{center}\n')
file.write("\\caption{Correlation matrix between target moments}\n")
file.write('\\label{tab:covar}\n')
file.write('\\begin{tabular}{l')
for i in range(N):
        file.write('c')
file.write('}')
file.write('\\toprule\n')

for l in mlabels:
        file.write('&'+l)
file.write('\\\\\n')
file.write('\\midrule\n')

for i in range(N):
    file.write(mlabels[i])
    for j in range(N):
        if(j>=i):
            file.write('&%0.3f'%cm[i,j])
        else:
            file.write('&--')
    file.write('\\\\\n')

file.write('\\bottomrule\n')
file.write('\\end{tabular}\n')
file.write('\\end{center}\n')
file.write('\\normalsize\n')
file.write('\\end{table}\n')
file.write('\\end{landscape}\n')

file.close()

#------------------------------------------------
# covariance matrix between moments

cm = np.corrcoef(np.hstack((params,moments)),rowvar=False)
file = open('output/corr2.tex','w')

file.write('\\begin{landscape}\n')
file.write('\\begin{table}[p]\n')
file.write('\\footnotesize\n')
file.write('\\renewcommand{\\arraystretch}{1.2}\n')
file.write('\\begin{center}\n')
file.write("\\caption{Correlation matrix between parameters and target moments}\n")
file.write('\\label{tab:covar}\n')
file.write('\\begin{tabular}{l')
for i in range(N):
        file.write('c')
file.write('}')
file.write('\\toprule\n')

for l in mlabels:
        file.write('&'+l)
file.write('\\\\\n')
file.write('\\midrule\n')

for i in range(N):
    file.write(plabels[i])
    for j in range(N):
        file.write('&%0.3f'%cm[i,j+N])
    file.write('\\\\\n')

file.write('\\bottomrule\n')
file.write('\\end{tabular}\n')
file.write('\\end{center}\n')
file.write('\\normalsize\n')
file.write('\\end{table}\n')
file.write('\\end{landscape}\n')

file.close()

