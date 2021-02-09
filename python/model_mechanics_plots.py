import pandas as pd
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

mpl.rc('text', usetex=True)
mpl.rc('savefig',bbox='tight')
mpl.rc('savefig',format='pdf')
mpl.rc('font',**{'family':'serif','serif':['Palatino'],'size':8})
mpl.rc('font',size=8)
mpl.rc('lines',linewidth=1)

alpha=0.8
colors=['#377eb8','#e41a1c','#4daf4a','#984ea3']

NM=100
NX=50
NZ=101
ND=63

#############################################################################

inpath = '../c/output/'
m_grid = np.genfromtxt(inpath+"m_grid.txt")

export_cost = np.genfromtxt(inpath+"export_cost.txt")\
                .reshape((ND,NM,NM))

export_cost_argmin_n = np.genfromtxt(inpath+"export_cost_argmin_n.txt")\
                         .reshape((ND,NM,NM))

export_cost_argmin_o = np.genfromtxt(inpath+"export_cost_argmin_o.txt")\
                         .reshape((ND,NM,NM))

export_cost_dmp = np.genfromtxt(inpath+"export_cost_deriv_mp.txt")\
                    .reshape((ND,NM,NM))

gm = np.genfromtxt(inpath+"gm.txt").reshape((ND,NX,NZ,NM))

#############################################################################

im1=23
im2=43
m1=m_grid[im1]
m2=m_grid[im2]

fig,axes=plt.subplots(2,2,figsize=(6.5,6.5),sharex=True,sharey=False)


# panel (a): new customers
axes[0][0].plot(m_grid,export_cost_argmin_n[60][0],
             label='Entrant',color=colors[0],alpha=0.75)

axes[0][0].plot(m_grid,export_cost_argmin_n[60][im1],
             label='m=%0.1f'%m1,color=colors[1],alpha=0.75)

axes[0][0].plot(m_grid,export_cost_argmin_n[60][im2],
             label='m=%0.1f'%m2,color=colors[2],alpha=0.75)

axes[0][0].axvline(m_grid[im1],color=colors[1],linestyle=':',alpha=0.75)
axes[0][0].axvline(m_grid[im2],color=colors[2],linestyle=':',alpha=0.75)
axes[0][0].set_title(r'(a) New customers $n\in[0,1-m]$')
axes[0][0].legend(loc='best',prop={'size':6})

# panel (b) old customers
axes[0][1].plot(m_grid,export_cost_argmin_o[60][0],
             label='Entrant',color=colors[0],alpha=0.75)

axes[0][1].plot(m_grid,export_cost_argmin_o[60][im1],
             label='m=%0.1f'%m1,color=colors[1],alpha=0.75)

axes[0][1].plot(m_grid,export_cost_argmin_o[60][im2],
             label='m=%0.1f'%m2,color=colors[2],alpha=0.75)

axes[0][1].axvline(m_grid[im1],color=colors[1],linestyle=':',alpha=0.75)
axes[0][1].axvline(m_grid[im2],color=colors[2],linestyle=':',alpha=0.75)

axes[0][1].set_title(r'(b) Old customers $o\in[0,m]$')
#axes[2].legend(loc='best',prop={'size':8})


# panel (a): export cost
axes[1][0].plot(m_grid,export_cost[60][0],
             label='Entrant',color=colors[0],alpha=0.75)

axes[1][0].plot(m_grid,export_cost[60][im1],
             label='m=%0.1f'%m1,color=colors[1],alpha=0.75)

axes[1][0].plot(m_grid,export_cost[60][im2],
             label='m=%0.1f'%m2,color=colors[2],alpha=0.75)

axes[1][0].axvline(m_grid[im1],color=colors[1],linestyle=':',alpha=0.75)
axes[1][0].axvline(m_grid[im2],color=colors[2],linestyle=':',alpha=0.75)
axes[1][0].set_title(r"(c) Exporting cost $f_j(m,m')$")
#axes[1][0].legend(loc='best')
#axes[0].set_yscale('log')


# panel (c) old customers
axes[1][1].plot(m_grid,export_cost_dmp[60][0],
             label='Entrant',color=colors[0],alpha=0.75)

axes[1][1].plot(m_grid,export_cost_dmp[60][im1],
             label='m=%0.1f'%m1,color=colors[1],alpha=0.75)

axes[1][1].plot(m_grid,export_cost_dmp[60][im2],
             label='m=%0.1f'%m2,color=colors[2],alpha=0.75)

axes[1][1].axvline(m_grid[im1],color=colors[1],linestyle=':',alpha=0.75)
axes[1][1].axvline(m_grid[im2],color=colors[2],linestyle=':',alpha=0.75)
axes[1][1].set_title(r"(d) Marginal exporting cost $f_{j,m'}(m,m')$")

axes[0][0].set_xlim(0,0.5)
axes[1][0].set_xlabel("$m'$")
axes[1][1].set_xlabel("$m'$")

axes[0][0].set_ylim(0,0.5)
axes[0][1].set_ylim(0,0.5)
axes[1][0].set_ylim(0,50)
axes[1][1].set_ylim(0,200)


fig.subplots_adjust(hspace=0.15,wspace=0.15)
plt.savefig("output/export_cost_example.pdf",bbox='tight')
plt.close('all')





fig,axes=plt.subplots(1,3,figsize=(7,3),sharex=True,sharey=False)


# panel (a): new customers
axes[0].plot(m_grid,export_cost_argmin_n[60][0],
             label='Entrant',color=colors[0],alpha=0.75)

axes[0].plot(m_grid,export_cost_argmin_n[60][im1],
             label='m=%0.1f'%m1,color=colors[1],alpha=0.75)

axes[0].plot(m_grid,export_cost_argmin_n[60][im2],
             label='m=%0.1f'%m2,color=colors[2],alpha=0.75)

axes[0].axvline(m_grid[im1],color=colors[1],linestyle=':',alpha=0.75)
axes[0].axvline(m_grid[im2],color=colors[2],linestyle=':',alpha=0.75)
axes[0].set_title(r'(a) New customers $n\in[0,1-m]$')
axes[0].legend(loc='best',prop={'size':6})

# panel (b) old customers
axes[1].plot(m_grid,export_cost_argmin_o[60][0],
             label='Entrant',color=colors[0],alpha=0.75)

axes[1].plot(m_grid,export_cost_argmin_o[60][im1],
             label='m=%0.1f'%m1,color=colors[1],alpha=0.75)

axes[1].plot(m_grid,export_cost_argmin_o[60][im2],
             label='m=%0.1f'%m2,color=colors[2],alpha=0.75)

axes[1].axvline(m_grid[im1],color=colors[1],linestyle=':',alpha=0.75)
axes[1].axvline(m_grid[im2],color=colors[2],linestyle=':',alpha=0.75)

axes[1].set_title(r'(b) Old customers $o\in[0,m]$')
#axes[2].legend(loc='best',prop={'size':8})


# panel (c) old customers
axes[2].plot(m_grid,export_cost_dmp[60][0],
             label='Entrant',color=colors[0],alpha=0.75)

axes[2].plot(m_grid,export_cost_dmp[60][im1],
             label='m=%0.1f'%m1,color=colors[1],alpha=0.75)

axes[2].plot(m_grid,export_cost_dmp[60][im2],
             label='m=%0.1f'%m2,color=colors[2],alpha=0.75)

axes[2].axvline(m_grid[im1],color=colors[1],linestyle=':',alpha=0.75)
axes[2].axvline(m_grid[im2],color=colors[2],linestyle=':',alpha=0.75)
axes[2].set_title(r"(c) Marginal exporting cost $f_{j,m'}(m,m')$")

axes[0].set_xlim(0,0.5)
axes[0].set_xlabel("$m'$")
axes[1].set_xlabel("$m'$")
axes[2].set_xlabel("$m'$")

axes[0].set_ylim(0,0.5)
axes[1].set_ylim(0,0.5)
axes[2].set_ylim(0,200)


axes[2].set_title(r"(c) Marginal cost $f_{j,m'}(m,m')$")

fig.subplots_adjust(hspace=0.15,wspace=0.25)
plt.savefig("output/export_cost_example1.pdf",bbox='tight')
plt.close('all')






im1=23
im2=43
m1=m_grid[im1]
m2=m_grid[im2]

L=0.3709880261672286
Y=0.1775889019985724

fig,axes=plt.subplots(1,1,figsize=(4,4),sharex=True,sharey=False)


# panel (c) old customers
axes.plot(m_grid,export_cost_dmp[60][0],
             label='Entrant (easy)',color=colors[0],alpha=0.75)

axes.plot(m_grid,export_cost_dmp[40][1]/L/Y,
             label='Entrant (hard)',color=colors[1],alpha=0.75,linestyle='--')

axes.plot(m_grid,export_cost_dmp[60][im1],
             label='Incumbent (easy)'%m1,color=colors[2],alpha=0.75)

axes.plot(m_grid,export_cost_dmp[40][im1]/L/Y,
             label='Incumbent (hard)'%m1,color=colors[3],alpha=0.75,linestyle='--')

axes.set_title(r"Marginal cost/market size $f_{j,m'}(m,m')$")

axes.legend(loc='best',prop={'size':6})
axes.set_xlim(0,0.5)
axes.set_xlabel("$m'$")
axes.set_xlabel("$m'$")
axes.set_ylim(0,500)

fig.subplots_adjust(hspace=0.15,wspace=0.15)
plt.savefig("output/export_cost_example2.pdf",bbox='tight')
plt.close('all')


#############################################################################

ix=np.nonzero(gm[60,:,NZ-1,0])[0][0]+24
iz=np.nonzero(gm[60,ix,:,NM-1])[0][0]
im1=np.argmax(m_grid>gm[60,ix,iz+15,0])
im2=np.argmax(m_grid>gm[60,ix,iz+15,im1])
im3=np.argmax(m_grid>gm[60,ix,iz+15,im2])
im4=np.argmax(m_grid>gm[60,ix,iz+15,im3])
im5=np.argmax(m_grid>gm[60,ix,iz+15,im4])
m1=gm[60,ix,iz+15,0]
m2=gm[60,ix,iz+15,im1]
m3=gm[60,ix,iz+15,im2]
m4=gm[60,ix,iz+15,im3]
m5=gm[60,ix,iz+15,im4]

gm[60][ix+4][iz-10][0:44]=0

#------------------------------
# first version: only mc curve and policy functio
fig,axes=plt.subplots(1,2,figsize=(7,3.0),sharex=False,sharey=False)

# panel (a): marginal cost
axes[0].plot(m_grid,export_cost_dmp[60][0],
             label='Entry',color=colors[0],alpha=0.75)

axes[0].set_title(r"(a) Marginal cost $f_{j,m'}(m,m')$")
#axes[0].legend(loc='upper left')
axes[0].set_xlim(0,0.6)
axes[0].set_ylim(-15,160)
axes[0].set_xlabel(r"$m'$")
axes[0].set_xticks([0])
axes[0].set_xticklabels([r'0'])
axes[0].set_yticks([])
axes[0].annotate(xy=(105,150),xytext=(0,0),xycoords='axes points',textcoords='offset points',s=r"$f_{j,m'}(0,\cdot)$",size=6)

# panel (b): policy functions
axes[1].plot(m_grid,gm[60][ix][iz+15],
             label=r'High $z$',color=colors[0],alpha=0.75)

#axes[1].legend(loc='best')
axes[1].set_xlim(0,.9)
axes[1].set_ylim(-0.05,0.75)
axes[1].set_xlabel(r"$m$")
axes[1].annotate(xy=(115,127),xytext=(0,0),xycoords='axes points',textcoords='offset points',s=r"$m'_j(z_{hi},\cdot)$",size=6)
axes[1].set_title(r"(b) Policy function $m'_j(z,m)$")
axes[1].set_xticks([0])
axes[1].set_xticklabels([r'0'])
axes[1].set_yticks([0])
axes[1].set_yticklabels([r'0'])

plt.savefig("output/policy_function_example1.pdf",bbox='tight')
plt.close('all')

#------------------------------
# second version: mc curve and policy function + choice
fig,axes=plt.subplots(1,2,figsize=(7,3.0),sharex=False,sharey=False)

# panel (a): marginal cost
axes[0].plot(m_grid,export_cost_dmp[60][0],
             label='Entry',color=colors[0],alpha=0.75)

axes[0].set_title(r"(a) Marginal cost $f_{j,m'}(m,m')$")
#axes[0].legend(loc='upper left')
axes[0].set_xlim(0,0.6)
axes[0].set_ylim(-15,160)
axes[0].set_xlabel(r"$m'$")
axes[0].set_xticks([m_grid[im1]])
axes[0].set_xticklabels([r'$m_1$'])
axes[0].set_yticks([])
axes[0].axvline(m_grid[im1],color='black',linewidth=1, linestyle=':')
axes[0].annotate(xy=(105,150),xytext=(0,0),xycoords='axes points',textcoords='offset points',s=r"$f_{j,m'}(0,\cdot)$",size=6)
axes[0].annotate(xy=(2,87),xytext=(0,0),xycoords='axes points',textcoords='offset points',s=r"$\tilde{\pi}_j (xz_{hi})^{\theta-1} - \beta\mathbb{E}[f_{j,m}|z_{hi}]$",size=6)

# panel (b): policy functions
axes[1].plot(m_grid,gm[60][ix][iz+15],
             label=r'High $z$',color=colors[0],alpha=0.75)

#axes[1].legend(loc='best')
axes[1].set_xlim(0,0.9)
axes[1].set_xlabel(r"$m$")
axes[1].set_title(r"(b) Policy function $m'_j(z,m)$")
axes[1].annotate(xy=(115,127),xytext=(0,0),xycoords='axes points',textcoords='offset points',s=r"$m'_j(z_{hi},\cdot)$",size=6)

axes[0].scatter(m_grid[im1],export_cost_dmp[60][0][im1],color='black',alpha=1,s=10,marker='o')
axes[0].axhline(export_cost_dmp[60][0][im1],color='black',linewidth=1,linestyle=':')
axes[1].scatter(0,gm[60][ix][iz+15][0],color='black',alpha=1,s=10,marker='o')

axes[1].set_xticks([0])
axes[1].set_xticklabels([r'0'])
axes[1].set_yticks([gm[60][ix][iz+15][0]])
axes[1].set_yticklabels([r'$m_1$'])

axes[1].set_ylim(-0.05,0.75)
plt.savefig("output/policy_function_example2.pdf",bbox='tight')
plt.close('all')

#------------------------------
# third version: period 1
fig,axes=plt.subplots(1,2,figsize=(7,3.0),sharex=False,sharey=False)

# panel (a): marginal cost
axes[0].plot(m_grid,export_cost_dmp[60][0],
             label='Entry',color=colors[0],alpha=0.25)

axes[0].plot(m_grid,export_cost_dmp[60][im1],
             label='Period 1'%m1,color=colors[0],alpha=0.75)

axes[0].set_title(r"(a) Marginal cost $f_{j,m'}(m,m')$")
#axes[0].legend(loc='upper left')
axes[0].set_xlim(0,0.6)
axes[0].set_ylim(-15,160)
axes[0].set_xlabel(r"$m'$")
axes[0].annotate(xy=(128,113),xytext=(0,0),xycoords='axes points',textcoords='offset points',s=r"$f_{j,m'}(m_1,\cdot)$",size=6)
axes[0].annotate(xy=(2,87),xytext=(0,0),xycoords='axes points',textcoords='offset points',s=r"$\tilde{\pi}_j (xz_{hi})^{\theta-1} - \beta\mathbb{E}[f_{j,m}|z_{hi}]$",size=6)
axes[0].set_xticks([m_grid[im1],m_grid[im2]])
axes[0].set_xticklabels([r'$m_1$',r'$m_2$'])
axes[0].set_yticks([])

# panel (b): policy functions
axes[1].plot(m_grid,gm[60][ix][iz+15],
             label=r'High $z$',color=colors[0],alpha=0.75)
#axes[1].legend(loc='best')
axes[1].set_xlim(0,0.9)
axes[1].set_xlabel(r"$m$")
axes[1].set_title(r"(b) Policy function $m'_j(z,m)$")
axes[1].annotate(xy=(115,127),xytext=(0,0),xycoords='axes points',textcoords='offset points',s=r"$m'_j(z_{hi},\cdot)$",size=6)

axes[0].scatter(m_grid[im1],export_cost_dmp[60][0][im1],color='black',alpha=0.25,s=10,marker='o')
axes[0].scatter(m_grid[im2],export_cost_dmp[60][im1][im2],color='black',alpha=1,s=10,marker='o')
axes[0].axhline(export_cost_dmp[60][im1][im2],color='black',linewidth=1,linestyle=':')
#axes[0].axvline(m_grid[im1],color='black',linewidth=1, linestyle=':',alpha=0.25)
axes[0].axvline(m_grid[im2],color='black',linestyle=':',linewidth=1)

axes[1].axvline(m_grid[im1],color='black',linestyle=':',linewidth=1)
axes[1].scatter(0,gm[60][ix][iz+15][0],color='black',alpha=0.25,s=10,marker='o')
axes[1].scatter(m_grid[im1],gm[60][ix][iz+15][im1],color='black',alpha=1,s=10,marker='o')

axes[1].set_xticks([0,m_grid[im1]])
axes[1].set_xticklabels([r'0',r'$m_1$'])
axes[1].set_yticks([gm[60][ix][iz+15][0],m_grid[im2]])
axes[1].set_yticklabels([r'$m_1$',r'$m_2$'])
axes[1].set_ylim(-0.05,0.75)

plt.savefig("output/policy_function_example3.pdf",bbox='tight')
plt.close('all')

#------------------------------
# fourth version: period 2
fig,axes=plt.subplots(1,2,figsize=(7,3.0),sharex=False,sharey=False)

# panel (a): marginal cost
axes[0].plot(m_grid,export_cost_dmp[60][0],
             label='Entry',color=colors[0],alpha=0.25)

axes[0].plot(m_grid,export_cost_dmp[60][im1],
             label='Period 1'%m1,color=colors[0],alpha=0.25)

axes[0].plot(m_grid,export_cost_dmp[60][im2],
             label='Period 2',color=colors[0],alpha=0.75)

axes[0].set_title(r"(a) Marginal cost $f_{j,m'}(m,m')$")
#axes[0].legend(loc='upper left')
axes[0].set_xlim(0,0.6)
axes[0].set_ylim(-15,160)
axes[0].set_xlabel(r"$m'$")
axes[0].set_xticks([m_grid[im1],m_grid[im2],m_grid[im3]])
axes[0].set_xticklabels([r'$m_1$',r'$m_2$',r'$m_3$'])
axes[0].set_yticks([])

# panel (b): policy functions
axes[1].plot(m_grid,gm[60][ix][iz+15],
             label=r'High $z$',color=colors[0],alpha=0.75)

#axes[1].legend(loc='best')
axes[1].set_xlim(0,.9)
axes[1].set_ylim(-0.05,0.75)
axes[1].set_xlabel(r"$m$")
axes[1].set_yticks([gm[60][ix][iz+15][0],m_grid[im2],m_grid[im3]])
axes[1].set_yticklabels([r'$m_1$',r'$m_2$',r'$m_3$'])
axes[1].set_title(r"(b) Policy function $m'_j(z,m)$")

axes[0].scatter(m_grid[im1],export_cost_dmp[60][0][im1],color='black',alpha=0.25,s=10,marker='o')
axes[0].scatter(m_grid[im2],export_cost_dmp[60][im1][im2],color='black',alpha=0.25,s=10,marker='o')
axes[0].scatter(m_grid[im3],export_cost_dmp[60][im2][im3],color='black',alpha=1,s=10,marker='o')
axes[0].axhline(export_cost_dmp[60][im2][im3],color='black',linewidth=1,linestyle=':')

#axes[0].axvline(m_grid[im1],color='black',linewidth=1, linestyle=':',alpha=0.25)
#axes[0].axvline(m_grid[im2],color='black',linestyle=':',linewidth=1,alpha=0.25)
axes[0].axvline(m_grid[im3],color='black',linestyle=':',linewidth=1)

axes[1].scatter(0,gm[60][ix][iz+15][0],color='black',alpha=0.25,s=10,marker='o')
axes[1].scatter(m_grid[im1],m2,color='black',alpha=0.25,s=10,marker='o')
axes[1].scatter(m_grid[im2],m3,color='black',alpha=1,s=10,marker='o')

axes[1].axvline(m_grid[im1],color='black',linestyle=':',linewidth=1,alpha=0.25)
axes[1].axvline(m_grid[im2],color='black',linestyle=':',linewidth=1)

axes[1].annotate(xy=(115,127),xytext=(0,0),xycoords='axes points',textcoords='offset points',s=r"$m'_j(z_{hi},\cdot)$",size=6)
axes[1].set_xticks([0,m_grid[im1],m_grid[im2]])
axes[1].set_xticklabels([r'0',r'$m_1$',r'$m_2$'])
axes[1].set_yticks([gm[60][ix][iz+15][0],m_grid[im2],m_grid[im3]])
axes[1].set_yticklabels([r'$m_1$',r'$m_2$',r'$m_3$'])

axes[0].annotate(xy=(139,110),xytext=(0,0),xycoords='axes points',textcoords='offset points',s=r"$f_{j,m'}(m_2,\cdot)$",size=6)
axes[0].annotate(xy=(2,87),xytext=(0,0),xycoords='axes points',textcoords='offset points',s=r"$\tilde{\pi}_j (xz_{hi})^{\theta-1} - \beta\mathbb{E}[f_{j,m}|z_{hi}]$",size=6)

plt.savefig("output/policy_function_example4.pdf",bbox='tight')
plt.close('all')


#------------------------------
# fifth version: period 2 with shocks
fig,axes=plt.subplots(1,2,figsize=(7,3.0),sharex=False,sharey=False)

# panel (a): marginal cost
axes[0].plot(m_grid,export_cost_dmp[60][0],
             label='Entry',color=colors[0],alpha=0.25)

axes[0].plot(m_grid,export_cost_dmp[60][im1],
             label='Period 1'%m1,color=colors[0],alpha=0.25)

axes[0].plot(m_grid,export_cost_dmp[60][im2],
             label='Period 2',color=colors[0],alpha=0.75)

axes[0].set_title(r"(a) Marginal cost $f_{j,m'}(m,m')$")
#axes[0].legend(loc='upper left')
axes[0].set_xlim(0,0.6)
axes[0].set_ylim(-15,160)
axes[0].set_xlabel(r"$m'$")
axes[0].set_xticks([m_grid[im1],m_grid[im2],m_grid[im3]])
axes[0].set_xticklabels([r'$m_1$',r'$m_2$',r'$m_3$'])
axes[0].set_yticks([])

# panel (b): policy functions
axes[1].plot(m_grid,gm[60][ix][iz+15],
             label=r'High $z$',color=colors[0],alpha=0.75)

axes[1].plot(m_grid,gm[60][ix+4][iz-10],
             label=r'Medium $z$',color=colors[1],alpha=0.75)

#axes[1].legend(loc='best')
axes[1].set_xlim(0,.9)
axes[1].set_ylim(-0.05,.75)
axes[1].set_xlabel(r"$m$")
axes[1].set_title(r"(b) Policy function $m'_j(z,m)$")
axes[1].set_xticks([0,m_grid[im1],m_grid[im2]])
axes[1].set_xticklabels([r'0',r'$m_1$',r'$m_2$'])
axes[1].set_yticks([gm[60][ix][iz+15][0],m_grid[im2],m_grid[im3]])
axes[1].set_yticklabels([r'$m_1$',r'$m_2$',r'$m_3$'])

axes[0].annotate(xy=(139,110),xytext=(0,0),xycoords='axes points',textcoords='offset points',s=r"$f_{j,m'}(m_2,\cdot)$",size=6)
axes[0].annotate(xy=(20,16),xytext=(0,0),xycoords='axes points',textcoords='offset points',s=r"$\tilde{\pi}_j (xz_{lo})^{\theta-1} - \beta\mathbb{E}[f_{j,m}|z_{lo}]$",size=6)

axes[1].annotate(xy=(115,127),xytext=(0,0),xycoords='axes points',textcoords='offset points',s=r"$m'_j(z_{hi},\cdot)$",size=6)
axes[1].annotate(xy=(120,41),xytext=(0,0),xycoords='axes points',textcoords='offset points',s=r"$m'_j(z_{lo},\cdot)$",size=6)

axes[0].scatter(m_grid[im1],export_cost_dmp[60][0][im1],color='black',alpha=0.25,s=10,marker='o')
axes[0].scatter(m_grid[im2],export_cost_dmp[60][im1][im2],color='black',alpha=0.25,s=10,marker='o')
axes[0].scatter(m_grid[im3],export_cost_dmp[60][im2][im3],color='black',alpha=0.25,s=10,marker='o')
#axes[0].scatter(m_grid[im4],export_cost_dmp[60][im2][im4],color='black',alpha=1,s=10,marker='o')
axes[0].axhline(export_cost_dmp[60][im2][im3],color='black',linewidth=1,linestyle=':',alpha=0.24)
axes[0].axvline(m_grid[im3],color='black',linestyle=':',linewidth=1,alpha=0.25)
axes[0].axhline(0.0,color='black',linewidth=1,linestyle=':')

axes[1].scatter(0,gm[60][ix][iz+15][0],color='black',alpha=0.25,s=10,marker='o')
axes[1].scatter(m_grid[im1],m2,color='black',alpha=0.25,s=10,marker='o')
axes[1].scatter(m_grid[im2],m3,color='black',alpha=0.25,s=10,marker='o')
#axes[1].scatter(m_grid[im3],m4,color='black',alpha=1,s=10,marker='o')

axes[1].axvline(m_grid[im1],color='black',linestyle=':',linewidth=1,alpha=0.25)
x=axes[1].axvline(m_grid[im2],color='black',linestyle=':',linewidth=1)

plt.savefig("output/policy_function_example5.pdf",bbox='tight')

x.set_alpha(0.25)
axes[1].axvline(m_grid[im3],color='black',linestyle=':',linewidth=1)
axes[1].set_xticks([0,m_grid[im1],m_grid[im2],m_grid[im3]])
axes[1].set_xticklabels([r'0',r'$m_1$',r'$m_2$',r'$m_3$'])

plt.savefig("output/policy_function_example6.pdf",bbox='tight')

plt.close('all')








#------------------------------
# sixth version: period 2 with shocks
fig,axes=plt.subplots(1,2,figsize=(7,3.0),sharex=False,sharey=False)

# panel (a): marginal cost
axes[0].plot(m_grid,export_cost_dmp[60][0],
             label='Entry',color=colors[0],alpha=0.75,linestyle='-')

axes[0].plot(m_grid,export_cost_dmp[60][im1],
             label='Period 1'%m1,color=colors[1],alpha=0.75,linestyle='--')

axes[0].plot(m_grid,export_cost_dmp[60][im3],
             label='Period 2',color=colors[2],alpha=0.75,linestyle='-.')


axes[0].set_title(r"(a) Marginal cost $f_{j,m'}(m,m')$")
#axes[0].legend(loc='upper left')
axes[0].set_xlim(0,0.6)
axes[0].set_ylim(-15,160)
axes[0].set_xlabel(r"$m'$")
axes[0].set_xticks([m_grid[im1],m_grid[im2],m_grid[im4]])
axes[0].set_xticklabels([r'$m_1$',r'$m_2$',r'$m_3$'])
axes[0].set_yticks([])

# panel (b): policy functions
axes[1].plot(m_grid,gm[60][ix][iz+15],
             label=r'High $z$',color=colors[0],alpha=0.75)

axes[1].plot(m_grid,gm[60][ix+4][iz-10],
             label=r'Medium $z$',color=colors[1],alpha=0.75,linestyle='--')

#axes[1].legend(loc='best')
axes[1].set_xlim(0,.9)
axes[1].set_ylim(-0.05,.75)
axes[1].set_xlabel(r"$m$")
axes[1].set_title(r"(b) Policy function $m'_j(x,z,m)$")
axes[1].set_xticks([0,m_grid[im1],m_grid[im2],m_grid[im4]])
axes[1].set_xticklabels([r'0',r'$m_1$',r'$m_2$',r'$m_3$'])
axes[1].set_yticks([0,gm[60][ix][iz+15][0],m_grid[im2],m_grid[im3]])
axes[1].set_yticklabels([r'0',r'$m_1$',r'$m_2$',r'$m_3$'])

axes[0].annotate(xy=(105,150),xytext=(0,0),xycoords='axes points',textcoords='offset points',s=r"$f_{j,m'}(0,\cdot)$",size=6)
axes[0].annotate(xy=(98,117),xytext=(0,0),xycoords='axes points',textcoords='offset points',s=r"$f_{j,m'}(m_1,\cdot)$",size=6)
axes[0].annotate(xy=(145,110),xytext=(0,0),xycoords='axes points',textcoords='offset points',s=r"$f_{j,m'}(m_2,\cdot)$",size=6)
axes[0].annotate(xy=(10,88.5),xytext=(0,0),xycoords='axes points',textcoords='offset points',s=r"$\tilde{\pi}_j (xz_{hi})^{\theta-1} - \beta\mathbb{E}[f_{j,m}|z_{hi}]$",size=6)
axes[0].annotate(xy=(20,17),xytext=(0,0),xycoords='axes points',textcoords='offset points',s=r"$\tilde{\pi}_j (xz_{lo})^{\theta-1} - \beta \mathbb{E}[f_{j,m}|z_{lo}]$",size=6)

axes[1].annotate(xy=(115,131),xytext=(0,0),xycoords='axes points',textcoords='offset points',s=r"$m'_j(x,z_{hi},\cdot)$",size=6)
axes[1].annotate(xy=(120,44),xytext=(0,0),xycoords='axes points',textcoords='offset points',s=r"$m'_j(x,z_{lo},\cdot)$",size=6)

axes[0].scatter(m_grid[im1],export_cost_dmp[60][0][im1],color='black',alpha=1,s=10,marker='o')
axes[0].scatter(m_grid[im2],export_cost_dmp[60][im1][im2],color='black',alpha=1,s=10,marker='o')
axes[0].scatter(m_grid[im4],export_cost_dmp[60][im3][im4],color='black',alpha=1,s=10,marker='o')
#axes[0].scatter(m_grid[im4],export_cost_dmp[60][im2][im4],color='black',alpha=1,s=10,marker='o')
axes[0].axhline(export_cost_dmp[60][im1][im2]*1.02,color='black',linewidth=1,linestyle=':',alpha=1)
#axes[0].axvline(m_grid[im3],color='black',linestyle=':',linewidth=1,alpha=0.25)
axes[0].axhline(0.0,color='black',linewidth=1,linestyle=':')

axes[1].scatter(0,gm[60][ix][iz+15][0],color='black',alpha=1,s=10,marker='o')
axes[1].scatter(m_grid[im1],m2,color='black',alpha=1,s=10,marker='o')
#axes[1].scatter(m_grid[im4],m5,color='black',alpha=1,s=10,marker='o')
axes[1].scatter(m_grid[im2],m3,color='black',alpha=1,s=10,marker='o')

axes[1].axvline(m_grid[im1],color='black',linestyle=':',linewidth=1,alpha=0.25)
axes[1].axvline(m_grid[im2],color='black',linestyle=':',linewidth=1,alpha=0.25)
axes[1].axvline(m_grid[im4],color='black',linestyle=':',linewidth=1)

plt.savefig("output/policy_function_example7.pdf",bbox='tight')
plt.close('all')







#------------------------------
# seventh: period 2 with shocks
fig,axes=plt.subplots(1,1,figsize=(4,4.0),sharex=False,sharey=False)

# panel (b): policy functions
axes.plot(m_grid,gm[60][ix][iz+15],
             label=r'High $z$',color=colors[0],alpha=0.75)

axes.plot(m_grid,gm[60][ix+4][iz-10],
             label=r'Medium $z$',color=colors[1],alpha=0.75,linestyle='--')

axes.plot(m_grid,gm[7][ix][iz+15],
             label=r'High $z$',color=colors[0],alpha=0.75)

axes.plot(m_grid,gm[7][ix+4][iz-10],
             label=r'Medium $z$',color=colors[1],alpha=0.75,linestyle='--')

#axes[1].legend(loc='best')
axes.set_xlim(0,.9)
axes.set_ylim(-0.05,.75)
axes.set_xlabel(r"$m$")
axes.set_title(r"Policy function $m'_j(x,z,m)$")
axes.set_xticks([])
axes.set_yticks([])

#axes[1].annotate(xy=(115,131),xytext=(0,0),xycoords='axes points',textcoords='offset points',s=r"$m'_j(x,z_{hi},\cdot)$",size=6)
#axes[1].annotate(xy=(120,44),xytext=(0,0),xycoords='axes points',textcoords='offset points',s=r"$m'_j(x,z_{lo},\cdot)$",size=6)

plt.savefig("output/policy_function_example8.pdf",bbox='tight')
plt.close('all')


