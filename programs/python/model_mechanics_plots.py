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
mpl.rcParams['savefig.pad_inches'] = 0

alpha=0.8
colors=['#377eb8','#e41a1c','#4daf4a','#984ea3','#ff7f00']

NM=100
NX=50
NZ=101
ND=63

figpath = "output/model_mech_figs/"

#############################################################################

inpath = '../c/output/'
m_grid = np.genfromtxt(inpath+"m_grid.txt")

export_cost_dmp = np.genfromtxt(inpath+"export_cost_deriv_mp.txt")\
                    .reshape((ND,NM,NM))

gm = np.genfromtxt(inpath+"gm.txt").reshape((ND,NX,NZ,NM))

#############################################################################

ix=np.nonzero(gm[60,:,NZ-1,0])[0][0]+22
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

gm[60][ix+2][iz-10][0:30]=0

# sixth version: period 2 with shocks
fig,axes=plt.subplots(1,2,figsize=(6,3.0),sharex=False,sharey=False)

# panel (a): marginal cost
axes[0].plot(m_grid,export_cost_dmp[60][0],
             label='Entry',color=colors[2],alpha=0.75,linestyle='-')

axes[0].plot(m_grid,export_cost_dmp[60][im1],
             label='Period 1'%m1,color=colors[3],alpha=0.75,linestyle='--')

axes[0].plot(m_grid,export_cost_dmp[60][im2],
             label='Period 2',color=colors[4],alpha=0.75,linestyle='-.')


axes[0].set_title(r"(a) Marginal cost $f_{m'}(m,m')$")
axes[0].set_xlim(0.25,0.6)
axes[0].set_ylim(-150,900)
axes[0].set_xlabel(r"$m'$")
axes[0].set_xticks([m_grid[im1],m_grid[im2],m_grid[im3]])
axes[0].set_xticklabels([r'$m_1$',r'$m_2$',r'$m_3$'])
axes[0].set_yticks([])

axes[0].scatter(m_grid[im1],export_cost_dmp[60][0][im1],color='black',alpha=1,s=10,marker='o')
axes[0].scatter(m_grid[im2],export_cost_dmp[60][im1][im2],color='black',alpha=1,s=10,marker='o')
axes[0].scatter(m_grid[im3],export_cost_dmp[60][im2][im3],color='black',alpha=1,s=10,marker='o')
axes[0].axhline(export_cost_dmp[60][0][im1],color='black',linewidth=1,linestyle=':',alpha=1)
axes[0].axhline(export_cost_dmp[60][im1][im2],color='black',linewidth=1,linestyle=':',alpha=1)
axes[0].axhline(export_cost_dmp[60][im2][im3],color='black',linewidth=1,linestyle=':',alpha=1)
axes[0].axhline(0.0,color='black',linewidth=1,linestyle=':')

axes[0].annotate(xy=(85,158),xytext=(0,0),xycoords='axes points',textcoords='offset points',s=r"$f_{m'}(0,\cdot)$",size=6)
axes[0].annotate(xytext=(-60,0),xy=(115,135),xycoords='axes points',
                 textcoords='offset points',s=r"$f_{m'}(m_1,\cdot)$",size=6,
                 arrowprops=dict(arrowstyle='->'))
axes[0].annotate(xy=(123,122),xytext=(0,0),xycoords='axes points',textcoords='offset points',s=r"$f_{m'}(m_2,\cdot)$",size=6)

axes[0].annotate(xy=(80,42),xytext=(0,0),xycoords='axes points',textcoords='offset points',s=r"$\tilde{\pi}_{hi} - Q\mathbb{E}[f_{m}(m_1,m'')|z_{hi}]$",size=6)

axes[0].annotate(xy=(1,73),xytext=(0,0),xycoords='axes points',textcoords='offset points',s=r"$\tilde{\pi}_{hi} - Q\mathbb{E}[f_{m}(m_2,m'')|z_{hi}]$",size=6)

axes[0].annotate(xy=(1,106),xytext=(0,0),xycoords='axes points',textcoords='offset points',s=r"$\tilde{\pi}_{hi} - Q\mathbb{E}[f_{m}(m_3,m'')|z_{hi}]$",size=6)

axes[0].annotate(xy=(85,17),xytext=(0,0),xycoords='axes points',textcoords='offset points',s=r"$\tilde{\pi}_{lo} - Q \mathbb{E}[f_{m}(0,m'')|z_{lo}]$",size=6)

# panel (b): policy functions
axes[1].plot(m_grid,gm[60][ix][iz+15],
             label=r'High $z$',color=colors[0],alpha=0.75)

axes[1].plot(m_grid,gm[60][ix+2][iz-10],
             label=r'Medium $z$',color=colors[1],alpha=0.75,linestyle='--')

axes[1].set_xlim(0,.9)
axes[1].set_ylim(-0.05,.75)
axes[1].set_xlabel(r"$m$")
axes[1].set_title(r"(b) Policy function $m'(a,z,m)$")
axes[1].set_xticks([0,m_grid[im1],m_grid[im2],m_grid[im4]])
axes[1].set_xticklabels([r'0',r'$m_1$',r'$m_2$',r'$m_3$'])
axes[1].set_yticks([0,gm[60][ix][iz+15][0],m_grid[im2],m_grid[im3]])
axes[1].set_yticklabels([r'0',r'$m_1$',r'$m_2$',r'$m_3$'])

axes[1].scatter(0,gm[60][ix][iz+15][0],color='black',alpha=1,s=10,marker='o')
axes[1].scatter(m_grid[im1],m2,color='black',alpha=1,s=10,marker='o')
axes[1].scatter(m_grid[im2],m3,color='black',alpha=1,s=10,marker='o')

axes[1].axvline(m_grid[im1],color='black',linestyle=':',linewidth=1,alpha=0.25)
axes[1].axvline(m_grid[im2],color='black',linestyle=':',linewidth=1,alpha=0.25)
axes[1].axvline(m_grid[im3],color='black',linestyle=':',linewidth=1)

axes[1].annotate(xy=(115,137),xytext=(0,0),xycoords='axes points',textcoords='offset points',s=r"$m'(a,z_{hi},\cdot)$",size=6)
axes[1].annotate(xy=(120,46),xytext=(0,0),xycoords='axes points',textcoords='offset points',s=r"$m'(a,z_{lo},\cdot)$",size=6)


fig.subplots_adjust(hspace=0.2,wspace=0.2)
plt.savefig(figpath + "fig3_policy_function_example.pdf",bbox='tight')
plt.close('all')

