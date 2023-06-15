import numpy as np
import pandas as pd
import numba
import datetime
import sys

TMAX=25

# group by number of destinations
def which_nd_group(nd):
    if nd==1:
        return 1
    elif nd==2:
        return 2
    elif nd==3:
        return 3
    elif nd==4:
        return 4
    elif(nd>=5 and nd<10):
        return 6
    elif nd>=10:
        return 10
    else:
        return np.nan

##############################################################################################3
# read the microdata



inpath='/home/linux/Documents/Research/dyn_mkt_pen/v3/programs/c/output/'
outpath='/home/linux/Documents/Research/dyn_mkt_pen/v3/programs/python/output/pik/'

fname=''
pref=''
if len(sys.argv)==1 or (len(sys.argv)>1 and sys.argv[1]=='dmp'):
    fname='model_microdata_calibration.csv'
    pref='model'
elif len(sys.argv)>1 and sys.argv[1]=='sunk':
    fname='sunkcost_microdata.csv'
    pref='sunkcost'
elif len(sys.argv)>1 and sys.argv[1]=='acr':
    fname='acr_microdata.csv'
    pref='acr'
elif len(sys.argv)>1 and sys.argv[1]=='smp':
    fname='static_mkt_pen_microdata.csv'
    pref='smp'
elif len(sys.argv)>1 and sys.argv[1]=='abn1':
    fname='abn1_microdata.csv'
    pref='abn1'
elif len(sys.argv)>1 and sys.argv[1]=='abn2':
    fname='abn2_microdata.csv'
    pref='abn2'
elif len(sys.argv)>1 and sys.argv[1]=='abo1':
    fname='abo1_microdata.csv'
    pref='abo1'
elif len(sys.argv)>1 and sys.argv[1]=='abo2':
    fname='abo2_microdata.csv'
    pref='abo2'
elif len(sys.argv)>1 and sys.argv[1]=='a0':
    fname='a0_microdata.csv'
    pref='a0'

    
print('\tLoading simulated model microdata from '+fname)

dtypes={'f':str,
        'd':str,
        'y':int,
        'popt':float,
        'gddpc':float,
        'tau':float,
        'v':float,
        'm':float,
        'cost':float,
        'cost2':float,
        'entry':int,
        'exit':int,
        'incumbent':int,
        'nd':int,
        'nd_group':int,
        'tenure':int,
        'max_tenure':int,
        'ix':int,
        'iz':int,
        'multilateral_exit':int}

agged = pd.read_csv(inpath + fname, engine='c', dtype=dtypes, sep=',')
y0 = agged.y.min()
yT = agged.y.max()


#################################################################################

#redo entry, exit, tenure stuff
agged.drop(['entry','exit','tenure','max_tenure'],axis=1,inplace=True)

# entry, exit
agged.sort_values(['f','d','y'],ascending=[True,True,True],inplace=True)
agged.reset_index(drop=True)
agged['entry']=0
agged['incumbent']=0
agged['exit']=0
agged['growth']=np.nan

# define entry as observation where:
# 1. first occurence of firm f OR first occurence of dest d in firm f's block OR year != previous year;
# 2. and year is not 2000 (because everyone would be an entrant in the first year of data
entry_mask = np.logical_or(agged.f!=agged.f.shift(),agged.d!=agged.d.shift())
entry_mask = np.logical_or(entry_mask,agged.y != agged.y.shift()+1)
entry_mask = np.logical_and(entry_mask,agged.y!=y0)

incumbent_mask = np.logical_and(agged.d==agged.d.shift(),agged.f==agged.f.shift())
incumbent_mask = np.logical_and(incumbent_mask, agged.y==agged.y.shift()+1)
incumbent_mask = np.logical_and(incumbent_mask,agged.y!=y0)

# define exit as the opposite
exit_mask = np.logical_or(agged.d != agged.d.shift(-1),agged.f != agged.f.shift(-1))
exit_mask = np.logical_or(exit_mask,agged.y != agged.y.shift(-1)-1)
exit_mask = np.logical_and(exit_mask,agged.y != yT)

agged.loc[entry_mask,'entry']=1
agged.loc[incumbent_mask,'incumbent']=1
agged.loc[exit_mask,'exit']=1

agged['censored']=0
agged.loc[np.logical_and(incumbent_mask,agged.y==yT),'censored']=1

agged['tenure']=np.nan
agged['spellid']=np.nan
agged.loc[agged.entry==1,'tenure']=0
agged.loc[agged.entry==1,'spellid'] = agged.loc[agged.entry==1,'f'] + agged.loc[agged.entry==1,'d'] + agged.loc[agged.entry==1,'y'].astype('string')

for k in range(1,TMAX+1):
    tmp_mask = np.logical_and(agged.tenure.shift()==k-1,incumbent_mask)
    agged.loc[tmp_mask,'tenure']=k
    agged.loc[tmp_mask,'spellid'] = agged.loc[tmp_mask,'f'] + agged.loc[tmp_mask,'d'] + (agged.loc[tmp_mask,'y']-k).astype('string')

tmp = agged.groupby(['f','d','spellid'])[['tenure','censored']].max().reset_index().rename(columns={'tenure':'max_tenure','censored':'censored_spell'})
agged = pd.merge(left=agged,right=tmp,how='left',on=['f','d','spellid'])

gr_mask = np.logical_and(agged.d == agged.d.shift(-1),agged.f == agged.f.shift(-1))
gr_mask = np.logical_and(gr_mask,agged.y == agged.y.shift(-1)-1)
gr_mask = np.logical_and(gr_mask,agged.y != yT)

agged['v_f'] = agged.v.shift(-1)
agged.loc[gr_mask,'growth']=agged.loc[gr_mask,'v_f']/agged.loc[gr_mask,'v']-1.0
agged.drop('v_f',axis=1)
agged.loc[agged.y==yT,'growth']=0

agged.loc[agged.growth<-0.99,'growth']=np.nan

agged['cohort'] = agged.groupby(['f','d'])['y'].transform(lambda x:x.min())


#####################################################################
# dropping industry/year/destinations with fewer than 30 firms
tmp=agged.groupby(['d','y'])['f'].agg(lambda x: x.nunique()).reset_index().rename(columns={'f':'count_f'})

agged=pd.merge(left=agged,right=tmp,how='left',on=['d','y'])
agged=agged[agged.count_f>=30]
agged.drop(['count_f'],axis=1,inplace=True)

tmp=agged.groupby(['d'])['y'].agg(lambda x: x.nunique()).reset_index().rename(columns={'y':'count_y'})
agged=pd.merge(left=agged,right=tmp,how='left',on=['d'])
agged=agged[agged.count_y>=3]
agged.drop(['count_y'],axis=1,inplace=True)

##############################################################################################3

print('\tSaving to disk...')

agged.to_pickle(outpath + pref+ '_microdata_processed.pik')

#print('\nNumber of firms per year:')

#tmp = agged.groupby(['d']).f.agg(lambda x:x.nunique()).reset_index()

#print(tmp)

#print('\nAverage:')
#print(tmp.mean())

##############################################################################################3

print('\tComputing cross-sectional and life-cycle facts...')

def top5_share(x):
    mask = x>=(x.quantile(0.95))
    return (x[mask].sum())/(x.sum())

def reset_multiindex(df,n,suff):
    tmp = df.columns
    #levels=df.columns.levels
    #labels=df.columns.labels
    df.columns=[x[0] for x in tmp[0:n]] + [x[1]+suff for x in tmp[n:]]
    #df.columns=levels[0][labels[0][0:n]].tolist()+[s+suff for s in levels[1][labels[1][n:]].tolist()]
    return df

def key_facts(df,industry=0):

    df['exit_v'] = df['exit']*df['v']

    agg_fns={'f':[('nf',lambda x:x.nunique())],
             'v':[('avg_exports',lambda x:np.mean(x)),
                  ('top5_share',top5_share),
                  ('p05_norm_exports',lambda x: x.quantile(q=0.05)/np.mean(x)),
                  ('p10_norm_exports',lambda x: x.quantile(q=0.10)/np.mean(x)),
                  ('p25_norm_exports',lambda x: x.quantile(q=0.20)/np.mean(x)),
                  ('p50_norm_exports',lambda x: x.quantile(q=0.50)/np.mean(x)),
                  ('p75_norm_exports',lambda x: x.quantile(q=0.75)/np.mean(x)),
                  ('p90_norm_exports',lambda x: x.quantile(q=0.90)/np.mean(x)),
                  ('p95_norm_exports',lambda x: x.quantile(q=0.95)/np.mean(x)),
                  ('total_v',np.sum)],
             'nd':[('avg_nd',np.mean),
                   ('med_nd',np.median)],
             'exit':[('num_exits',np.sum)],
             'exit_v':[('exit_v',np.sum)],
             'cost':[('avg_cost',lambda x:np.mean(x))],
             'cost2':[('avg_cost2',lambda x:np.mean(x))]}
             
    entrants = df[df.entry==1]
    incumbents = df[df.incumbent==1]
             
    icols = ['d','y','popt','gdppc','tau']
    if industry==1:
        icols = ['d','y','industry','maquiladora','popt','gdppc','tau']
        
    agg = reset_multiindex(df.groupby(icols).agg(agg_fns).reset_index(),
                           len(icols),
                           '')
             
    agg_e = reset_multiindex(entrants.groupby(icols).agg(agg_fns).reset_index(),
                             len(icols),
                             '_entrant')
             
    agg_i = reset_multiindex(incumbents.groupby(icols).agg(agg_fns).reset_index(),
                             len(icols),
                             '_incumbent')

    agg = pd.merge(left=agg,right=agg_e,how='left',on=icols)
    agg = pd.merge(left=agg,right=agg_i,how='left',on=icols)

    agg['exit_v_share'] = agg.exit_v/agg.total_v
    agg['entrant_v_share'] = agg.total_v_entrant/agg.total_v
    agg['exit_rate']=agg.num_exits/agg.nf
    agg['exit_rate_entrant']=agg.num_exits_entrant/agg.nf_entrant
    agg['exit_rate_incumbent']=agg.num_exits_incumbent/agg.nf_incumbent
    agg['erel_exit_rate']=agg.exit_rate_entrant-agg.exit_rate_incumbent
    agg['erel_size']=agg.avg_exports_entrant/agg.avg_exports_incumbent
    agg['erel_nd']=agg.avg_nd_entrant- agg.avg_nd_incumbent
    agg['erel_cost']=agg.avg_cost_entrant/agg.avg_cost_incumbent
    agg['erel_cost2']=agg.avg_cost2_entrant/agg.avg_cost2_incumbent

    return agg

agg_by_d_s = key_facts(agged,0)

agg_by_d2_s = agg_by_d_s.groupby('d').mean().reset_index()

agg_by_d_s.to_pickle(outpath + pref+ '_microdata_agg_by_d.pik')
agg_by_d2_s.to_pickle(outpath + pref +'_microdata_agg_by_d2.pik')

