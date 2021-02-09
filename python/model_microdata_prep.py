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



inpath='/home/joseph/Research/ongoing_projects/dyn_mkt_pen/v2/programs/c/output/'
outpath='/home/joseph/Research/ongoing_projects/dyn_mkt_pen/v2/programs/python/output/'

fname=''
pref=''
if len(sys.argv)==1 or (len(sys.argv)>1 and sys.argv[1]=='dmp'):
    fname='model_microdata_calibration.csv'
    pref='model'
elif len(sys.argv)>1 and sys.argv[1]=='sunk':
    fname='sunkcost_microdata.csv'
    pref='sunkcost'
elif len(sys.argv)>1 and sys.argv[1]=='sunk2':
    fname='sunkcost2_microdata.csv'
    pref='sunkcost2'
elif len(sys.argv)>1 and sys.argv[1]=='acr':
    fname='acr_microdata.csv'
    pref='acr'
elif len(sys.argv)>1 and sys.argv[1]=='acr2':
    fname='acr2_microdata.csv'
    pref='acr2'

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
        'iz':int}

agged = pd.read_csv(inpath + fname, engine='c', dtype=dtypes, sep=',')
y0 = agged.y.min()
yT = agged.y.max()

###############################################################################################
#print('\tComputing main firm-level variables...')

# count number of destinations each firm sells to
#agged['nd2'] = agged.groupby(['f','y'])['d'].transform(lambda x:x.nunique()).astype(int)
#agged['nd'] = agged.groupby(['f','y'])['d'].transform(lambda x: fast_nunique).astype(int)
#print('test1')

#agged['nd_group2'] = agged.nd2.apply(which_nd_group)
#print('test2')


#.transform(lambda x: fast_rank(x.values))
#agged['drank'] = agged.groupby(['f','y'])['v']\
#                      .transform(lambda x: x.rank(ascending=False))\
#                      .astype(int)
#print('test3 %d'%((b-a).seconds))

#agged['drank2'] = agged.groupby(['f','y'])['v']\
#                      .transform(lambda x: rnk(x.values))\
#                      .astype(int)
#print('test3a %d'%((b-a).seconds))


#agged.loc[agged.drank>=10,'drank']=10
#agged.loc[agged.drank.isin(range(5,10)),'drank'] = 6

# entry, exit
#agged.sort_values(['f','d','y'],ascending=[True,True,True],inplace=True)
#agged.reset_index(drop=True)

#agged['entry']=0
#agged['incumbent']=0
#agged['exit']=0
#agged['growth']=np.nan

#agged['fl']=agged.f.shift()
#agged['yl']=agged.y.shift()
#agged['dl']=agged.d.shift()
#agged['fp']=agged.f.shift(-1)
#agged['yp']=agged.y.shift(-1)
#agged['dp']=agged.d.shift(-1)


# define entry as observation where:
# 1. first occurence of firm f OR first occurence of dest d in firm f's block OR year != previous year;
# 2. and year is not 2000 (because everyone would be an entrant in the first year of data
#entry_mask = np.logical_or(agged.f!=agged.f.shift(),agged.d!=agged.d.shift())
#entry_mask = np.logical_or(entry_mask,agged.y != agged.y.shift()+1)
#entry_mask = np.logical_and(entry_mask,agged.y!=y0)

#incumbent_mask = np.logical_and(agged.d==agged.d.shift(),agged.f==agged.f.shift())
#incumbent_mask = np.logical_and(incumbent_mask,agged.y==agged.y.shift()+1)
#incumbent_mask = np.logical_and(incumbent_mask,agged.y!=y0)

# define exit as the opposite
#exit_mask = np.logical_or(agged.d != agged.d.shift(-1),agged.f != agged.f.shift(-1))
#exit_mask = np.logical_or(exit_mask,agged.y != agged.y.shift(-1)-1)
#exit_mask = np.logical_and(exit_mask,agged.y != yT)

#agged.loc[entry_mask,'entry']=1
#agged.loc[incumbent_mask,'incumbent']=1
#agged.loc[exit_mask,'exit']=1

#agged.loc[agged.query('(f!=fl | d!=dl | y!=yl+1) & y!=@y0'),'entry']=1
#agged.loc[agged.query('(f==fl & d==dl & y==yl+1 & y!=@y0'),'incumbent']=1
#agged.loc[agged.query('(d!=dl | f!=fl | y!=yp-1) & y!=@yT'),'exit']=1

#agged['tenure']=np.nan
#agged.loc[agged.entry==1,'tenure']=0

#for k in range(1,TMAX+1):
#    agged.loc[np.logical_and(agged.tenure.shift()==k-1,incumbent_mask),'tenure']=k

#tmp = agged.groupby(['f','d'])['tenure'].max().reset_index().rename(columns={'tenure':'max_tenure'})
#agged = pd.merge(left=agged,right=tmp,how='left',on=['f','d'])
    
#gr_mask = np.logical_and(agged.d == agged.d.shift(-1),agged.f == agged.f.shift(-1))
#gr_mask = np.logical_and(gr_mask,agged.y == agged.y.shift(-1)-1)
#gr_mask = np.logical_and(gr_mask,agged.y != yT)

#agged['v_f'] = agged.v.shift(-1)
#agged.loc[gr_mask,'growth']=agged.loc[gr_mask,'v_f']/agged.loc[gr_mask,'v']-1.0
#agged.drop('v_f',axis=1)

#agged.loc[agged.growth<-0.99,'growth']=np.nan

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

 #   df['exit_v'] = df['exit']*df['v']
    
    agg_fns={'f':[('nf',lambda x:x.nunique())],
             #'v_norm':[('avg_exports_norm',np.mean),
             #          ('med_exports_norm',np.median)],
             'v':[('avg_exports',lambda x:np.mean(x)/1000),
                  ('top5_share',top5_share)],
#             'v':[('top5_share',top5_share)],
 #                 ('med_exports',np.median),
#                  ('total_v',np.sum)],
             'nd':[('avg_nd',np.mean)],
#                   ('med_nd',np.median)],
             'exit':[('num_exits',np.sum)]}
                     #('exit_rate_w',lambda x:np.average(x,weights=df.loc[x.index,"v"]))]}
    
    #                'exit_v':[('exit_v',np.sum)],
    #                'growth':[('avg_growth', lambda x: x[x>-0.99].mean()),
    #                          ('med_growth', lambda x:x[x>-0.99].median())]}
             
    entrants = df[df.entry==1]
    incumbents = df[df.incumbent==1]
             
    icols = ['c','d','y','popt','gdppc','tau']
    if industry==1:
        icols = ['c','d','y','industry','maquiladora','popt','gdppc','tau']
        
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

    #        agg['exit_v_share'] = agg.exit_v/agg.total_v
    agg['exit_rate']=agg.num_exits/agg.nf
    #        agg['exit_v_share_entrant'] = agg.exit_v_entrant/agg.total_v_entrant
    agg['exit_rate_entrant']=agg.num_exits_entrant/agg.nf_entrant
    agg['exit_rate_incumbent']=agg.num_exits_incumbent/agg.nf_incumbent
    agg['erel_exit_rate']=agg.exit_rate_entrant-agg.exit_rate_incumbent
    #        agg['erel_exit_v_share'] = agg.exit_v_share_entrant-agg.exit_v_share
    #        agg['erel_exit_rate_w']=agg.exit_rate_w_entrant-agg.exit_rate_w_incumbent
    agg['erel_size']=agg.avg_exports_entrant/agg.avg_exports_incumbent
    #agg['erel_size_med']=agg.med_exports_entrant/agg.med_exports_incumbent
    #        agg['erel_growth']=agg.avg_growth_entrant- agg.avg_growth_incumbent
    #        agg['erel_growth_med']=agg.med_growth_entrant- agg.med_growth_incumbent
    agg['erel_nd']=agg.avg_nd_entrant- agg.avg_nd_incumbent
 #   agg['erel_nd2']=agg.med_nd_entrant- agg.med_nd_incumbent

    return agg

agged['c']='BRA'

agg_by_d_s = key_facts(agged,0)

agg_by_d2_s = agg_by_d_s.groupby(['c','d']).mean().reset_index()

agg_by_d_s.to_pickle(outpath + pref+ '_microdata_agg_by_d.pik')
agg_by_d2_s.to_pickle(outpath + pref +'_microdata_agg_by_d2.pik')

