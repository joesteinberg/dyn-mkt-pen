import numpy as np
import pandas as pd

TMAX=25

# industry definitions
def which_industry(cnae5):
    
    industry=''
    if cnae5[0:3]=='151':
        industry='Animals'
    elif cnae5[0:3]=='152':
        industry='Vegetables'
    elif cnae5[0:3]=='153':
        industry='Fats'
    elif cnae5[0:2]=='15' or cnae5[0:2]=='16':
        industry='Mfg: Food'
    elif cnae5[0:2]=='17' or cnae5[0:2]=='18':
        industry='Mfg: Textiles'
    elif cnae5[0:3]=='191' or cnae5[0:3]=='192':
        industry='Mfg: Leather'
    elif cnae5[0:2]=='19':
        industry='Mfg: Footwear'
    elif cnae5[0:2]=='20':
        industry ='Mfg: Wood'
    elif cnae5[0:2]=='21' or cnae5[0:3]=='221' or cnae5[0:3]=='222':
        industry='Mfg: Pulp'
    elif cnae5[0:2]=='23':
        industry='Minerals'
    elif cnae5[0:2]=='24':
        industry='Mfg: Chemicals'
    elif cnae5[0:2]=='25':
        industry='Mfg: Plastics'
    elif cnae5[0:2]=='26':
        industry='Mfg: Ceramic'
    elif cnae5[0:2]=='27' or cnae5[0:2]=='28':
        industry='Mfg: Metal'
    elif cnae5[0:2]=='29':
        industry='Mfg: Machinery'
    elif cnae5[0:2]=='30' or cnae5[0:2]=='31' or cnae5[0:2]=='32' or cnae5[0:3]=='223':
        industry='Mfg: Electrical'
    elif cnae5[0:2]=='33':
        industry = 'Mfg: Optical'
    elif cnae5[0:2]=='34' or cnae5[0:2]=='35':
        industry='Mfg: Vehicles'
    elif cnae5[0:4]=='3691':
        industry='Mfg: Jewelry'
    elif cnae5[0:4]=='3693':
        industry='Mfg: Guns'
    elif cnae5[0:2]=='36':
        industry='Mfg: Misc'
    else:
        industry='Other'

    return industry

# mfg vs. non-mfg
def mfg(industry):
    return 'Mfg' in industry

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

print('Loading SECEX data for Brazil...')

# ignore great recession
years = range(1996,2008)
y0 = years[0]
yT = years[-1]

df = pd.read_csv('=../data/secex.csv',sep=',',
                 dtype={'cnpj8':str,'ano':int,'mes':int,'pais':int,'cnae2':str,'cnae5':str,'valor':float,'ncm':str,'pais':int})

df.loc[df.cnae2=='','cnae2']='-1'
df.loc[df.cnae5=='','cnae5']='-1'

x=len(df)
df=df[df.cnpj8!='']
y=len(df)
print('Dropped %d (%0.2f pct) observations without firm codes'%((x-y),100*(x-y)/x))

# merge on country iso codes, which we will use to merge on gravity data
pais2iso = pd.read_csv('../data/pais2iso.csv')
df=pd.merge(left=df,right=pais2iso,how='left',on='pais',indicator=True)

x=len(df)
df=df[df._merge!='left_only'].drop('_merge',axis=1)
y=len(df)
print('Dropped %d (%0.2f pct) observations without country ISO codes'%((x-y),100*(x-y)/x))

# get rid of non-manufacturing
x=len(df)
df['industry'] = df.cnae5.apply(lambda x: which_industry(str(x)))
df=df[df.industry.apply(mfg)].reset_index(drop=True)
y=len(df)
print('Dropped %d (%0.2f pct) non-manufacturing observations'%((x-y),100*(x-y)/x))

# rename to make consistent with WBEDD
df.rename(columns={'ano':'y','mes':'m','cnpj8':'f','iso3':'d','valor':'v'},
          inplace=True)
df=df[['y','m','f','d','v','industry']]
df['f'] = 'BRA'+df['f']

###############################################################################################
print('Computing main firm-level variables...')


# first, assign one industry constant to each firm based on its sales ranking across industries
tmp = df.groupby(['f','industry'])['v'].sum().reset_index()
industrymax = lambda x: tmp.loc[x.idxmax(),'industry']
tmp = tmp.groupby('f')['v'].apply(industrymax).reset_index().rename(columns={'v':'industry'})
df_imax = pd.merge(left=df.drop('industry',axis=1),right=tmp,how='left',on='f')

# now aggregate by f/d/y
agged = df_imax.groupby(['f','industry','d','y'])['v']\
               .sum()\
               .reset_index()\
               .sort_values(by=['f','industry','d','y'])\
               .reset_index(drop=True)

# count number of destinations each firm sells to
agged['nd'] = agged.groupby(['f','industry','y'])['d'].transform(lambda x:x.nunique()).astype(int)

agged['nd_group'] = agged.nd.apply(which_nd_group)
agged['drank'] = agged.groupby(['f','industry','y'])['v']\
                        .transform(lambda x: x.rank(ascending=False))\
                        .astype(int)
                        
# count number of months
mcnt = df_imax.groupby(['f','y','d'])['m'].nunique().reset_index()
agged = pd.merge(left=agged,right=mcnt,how='left',on=['f','y','d'])

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

##############################################################################################3
# filtering

print('Filtering dataset...')

# drop years with no entry/exit data
agged=agged[agged.y>y0]

l0=len(agged)

# drop observations with sales < 1000
agged['flag'] = agged.v<1000
tmp = agged.groupby(['f','d'])['flag'].sum().reset_index().rename(columns={'flag':'sum_flag'})
agged=pd.merge(left=agged,right=tmp,how='left',on=['f','d'])
l0=len(agged)
agged=agged[agged.flag==0]
l1=len(agged)
agged.drop(['flag','sum_flag'],axis=1,inplace=True)

# drop observations with very low relative sales
tmp = agged.groupby(['d','y'])['v'].agg([('v_p01',lambda x:x.quantile(0.01))]).reset_index()
agged=pd.merge(left=agged,right=tmp,how='left',on=['d','y'])
agged=agged[agged.v_p01.isnull()==False]
agged['flag'] = agged.v<agged.v_p01
tmp = agged.groupby(['f','d'])['flag'].sum().reset_index().rename(columns={'flag':'sum_flag'})
agged=pd.merge(left=agged,right=tmp,how='left',on=['f','d'])
agged = agged[agged.sum_flag==0]
agged.drop(['flag','sum_flag','v_p01'],axis=1,inplace=True)
l2=len(agged)

# drop firms with crazy growth rates
agged=agged[~(agged.growth>25.0)]
tmp = agged.groupby(['d','y'])['growth'].agg([('growth_p05',(lambda x:x.quantile(0.05))),
                                              ('growth_p95',(lambda x:x.quantile(0.95)))])\
                                        .reset_index()
agged=pd.merge(left=agged,right=tmp,how='left',on=['d','y'])
agged=agged[agged.growth_p95.isnull()==False]
agged=agged[agged.growth_p05.isnull()==False]
agged['flag1'] = agged.growth>agged.growth_p95
agged['flag2'] = agged.growth<agged.growth_p05
agged['flag'] = agged.flag1
tmp = agged.groupby(['f','d'])['flag'].sum().reset_index().rename(columns={'flag':'sum_flag'})
agged=pd.merge(left=agged,right=tmp,how='left',on=['f','d'])
agged = agged[agged.sum_flag==0]
agged.drop(['flag','sum_flag','growth_p95','growth_p05'],axis=1,inplace=True)
agged.loc[agged['exit']==1,'growth']=np.nan;
l3=len(agged)

# dropping industry/year/destinations with fewer than 20 firms
tmp=agged.groupby(['d','y','industry'])['f'].agg(lambda x: x.nunique()).reset_index().rename(columns={'f':'count_f'})
agged=pd.merge(left=agged,right=tmp,how='left',on=['d','y','industry'])
agged=agged[agged.count_f>=20]
agged.drop(['count_f'],axis=1,inplace=True)
l4=len(agged)

tmp=agged[agged.y.isin(range(2001,2006))].groupby(['d','industry'])['y'].agg(lambda x: x.nunique()).reset_index().rename(columns={'y':'count_y'})
agged=pd.merge(left=agged,right=tmp,how='left',on=['d','industry'])
agged=agged[agged.count_y>=3]
agged.drop(['count_y'],axis=1,inplace=True)
l5=len(agged)

# extremely high entrant size/exit rate
agged=agged[agged.d!='FIN']
agged=agged[agged.d!='AUT']
agged=agged[agged.d!='SAU']
agged=agged[agged.d!='SEN']
agged=agged[agged.d!='NAM']
l6=len(agged)

print('\t%d total (f,d,y) observations' % l0)
print('\tDropped %d observations for (f,d) pairs with exports < 1000' % (l0-l1))
print('\tDropped %d observations for (f,d) pairs with very low sales' % (l1-l2))
print('\tDropped %d observations for (f,d) pairs with very high growth rates' % (l2-l3))
print('\tDropped %d observations for (d,industry,y) with <20 unique firms' % (l3-l4))
print('\tDropped %d observations for (d,industry) with <4 years of data' % (l4-l5))
print('\tDropped %d observations for Finland, Austria, Saudi Arabia, Senegal, Namibia' % (l5-l6))

##############################################################################################3
# merge on gravity information

print('Merging on gravity indicators...')

grav = pd.read_pickle('output/pik/gravdata.pik')
grav=grav[grav.iso3_o=='BRA'].reset_index(drop=True)
grav=grav.rename(columns={'iso3_d':'d','pop_d':'popt','gdpcap_d':'gdppc','Exports':'exports','year':'y'})
merged = pd.merge(left=agged,right=grav,how='left',on=['d','y'])


##############################################################################################3

print('Saving to disk...')

merged.to_pickle('output/pik/bra_microdata_processed.pik')

print('\nNumber of destinations: %d' % (merged.d.nunique()))

print('\nNumber of firms per year:')
tmp = merged.groupby('y').f.agg(lambda x:x.nunique())

print(tmp)

print('\nAverage:')
print(tmp.mean())


##############################################################################################

print('Computing cross-sectional and life-cycle facts...')

def top5_share(x):
    mask = x>=(x.quantile(0.95))
    return (x[mask].sum())/(x.sum())

def reset_multiindex(df,n,suff):
    tmp = df.columns
    df.columns=[x[0] for x in tmp[0:n]] + [x[1]+suff for x in tmp[n:]]
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
             'exit_v':[('exit_v',np.sum)]}
             
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

    agg['exit_v_share'] = agg.exit_v/agg.total_v
    agg['entrant_v_share'] = agg.total_v_entrant/agg.total_v
    agg['exit_rate']=agg.num_exits/agg.nf
    agg['exit_rate_entrant']=agg.num_exits_entrant/agg.nf_entrant
    agg['exit_rate_incumbent']=agg.num_exits_incumbent/agg.nf_incumbent
    agg['erel_exit_rate']=agg.exit_rate_entrant-agg.exit_rate_incumbent
    agg['erel_size']=agg.avg_exports_entrant/agg.avg_exports_incumbent
    agg['erel_nd']=agg.avg_nd_entrant- agg.avg_nd_incumbent
    
    return agg

merged['c'] = 'BRA'
merged['maquiladora'] = False

agg_by_d = key_facts(merged,0)
agg_by_d_i = key_facts(merged,1)
agg_by_d2 = agg_by_d.groupby(['c','d']).mean().reset_index()

# normalize destination characteristics to 1 for usa
d_usa = agg_by_d2.loc[agg_by_d2.d=='USA']
agg_by_d2['popt']=agg_by_d2.popt/(d_usa.popt.values[0])
agg_by_d2['gdppc']=agg_by_d2.gdppc/(d_usa.gdppc.values[0])
agg_by_d2['tau']=agg_by_d2.tau/(d_usa.tau.values[0])
agg_by_d['popt']=agg_by_d.popt/(d_usa.popt.values[0])
agg_by_d['gdppc']=agg_by_d.gdppc/(d_usa.gdppc.values[0])
agg_by_d['tau']=agg_by_d.tau/(d_usa.tau.values[0])
agg_by_d_i['popt']=agg_by_d_i.popt/(d_usa.popt.values[0])
agg_by_d_i['gdppc']=agg_by_d_i.gdppc/(d_usa.gdppc.values[0])
agg_by_d_i['gdppc']=agg_by_d_i.tau/(d_usa.tau.values[0])

# save output
agg_by_d2.to_csv('output/calibration/dests_for_c_program.txt',sep=' ',header=False,index=False,
                 columns=['d','popt','gdppc','tau'])

agg_by_d.to_pickle('output/pik/bra_microdata_agg_by_d.pik')
agg_by_d_i.to_pickle('output/pik/bra_microdata_agg_by_d_i.pik')
agg_by_d2.to_pickle('output/pik/bra_microdata_agg_by_d2.pik')


#########################

tmp = merged.groupby(['f','nd_group','y'])['exit'].min().reset_index()
tmp2 = tmp.groupby('y')['exit'].mean().reset_index()
mu = tmp2['exit'].mean()
se = tmp2['exit'].std()/np.sqrt(len(tmp2))
print('Multilateral exit rate: %0.4g (%0.4g)' %(mu,se))


tmp2 = tmp.groupby(['y','nd_group'])['exit'].mean().reset_index()
tmp3 = tmp2.groupby('nd_group').agg({'exit':[('mean',lambda x:np.mean(x)),('se',lambda x:np.std(x)/np.sqrt(len(x)))]})
print('By ND group')
print(tmp3)



