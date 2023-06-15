import numpy as np
import pandas as pd

##############################################################################################3
# utility functions

TMAX=50

# hs12 --> hs2
def hs2(hs12):
    x=len(hs12)
    y=0
    if x==5:
        y=int(hs12[0])
    else:
        y=int(hs12[0:2])
    return y

# industry definitions
def which_industry(hs2):
    industry=''
    if hs2<=5:
        industry='Animals'
    elif hs2<=14:
        industry='Vegetables'
    elif hs2==15:
        industry='Fats'
    elif hs2 <=24:
        industry='Mfg: Food'
    elif hs2<=27:
        industry='Minerals'
    elif hs2<=38:
        industry='Mfg: Chemicals'
    elif hs2<=40:
        industry='Mfg: Plastics'
    elif hs2<=43:
        industry='Mfg: Leather'
    elif hs2<=46:
        industry='Mfg: Wood'
    elif hs2<=49:
        industry='Mfg: Pulp'
    elif hs2<=63:
        industry='Mfg: Textiles'
    elif hs2<=67:
        industry='Mfg: Footwear'
    elif hs2<=70:
        industry='Mfg: Ceramic'
    elif hs2==71:
        industry='Mfg: Jewelry'
    elif hs2==72:
        industry='Mfg: Metal'
    elif hs2<=83:
        industry='Mfg: Metal'
    elif hs2<=84:
        industry='Mfg: Machinery'
    elif hs2<=85:
        industry='Mfg: Electrical'
    elif hs2<=89:
        industry='Mfg: Vehicles'
    elif hs2<=92:
        industry='Mfg: Optical'
    elif hs2==93:
        industry='Mfg: Guns'
    elif hs2<=96:
        industry='Mfg: Misc'
    elif hs2==97:
        industry='Art'
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

# data dictionary:
# c: exporter country
# y: year
# f: firm ID
# hs: HS code
# d: destination
# v: value
# q: quantity

# Three African countries not used due to requirement of 4 years of data per destination:
# YEM runs from 2008-2012. Break in 2008 so could only use 2009-2012.
# MWI runs from 2006-2012. Break in 2008 so could only use 2006-2007 and 2009-2012.
# BFA runs from 2005-2012. Break in 2007 so could only use 2008-2012.
countries = ['MEX','PER']
df=None
cnt=0

consol = pd.read_stata('../../../data/consolidation.dta')

print('Loading World Bank Exporter Dynamics microdata for...')

for country in countries:
    print('\t'+country)
    
    tmp = pd.read_stata('../../../data/%s.dta'%country)

    # drop firms with 'OTH' as firm code
    tmp=tmp[tmp.f!='OTH']

    # drop OTH destinations
    tmp=tmp[tmp.d!='OTH']

    # MEX: use data only for 2000-2007 because of changes in firm ID coding in 2007, 2009, 2011
    if country=='MEX':
        tmp['c']='MEX'
        tmp=tmp[tmp.y<2007]

    # use consolidated hs codes provided by WB
    tmp = pd.merge(left=tmp,right=consol,how='left',on='hs',indicator=True)
    mask = tmp._merge=='left_only'
    tmp.loc[mask,'h'] = tmp.loc[mask,'hs']

    # coarserssectoral aggregation
    tmp['hs2']=tmp.h.str[0:2]
    tmp.loc[tmp.hs2=='OT','hs2']='99'
    tmp['hs2']=tmp.hs2.astype(int)
    tmp['industry']=tmp.hs2.apply(which_industry)
    tmp=tmp[tmp.industry.apply(mfg)]
    
    tmp['maquiladora']=False
    if country=='MEX':
        tmp.loc[tmp.hs2==61,'maquiladora']=True # knitted/non-knitted apparel
        tmp.loc[tmp.hs2==62,'maquiladora']=True # "
        tmp.loc[tmp.hs2==39,'maquiladora']=True # plastics
        tmp.loc[tmp.hs2==84,'maquiladora']=True # machinery
        tmp.loc[tmp.hs2==85,'maquiladora']=True # electrical machinery
        tmp.loc[tmp.hs2==87,'maquiladora']=True # car parts
        tmp.loc[tmp.hs2==90,'maquiladora']=True # medical instruments
        tmp.loc[tmp.hs2==94,'maquiladora']=True # furniture
    
    tmp['y0'] = tmp.y.min()
    tmp['yT'] = tmp.y.max()
    
    tmp=tmp[['c','y0','yT','f','industry','h','maquiladora','d','y','v']].reset_index(drop=True)

    if cnt==0:
        df=tmp
    else:
        df=df.append(tmp)

    cnt+=1
    
##############################################################################################3

print('Computing main firm-level variables...')

# first, assign one industry constant to each firm based on its sales ranking across industries
tmp = df.groupby(['c','f','industry'])['v'].sum().reset_index()
industrymax = lambda x: tmp.loc[x.idxmax(),'industry']
tmp = tmp.groupby(['c','f'])['v'].apply(industrymax).reset_index().rename(columns={'v':'industry'})
merged = pd.merge(left=df.drop('industry',axis=1),right=tmp,how='left',on=['c','f'])

tmp2 = df.groupby(['c','f'])['maquiladora'].agg(lambda x:x.sum()>1).reset_index()
merged = pd.merge(left=merged.drop('maquiladora',axis=1),right=tmp2,how='left',on=['c','f'])

# ensure firm codes are unique
merged['f'] = merged.f

# now aggregate by f/d/y
merged = merged.groupby(['c','y0','yT','f','industry','maquiladora','d','y'])['v']\
           .sum()\
           .reset_index()\
           .sort_values(by=['c','f','industry','d','y'])\
           .reset_index(drop=True)

# count number of destinations each firm sells to
merged['nd'] = merged.groupby(['c','f','industry','y'])['d'].transform(lambda x:x.nunique()).astype(int)
merged['nd_group'] = merged.nd.apply(which_nd_group)
merged['drank'] = merged.groupby(['c','f','industry','y'])['v']\
                        .transform(lambda x: x.rank(ascending=False))\
                        .astype(int)
#merged.loc[merged.drank>=10,'drank']=10
#merged.loc[merged.drank.isin(range(5,10)),'drank'] = 6
#test=merged.loc[merged.drank>merged.nd_group]


# entry, exit
merged.sort_values(['c','f','d','y'],ascending=[True,True,True,True],inplace=True)
merged.reset_index(drop=True)
merged['entry']=0
merged['incumbent']=0
merged['exit']=0
merged['growth']=np.nan

# define entry as observation where:
# 1. first occurence of firm f OR first occurence of dest d in firm f's block OR year != previous year;
# 2. and year is not 2000 (because everyone would be an entrant in the first year of data
#merged.y0 = merged.groupby('c')['y'].transform(lambda x: x.min())
#merged.yT = merged.groupby('c')['y'].transform(lambda x: x.max())

entry_mask = np.logical_or(merged.f!=merged.f.shift(),merged.d!=merged.d.shift())
entry_mask = np.logical_or(entry_mask,merged.y != merged.y.shift()+1)
entry_mask = np.logical_and(entry_mask,merged.y!=merged.y0)

incumbent_mask = np.logical_and(merged.d==merged.d.shift(),merged.f==merged.f.shift())
incumbent_mask = np.logical_and(incumbent_mask,merged.y==merged.y.shift()+1)
incumbent_mask = np.logical_and(incumbent_mask,merged.y!=merged.y0)

# define exit as the opposite
exit_mask = np.logical_or(merged.d != merged.d.shift(-1),merged.f != merged.f.shift(-1))
exit_mask = np.logical_or(exit_mask,merged.y != merged.y.shift(-1)-1)
exit_mask = np.logical_and(exit_mask,merged.y != merged.yT)

merged.loc[entry_mask,'entry']=1
merged.loc[incumbent_mask,'incumbent']=1
merged.loc[exit_mask,'exit']=1

merged['censored']=0
merged.loc[np.logical_and(incumbent_mask,merged.y==merged.yT),'censored']=1

merged['tenure']=np.nan
merged['spellid']=np.nan
merged.loc[merged.entry==1,'tenure']=0
merged.loc[merged.entry==1,'tenure']=0
merged.loc[merged.entry==1,'spellid'] = merged.loc[merged.entry==1,'f'] + merged.loc[merged.entry==1,'d'] + merged.loc[merged.entry==1,'y'].astype('string')

for k in range(1,TMAX+1):
    tmp_mask = np.logical_and(merged.tenure.shift()==k-1,incumbent_mask)
    merged.loc[tmp_mask,'tenure']=k
    merged.loc[tmp_mask,'spellid'] = merged.loc[tmp_mask,'f'] + merged.loc[tmp_mask,'d'] + (merged.loc[tmp_mask,'y']-k).astype('string')

tmp = merged.groupby(['c','f','d','spellid'])[['tenure','censored']].max().reset_index().rename(columns={'tenure':'max_tenure','censored':'censored_spell'})
merged = pd.merge(left=merged,right=tmp,how='left',on=['c','f','d','spellid'])
        
gr_mask = np.logical_and(merged.d == merged.d.shift(-1),merged.f == merged.f.shift(-1))
gr_mask = np.logical_and(gr_mask,merged.y == merged.y.shift(-1)-1)
gr_mask = np.logical_and(gr_mask,merged.y != merged.yT)

merged['v_f'] = merged.v.shift(-1)
merged.loc[gr_mask,'growth']=merged.loc[gr_mask,'v_f']/merged.loc[gr_mask,'v']-1.0
merged.drop('v_f',axis=1)

merged.loc[merged.growth<-0.99,'growth']=np.nan

merged['cohort'] = merged.groupby(['c','f','d'])['y'].transform(lambda x:x.min())

##############################################################################################3
# maquiladoras

#mex2 = merged.loc[(merged.c=='MEX') & (merged.maquiladora==0),:].reset_index()
#mex2['c'] = 'MEX2'
#merged = merged.append(mex2)

##############################################################################################3
# filtering

print('Dropping outliers...')

# drop years with no entry/exit data
merged=merged[merged.y>merged.y0]
merged=merged[merged.y<merged.yT]

l0=len(merged)

# drop observations with sales < 1000
merged['flag'] = merged.v<1000
tmp = merged.groupby(['c','f','d'])['flag'].sum().reset_index().rename(columns={'flag':'sum_flag'})
merged=pd.merge(left=merged,right=tmp,how='left',on=['c','f','d'])
l0=len(merged)
merged=merged[merged.flag==0]
l1=len(merged)
merged.drop(['flag','sum_flag'],axis=1,inplace=True)

# drop observations with very low relative sales
tmp = merged.groupby(['c','d','y'])['v']\
            .agg([('v_p01',(lambda x:x.quantile(0.01)))])\
            .reset_index()
merged=pd.merge(left=merged,right=tmp,how='left',on=['c','d','y'])
merged=merged[merged.v_p01.isnull()==False]
merged['flag'] = merged.v<merged.v_p01
tmp = merged.groupby(['c','f','d'])['flag'].sum().reset_index().rename(columns={'flag':'sum_flag'})
merged=pd.merge(left=merged,right=tmp,how='left',on=['c','f','d'])
merged = merged[merged.sum_flag==0]
merged.drop(['flag','sum_flag','v_p01'],axis=1,inplace=True)
l2=len(merged)

# drop firms with crazy growth rates
merged=merged[~(merged.growth>25.0)]
tmp = merged.groupby(['c','d','y'])['growth']\
            .agg([('growth_p05',(lambda x:x.quantile(0.05))),
                  ('growth_p95',(lambda x:x.quantile(0.95)))])\
            .reset_index()
merged=pd.merge(left=merged,right=tmp,how='left',on=['c','d','y'])
merged=merged[merged.growth_p95.isnull()==False]
merged=merged[merged.growth_p05.isnull()==False]
merged['flag1'] = merged.growth>merged.growth_p95
merged['flag2'] = merged.growth<merged.growth_p05
merged['flag'] = merged.flag1
tmp = merged.groupby(['c','f','d'])['flag'].sum().reset_index().rename(columns={'flag':'sum_flag'})
merged=pd.merge(left=merged,right=tmp,how='left',on=['c','f','d'])
merged = merged[merged.sum_flag==0]
merged.drop(['flag','sum_flag','growth_p95','growth_p05'],axis=1,inplace=True)
merged.loc[merged['exit']==1,'growth']=np.nan;
l3=len(merged)

# dropping industry/year/destinations with fewer than 20 firms
#tmp=merged.groupby(['c','d','y','industry'])['f'].agg(lambda x: x.nunique()).reset_index().rename(columns={'f':'count_f'})
#merged=pd.merge(left=merged,right=tmp,how='left',on=['c','d','y','industry'])
#merged=merged[merged.count_f>=20]
#merged.drop(['count_f'],axis=1,inplace=True)
l4=len(merged)

tmp=merged[merged.y.isin(range(2001,2006))].groupby(['c','d','industry'])['y'].agg(lambda x: x.nunique()).reset_index().rename(columns={'y':'count_y'})
merged=pd.merge(left=merged,right=tmp,how='left',on=['c','d','industry'])
merged=merged[merged.count_y>=3]
merged.drop(['count_y'],axis=1,inplace=True)
l5=len(merged)

# very high exit rate
merged=merged[~( (merged.c=='PER') & (merged.d=='LTU'))]
merged=merged[~( (merged.c=='PER') & (merged.d=='RUS'))]
merged=merged[~( (merged.c=='PER') & (merged.d=='CUB'))]
merged=merged[~( (merged.c=='PER') & (merged.d=='GRC'))]
merged=merged[~( (merged.c=='MEX') & (merged.d=='KWT'))]
merged=merged[~( (merged.c=='MEX') & (merged.d=='LBN'))]
merged=merged[~( (merged.c=='MEX') & (merged.d=='PAK'))]
merged=merged[~( (merged.c=='MEX') & (merged.d=='DMA'))]
merged=merged[~( (merged.c=='MEX') & (merged.d=='SUR'))]
merged=merged[~( (merged.c=='PER') & (merged.d=='AUT'))]
merged=merged[~( (merged.c=='MEX') & (merged.d=='NOR'))]
l6=len(merged)

print('\t%d total (f,d,y) observations' % l0)
print('\tDropped %d observations for (f,d) pairs with exports < 1000' % (l0-l1))
print('\tDropped %d observations for (f,d) pairs with very low sales' % (l1-l2))
print('\tDropped %d observations for (f,d) pairs with very high growth rates' % (l2-l3))
print('\tDropped %d observations for (d,industry,y) with <20 unique firms' % (l3-l4))
print('\tDropped %d observations for (d,industry) with <4 years of data' % (l4-l5))
print('\tDropped %d observations for BGR/IRQ' % (l5-l6))

##############################################################################################3
# merge on gravity information

print('Merging on gravity indicators...')

grav = pd.read_pickle('../output/pik/gravdata.pik')
grav=grav.rename(columns={'iso3_o':'c',
                          'iso3_d':'d',
                          'year':'y',
                          'pop_d':'popt',
                          'Exports':'exports',
                          'gdpcap_d':'gdppc'})

#grav_m2 = grav.loc[grav.c=='MEX',:].reset_index()
#grav_m2['c']='MEX2'
#grav = grav.append(grav_m2)

merged2 = pd.merge(left=merged,right=grav,how='left',on=['c','d','y'])
merged2['v_norm'] = merged2.v/(merged2.popt*merged2.gdppc)

##############################################################################################3

print('Saving to disk...')
merged2.to_pickle('output/wbedd_microdata_processed.pik')

print('\nNumber of firms per year:')
tmp = merged2.groupby(['c','y']).f.agg(lambda x:x.nunique())
print(tmp)

##############################################################################################3

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
             #'v_norm':[('avg_exports_norm',np.mean),
             #          ('med_exports_norm',np.median)],
             'v':[('avg_exports',lambda x:np.mean(x)/1000),
                  ('top5_share',top5_share),
                  ('med_exports',np.median),
                  ('total_v',np.sum)],
             'nd':[('avg_nd',np.mean),
                   ('med_nd',np.median)],
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

    agg = agg.loc[agg.nf>20,:].reset_index(drop=True)
    
    #        agg['exit_v_share'] = agg.exit_v/agg.total_v
    agg['exit_rate']=agg.num_exits/agg.nf
    #        agg['exit_v_share_entrant'] = agg.exit_v_entrant/agg.total_v_entrant
    agg['exit_rate_entrant']=agg.num_exits_entrant/agg.nf_entrant
    agg['exit_rate_incumbent']=agg.num_exits_incumbent/agg.nf_incumbent
    agg['erel_exit_rate']=agg.exit_rate_entrant-agg.exit_rate_incumbent
    #        agg['erel_exit_v_share'] = agg.exit_v_share_entrant-agg.exit_v_share
    #        agg['erel_exit_rate_w']=agg.exit_rate_w_entrant-agg.exit_rate_w_incumbent
    agg['erel_size']=agg.avg_exports_entrant/agg.avg_exports_incumbent
    agg['erel_size_med']=agg.med_exports_entrant/agg.med_exports_incumbent
    #        agg['erel_growth']=agg.avg_growth_entrant- agg.avg_growth_incumbent
    #        agg['erel_growth_med']=agg.med_growth_entrant- agg.med_growth_incumbent
    agg['erel_nd']=agg.avg_nd_entrant- agg.avg_nd_incumbent
    agg['erel_nd2']=agg.med_nd_entrant- agg.med_nd_incumbent

    return agg

agg_by_d = key_facts(merged2,0)
agg_by_d_i = key_facts(merged2,1)
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
#agg_by_d2.to_csv('output/dests_for_c_program.txt',sep=' ',header=False,index=False,
#                 columns=['d','popt','gdppc','tau'])

agg_by_d.to_pickle('output/wbedd_microdata_agg_by_d.pik')
agg_by_d_i.to_pickle('output/wbedd_microdata_agg_by_d_i.pik')
agg_by_d2.to_pickle('output/wbedd_microdata_agg_by_d2.pik')

#########################################################################

tmp = merged2.groupby(['c','f','nd_group','y'])['exit'].min().reset_index()
tmp2 = tmp.groupby(['c','y'])['exit'].mean().reset_index()
mu = tmp2.groupby('c')['exit'].mean()
se = tmp2.groupby('c')['exit'].agg(lambda x: x.std()/np.sqrt(len(x)))

print('Multilateral exit rate: %0.4g (%0.4g)')
print(mu,se)


tmp2 = tmp.groupby(['c','y','nd_group'])['exit'].mean().reset_index()
tmp3 = tmp2.groupby(['c','nd_group']).agg({'exit':[('mean',lambda x:np.mean(x)),('se',lambda x:np.std(x)/np.sqrt(len(x)))]})
print('By ND group')
print(tmp3)


