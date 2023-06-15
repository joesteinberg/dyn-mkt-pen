import numpy as np
import pandas as pd
import wbdata
from statsmodels.formula.api import ols

####################################################################################
# load all of the data

# ----------------------------------------------------------------------------------
print('Loading CEPII Gravity database...')

grav = pd.read_stata('../../data/gravdata.dta')

# ----------------------------------------------------------------------------------
print('Merging on aggregate bilateral tariff data from TRAINS...')

tau = pd.read_csv('../../data/TRAINS_agg_bilateral_tariffs.csv',sep=',',engine='python')\
        .rename(columns={'Reporter Name':'d_country',
                         'Partner Name':'o_country',
                         'Tariff Year':'year',
                         'Simple Average':'tariff'})

tau = tau[['d_country','o_country','year','tariff']]\
      .reset_index(drop=True)\

tau['tariff']=tau.tariff/100.0

clist = wbdata.get_country(display=False)
codes = [i['id'] for i in clist]
names = [i['name'] for i in clist]
countries = pd.DataFrame({'code':codes,'country':names})
extras = {'Cape Verde':'CPV',
          'Venezuela':'VEN',
          'Brunei':'BRN',
          'Hong Kong, China':'HKG',
          'Swaziland':'SWZ',
          'Serbia, FR(Serbia/Montenegro)':'SRB',
          'Yemen':'YEM',
          'Ethiopia(excludes Eritrea':'ETH'}

tmp1 = pd.merge(left=tau,right=countries,how='left',left_on=['d_country'],right_on=['country'])
tmp1 = tmp1.rename(columns={'code':'iso3_d'}).drop('country',axis=1)

tmp2 = pd.merge(left=tmp1,right=countries,how='left',left_on=['o_country'],right_on=['country'])
tmp2 = tmp2.rename(columns={'code':'iso3_o'})

for e in extras.keys():
    tmp2.loc[tmp2.d_country==e,'iso3_d']=extras[e]

for e in extras.keys():
    tmp2.loc[tmp2.o_country==e,'iso3_o']=extras[e]

tmp2 = tmp2.drop(['d_country','o_country','country'],axis=1)

merged = pd.merge(left=grav,right=tmp2,how='left',on=['iso3_o','iso3_d','year'])

# ----------------------------------------------------------------------------------
print('Merging on aggregate bilateral trade data from DOTS...')

trade = pd.read_csv('../../data/DOT_06-15-2020 23-46-15-59.csv',low_memory=False)\
        .rename(columns={'Country Name':'o_country',
                         'Counterpart Country Name':'d_country',
                         'Value':'Exports',
                         'Time Period':'year'})

trade=trade[['o_country','d_country','Exports','year']][trade['Indicator Code']=='TXG_FOB_USD']

extras = {'China, P.R.: Hong Kong':'HKG',
          'Armenia, Republic of':'ARM',
          'Afghanistan, Republic of':'AFG',
          'Bahrain, Kingdom of':'BHR',
          'Azerbaijan, Republic of':'AZE',
          'Congo, Republic of':'COG',
          'Congo, Democratic Republic of':'COD',
          "C\xc3\xb4te d'Ivoire":'CIV',
          'Egypt':'EGY',
          'China, P.R.: Mainland':'CHN',
          'Iran, Islamic Republic of':'IRN',
          'Korea, Republic of':'KOR',
          "Lao People's Democratic Republic":'LAO',
          'Venezuela, Republica Bolivariana de':'VEN',
          'Yenem, Republic of':'YEM',
          'S\xc3\xa3o Tom\xc3\xa9 & Pr\xc3\xadncipe':'SAO',
          'Taiwan, Province of China':'TWN',
          'Serbia and Montenegro':'SRB',
          'Kosovo, Republic of':'UVK'}

tmp1 = pd.merge(left=trade,right=countries,how='left',left_on=['d_country'],right_on=['country'])
tmp1 = tmp1.rename(columns={'code':'iso3_d'}).drop('country',axis=1)

tmp2 = pd.merge(left=tmp1,right=countries,how='left',left_on=['o_country'],right_on=['country'])
tmp2 = tmp2.rename(columns={'code':'iso3_o'})

for e in extras.keys():
    tmp2.loc[tmp2.d_country==e,'iso3_d']=extras[e]

for e in extras.keys():
    tmp2.loc[tmp2.o_country==e,'iso3_o']=extras[e]

tmp2 = tmp2.drop(['d_country','o_country','country'],axis=1)

merged2 = pd.merge(left=merged,right=tmp2,how='left',on=['iso3_o','iso3_d','year'])

####################################################################################
# filter

merged2 = merged2[pd.notnull(merged2.iso3_d)]
merged2 = merged2[pd.notnull(merged2.iso3_o)]
merged2 = merged2[pd.notnull(merged2.Exports)]
merged2 = merged2[pd.notnull(merged2.distw)]
merged2 = merged2[pd.notnull(merged2.gdpcap_d)]
merged2 = merged2[pd.notnull(merged2.gdpcap_o)]
merged2 = merged2[pd.notnull(merged2.pop_d)]
merged2 = merged2[pd.notnull(merged2.pop_o)]
merged2 = merged2[~(merged2.iso3_o.isin(['LUX','SMR']))]
merged2 = merged2.reset_index(drop=True)
                  
####################################################################################
# run gravity regression

formula='np.log(Exports) ~ np.log(gdpcap_d) + np.log(gdpcap_o) + np.log(pop_d) + np.log(pop_o)'

reg = ols(formula=formula,data=merged2).fit()
print(reg.summary())


####################################################################################
# compute non-tariff barriers

merged2['tau'] = np.exp(-reg.resid)

####################################################################################
# write output files

merged2[['iso3_o','iso3_d','year','Exports','pop_d','gdpcap_d','tau']].to_pickle('output/pik/gravdata.pik')
