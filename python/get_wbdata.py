import wbdata
import datetime
import pandas as pd

wbcols = {'PA.NUS.FCRF':'NER',
          'FP.CPI.TOTL':'CPI',
          'NY.GDP.MKTP.CD':'NGDP_USD',
          'NY.GDP.MKTP.CN':'NGDP_LCU',
          'NY.GDP.MKTP.KD':'RGDP_USD',
          'NY.GDP.MKTP.KN':'RGDP_LCU',
          'NE.IMP.GNFS.CD':'Imports_USD',
          'NE.IMP.GNFS.CN':'Imports_LCU'}

data_date = datetime.datetime(1990, 1, 1), datetime.datetime(2010, 1, 1)
controls = wbdata.get_dataframe(indicators=wbcols,country="all",data_date=data_date,convert_date=False,keep_levels=False).reset_index()

clist = wbdata.get_country("all",display=False)
ids = [c['id'] for c in clist]
names = [c['name'] for c in clist]
countries = pd.DataFrame({'d':ids,'country':names})
controls = pd.merge(left=controls,right=countries,how='left',on='country')

controls.rename(columns={'date':'y'},inplace=True)
controls['y'] = controls['y'].astype(int)
controls.to_pickle('output/wdi_data.pik')

