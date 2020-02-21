import pandas as pd
import pandas_datareader as pdr
from datetime import datetime

start = datetime(2000, 1, 1)
end = datetime(2020, 1, 1)

code_df = pd.read_html('http://kind.krx.co.kr/corpgeneral/corpList.do?method=download', header=0)[0]

code_df = code_df[['회사명', '종목코드']]
code_df = code_df.rename(columns={'회사명' : 'name', '종목코드' : 'code'})
code_df.code = code_df.code.map('{:06d}'.format)

code = code_df.query("name=='{}'".format('삼성전자'))['code'].to_string(index=False)
code = code.strip()
code = code + '.KS'

df = pdr.get_data_yahoo(code, start, end)

df.to_csv('sample.csv')