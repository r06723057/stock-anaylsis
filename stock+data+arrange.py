
# coding: utf-8

# In[221]:

import numpy as np
np.set_printoptions(threshold=np.inf)  
import pandas as pd

usdjpy=pd.read_csv('C:\\data3\\usdjpy.csv',sep=',',encoding='utf-8')
usdjpy.rename(columns={'\ufeffDate':'Date'},inplace=True)
usdjpy=usdjpy.iloc[1:3913,:]
#print(usdjpy)
#print(twdow.columns.values)
#print(twdow)
txf=pd.read_excel('C:\\data3\\TXF.xlsx')
txf=txf.iloc[1:,:-1]
#print(txf)
tx=pd.merge(usdjpy,txf,on='Date',how='outer',indicator=True)

print(tx.columns)
print(tx.values)


# In[222]:

df2 = pd.DataFrame(np.ones((tx.shape[0],1))*0, columns=['tw_holiday'])
for i in range(0,tx.shape[0]):
    if tx['_merge'][i]=='left_only':       
        df2['tw_holiday'][i-1]=1

tx=pd.concat([tx, df2], axis=1)
del tx['_merge']
tx=tx.dropna(axis=0,how='any')
tx=tx.iloc[::-1,:]
print(tx)


# In[223]:

import numpy as np
np.set_printoptions(threshold=np.inf)  
import pandas as pd

twdow=pd.read_csv('C:\\data3\\twdow.csv',sep=',',encoding='utf-8')
twdow.rename(columns={'\ufeffDate':'Date'},inplace=True)
twdow=twdow.iloc[1:,:-2]
#print(twdow.columns.values)
#print(twdow)
#print(txf)
#print(txf.columns.values)
#print(type(txf['Date'][1]))
twse=pd.read_csv('C:\\data3\\twse.csv',sep=',',encoding='utf-8')
twse.rename(columns={'\ufeffDate':'Date'},inplace=True)
twse=twse.iloc[1:,:]

#print(txf)
tx=pd.merge(tx,twse,on='Date',how='inner')
tx=pd.merge(tx,twdow,on='Date',how='inner')
#print(tx.shape)
#print(tx)
#print(tx.columns)
#print(tx.values)


# In[224]:

tw50=pd.read_csv('C:\\data3\\tw50.csv',sep=',',encoding='utf-8')
tw50.rename(columns={'\ufeffDate':'Date'},inplace=True)
tw50=tw50.iloc[1:,:-2]
#print(tw50)
#print(txf)

tx=pd.merge(tx,tw50,on='Date',how='inner')
#print(tx)
#print(tx.shape)
#print(tx.columns)
#print(tx.values)


# In[225]:

tamsci=pd.read_csv('C:\\data3\\tamsci.csv',sep=',',encoding='utf-8')
tamsci.rename(columns={'\ufeffDate':'Date'},inplace=True)
tamsci=tamsci.iloc[1:,:]
#print(tamsci)
#print(txf)

tx=pd.merge(tx,tamsci,on='Date',how='inner')
#print(tx.shape)
#print(tx)
#print(tx.columns)
#print(tx.values)


# In[226]:

spx=pd.read_csv('C:\\data3\\spx.csv',sep=',',encoding='utf-8')
spx.rename(columns={'\ufeffDate':'Date'},inplace=True)
spx=spx.iloc[1:3777,:]
#print(spx)
print(spx['Date'][1])
print(tx['Date'][1])
#print(spx.columns.values[0])
#print(tx.columns.values[0])
#print(type(tx.columns.values[0]))
#print(type(szcomp.columns.values[0]))
#print(txf)

tx=pd.merge(tx,spx,on='Date',how='outer',indicator=True)
print(tx.columns)
print(tx.values)
#print(tx.ix[3,['o_spx', 'h_spx', 'l_spx', 'c_spx', 'v_spx']])



# In[227]:

for i in range(0,tx.shape[0]):
    if tx['_merge'][i]=='left_only':
        tx.ix[i,['o_spx', 'h_spx', 'l_spx', 'c_spx', 'v_spx','SMAVG (15)_spx']]=tx.ix[i-1,['o_spx', 'h_spx', 'l_spx', 'c_spx', 'v_spx','SMAVG (15)_spx']]

#print(tx.values)


# In[228]:

tx=tx.dropna(axis=0,how='any')
tx=tx.iloc[:,:-1]
#print(tx)


# In[229]:

szcomp=pd.read_csv('C:\\data3\\szcomp.csv',sep=',',encoding='utf-8')
szcomp.rename(columns={'\ufeffDate':'Date'},inplace=True)
szcomp=szcomp.iloc[1:,:]

#print(szcomp)
#print(txf)

tx=pd.merge(tx,szcomp,on='Date',how='outer',indicator=True)
#print(tx.columns)
#print(tx.values)




# In[230]:

df = pd.DataFrame(np.ones((tx.shape[0],1))*0, columns=['chinese_holiday'])
for i in range(0,tx.shape[0]):
    if tx['_merge'][i]=='left_only':
        tx.ix[i,['o_szcomp', 'h_szcomp', 'l_szcomp', 'c_szcomp', 'v_szcomp','SMAVG (15)_szcomp']]=tx.ix[i-1,['o_szcomp', 'h_szcomp', 'l_szcomp', 'c_szcomp', 'v_szcomp','SMAVG (15)_szcomp']]
        df['chinese_holiday'][i]=df['chinese_holiday'][i-1]+1

tx=pd.concat([tx, df], axis=1)
#print(tx.values)


# In[231]:

del tx['_merge']


# In[232]:

tx=tx.dropna(axis=0,how='any')
#print(tx)


# In[233]:

shcomp=pd.read_csv('C:\\data3\\shcomp.csv',sep=',',encoding='utf-8')
shcomp.rename(columns={'\ufeffDate':'Date'},inplace=True)
shcomp=shcomp.iloc[:,:]

#print(szcomp)
#print(txf)

tx=pd.merge(tx,shcomp,on='Date',how='outer',indicator=True)
#print(tx.columns)
#print(tx.values)
#print(tx.columns)


# In[234]:


for i in range(0,tx.shape[0]):
    if tx['_merge'][i]=='left_only':
        tx.ix[i,['o_shcomp', 'h_shcomp', 'l_shcomp', 'c_shcomp', 'v_shcomp','SMAVG (15)_shcomp']]=tx.ix[i-1,['o_shcomp', 'h_shcomp', 'l_shcomp', 'c_shcomp', 'v_shcomp','SMAVG (15)_shcomp']]

del tx['_merge']


# In[235]:

tx=tx.dropna(axis=0,how='any')
#print(tx)


# In[236]:

oil=pd.read_csv('C:\\data3\\oil sxep.csv',sep=',',encoding='utf-8')
oil.rename(columns={'\ufeffDate':'Date'},inplace=True)
oil=oil.iloc[1:,:]

#print(oil)
#print(txf)

tx=pd.merge(tx,oil,on='Date',how='outer',indicator=True)
#print(tx.columns)
#print(tx.values)


# In[237]:

for i in range(0,tx.shape[0]):
    if tx['_merge'][i]=='left_only':
        tx.ix[i,['o_oil','h_oil', 'l_oil', 'c_oil', 'v_oil', 'SMAVG (15)_oil']]=tx.ix[i-1,['o_oil','h_oil', 'l_oil', 'c_oil', 'v_oil', 'SMAVG (15)_oil']]

del tx['_merge']
tx=tx.dropna(axis=0,how='any')
#print(tx)


# In[238]:

nky=pd.read_csv('C:\\data3\\nky.csv',sep=',',encoding='utf-8')
nky.rename(columns={'\ufeffDate':'Date'},inplace=True)
nky=nky.iloc[1:,:]
#print(nky)

#print(txf)

tx=pd.merge(tx,nky,on='Date',how='outer',indicator=True)
#print(tx.columns)
#print(tx.values)


# In[239]:

df1 = pd.DataFrame(np.ones((tx.shape[0],1))*0, columns=['jp_holiday'])
for i in range(0,tx.shape[0]):
    if tx['_merge'][i]=='left_only':
        tx.ix[i,['o_nky', 'h_nky','l_nky', 'c_nky', 'v_nky', 'SMAVG (15)_nky']]=tx.ix[i-1,['o_nky', 'h_nky','l_nky', 'c_nky', 'v_nky', 'SMAVG (15)_nky']]
        df1['jp_holiday'][i]=df1['jp_holiday'][i-1]+1

tx=pd.concat([tx, df1], axis=1)
#print(tx.values)


# In[240]:

del tx['_merge']
tx=tx.dropna(axis=0,how='any')
#print(tx)


# In[241]:

nasdap=pd.read_csv('C:\\data3\\nasdap.csv',sep=',',encoding='utf-8')
nasdap.rename(columns={'\ufeffDate':'Date'},inplace=True)
nasdap=nasdap.iloc[1:,:]
#print(nasdap)

#print(txf)

tx=pd.merge(tx,nasdap,on='Date',how='outer',indicator=True)
#print(tx.columns)
#print(tx.values)


# In[242]:

df2 = pd.DataFrame(np.ones((tx.shape[0],1))*0, columns=['us_holiday'])
for i in range(0,tx.shape[0]):
    if tx['_merge'][i]=='left_only':
        tx.ix[i,['o_nasdap','h_nasdap', 'l_nasdap', 'c_nasdap', 'v_nasdap', 'SMAVG (15)_nasdap']]=tx.ix[i-1,['o_nasdap','h_nasdap', 'l_nasdap', 'c_nasdap', 'v_nasdap', 'SMAVG (15)_nasdap']]
        df2['us_holiday'][i]=df2['us_holiday'][i-1]+1

tx=pd.concat([tx, df2], axis=1)
del tx['_merge']
tx=tx.dropna(axis=0,how='any')
#print(tx)


# In[243]:

gold=pd.read_csv('C:\\data3\\gold usd.csv',sep=',',encoding='utf-8')
gold.rename(columns={'\ufeffDate':'Date'},inplace=True)
gold=gold.iloc[1:,:]
#print(gold)

#print(txf)

tx=pd.merge(tx,gold,on='Date',how='outer',indicator=True)
#print(tx.columns)
#print(tx.values)


# In[244]:

for i in range(0,tx.shape[0]):
    if tx['_merge'][i]=='left_only':
        tx.ix[i,['o_nasdap','h_nasdap', 'l_nasdap', 'c_nasdap', 'v_nasdap', 'SMAVG (15)_nasdap']]=tx.ix[i-1,['o_nasdap','h_nasdap', 'l_nasdap', 'c_nasdap', 'v_nasdap', 'SMAVG (15)_nasdap']]
del tx['_merge']
tx=tx.dropna(axis=0,how='any')
#print(tx)


# In[245]:

dax=pd.read_csv('C:\\data3\\dax.csv',sep=',',encoding='utf-8')
dax.rename(columns={'\ufeffDate':'Date'},inplace=True)
dax=dax.iloc[1:,:]
#print(dax.values)

#print(txf)

tx=pd.merge(tx,dax,on='Date',how='outer',indicator=True)
#print(tx.columns)
#print(tx.values)


# In[246]:

df3 = pd.DataFrame(np.ones((tx.shape[0],1))*0, columns=['german_holiday'])
for i in range(0,tx.shape[0]):
    if tx['_merge'][i]=='left_only':
        tx.ix[i,['o_dax', 'h_dax','l_dax', 'c_dax', 'v_dax', 'SMAVG (15)_dax']]=tx.ix[i-1,['o_dax', 'h_dax','l_dax', 'c_dax', 'v_dax', 'SMAVG (15)_dax']]
        df3['german_holiday'][i]=df3['german_holiday'][i-1]+1

tx=pd.concat([tx, df3], axis=1)
del tx['_merge']
tx=tx.dropna(axis=0,how='any')
#print(tx)


# In[247]:

cpi=pd.read_csv('C:\\data3\\cpi exchange based by jp morgan.csv',sep=',',encoding='utf-8')
cpi.rename(columns={'\ufeffDate':'Date'},inplace=True)
cpi=cpi.iloc[1:,:]
#print(cpi.values)
print(tx.columns)
print(cpi.columns)
#print(txf)

tx=pd.merge(tx,cpi,on='Date',how='outer',indicator=True)
print(tx.columns)
print(tx.values)


# In[248]:

print(tx.ix[8,'中價'])


# In[249]:

for i in range(0,tx.shape[0]):
    if tx['_merge'][i]=='left_only':
        tx.ix[i,'中價']=tx.ix[i-1,'中價']

del tx['_merge']
tx=tx.dropna(axis=0,how='any')
print(tx)


# In[250]:

cac=pd.read_csv('C:\\data3\\cac.csv',sep=',',encoding='utf-8')
cac.rename(columns={'\ufeffDate':'Date'},inplace=True)
cac=cac.iloc[1:,:]
#print(cpi.values)
#print(tx.columns)
#print(cac.values)
#print(txf)

tx=pd.merge(tx,cac,on='Date',how='outer',indicator=True)
print(tx.columns)
print(tx.values)


# In[251]:

df3 = pd.DataFrame(np.ones((tx.shape[0],1))*0, columns=['france_holiday'])
for i in range(0,tx.shape[0]):
    if tx['_merge'][i]=='left_only':
        tx.ix[i,['o_cac', 'h_cac', 'l_cac', 'c_cac', 'v_cac', 'SMAVG (15)_cac']]=tx.ix[i-1,['o_cac', 'h_cac', 'l_cac', 'c_cac', 'v_cac', 'SMAVG (15)_cac']]
        df3['france_holiday'][i]=df3['france_holiday'][i-1]+1

tx=pd.concat([tx, df3], axis=1)
del tx['_merge']
tx=tx.dropna(axis=0,how='any')
print(tx)


# In[252]:

ukx=pd.read_csv('C:\\data3\\ukx.csv',sep=',',encoding='utf-8')
ukx.rename(columns={'\ufeffDate':'Date'},inplace=True)
ukx=ukx.iloc[1:,:]
#print(ukx.values)
#print(twsenfne.values)
#print(cac.values)


tx=pd.merge(tx,ukx,on='Date',how='outer',indicator=True)
print(tx.columns)
print(tx.values)


# In[253]:

df3 = pd.DataFrame(np.ones((tx.shape[0],1))*0, columns=['uk_holiday'])
for i in range(0,tx.shape[0]):
    if tx['_merge'][i]=='left_only':
        tx.ix[i,['o_ukx', 'h_ukx', 'l_ukx', 'c_ukx', 'v_ukx','SMAVG (15)_ukx']]=tx.ix[i-1,['o_ukx', 'h_ukx', 'l_ukx', 'c_ukx', 'v_ukx','SMAVG (15)_ukx']]
        df3['uk_holiday'][i]=df3['uk_holiday'][i-1]+1

tx=pd.concat([tx, df3], axis=1)
del tx['_merge']
tx=tx.dropna(axis=0,how='any')
print(tx)


# In[ ]:




# In[ ]:




# In[254]:

usdtwd=pd.read_csv('C:\\data3\\usdtwd.csv',sep=',',encoding='utf-8')
usdtwd.rename(columns={'\ufeffDate':'Date'},inplace=True)
usdtwd=usdtwd.iloc[1:,:]
#print(usdtwd.values)
#print(twsenfne.values)
#print(cac.values)


tx=pd.merge(tx,usdtwd,on='Date',how='outer',indicator=True)
print(tx.columns)
print(tx.values)


# In[255]:

for i in range(0,tx.shape[0]):
    if tx['_merge'][i]=='left_only':
        tx.ix[i,['o_usdtwd', 'h_usdtwd', 'l_usdtwd', 'c_usdtwd']]=tx.ix[i-1,['o_usdtwd', 'h_usdtwd', 'l_usdtwd', 'c_usdtwd']]

del tx['_merge']
tx=tx.dropna(axis=0,how='any')
print(tx)


# In[256]:

vxx=pd.read_csv('C:\\data3\\vxx 500.csv',sep=',',encoding='utf-8')
vxx.rename(columns={'\ufeffDate':'Date'},inplace=True)
vxx=vxx.iloc[1:,:]
#print(vxx.values)
#print(twsenfne.values)
#print(cac.values)


#tx=pd.merge(tx,vxx,on='Date',how='outer',indicator=True)
#print(tx.columns)
#print(tx.values)


# In[257]:

print(tx)


# In[258]:

print(tx.columns)


# In[263]:

tx.to_csv('c:\\data3\\out.csv',index=False)


# In[264]:

tx.to_excel('c:\\data3\\out.xlsx',index=False)


# In[268]:

y=pd.read_excel('c:\\data3\\out.xlsx')


# In[270]:

#print(y)
print(y.columns)


# In[271]:

tw_holiday=y.pop('tw_holiday')
chinese_holiday=y.pop('chinese_holiday')
jp_holiday=y.pop('jp_holiday')
us_holiday=y.pop('us_holiday')
german_holiday=y.pop('german_holiday')
france_holiday=y.pop('france_holiday')
uk_holiday=y.pop('uk_holiday')
print(y)


# In[272]:

y = pd.concat([y, tw_holiday, chinese_holiday,jp_holiday,us_holiday,german_holiday,france_holiday,uk_holiday], axis=1)
print(y)


# In[274]:

y.to_excel('c:\\data3\\out1.xlsx',index=False)


# In[ ]:



