
# coding: utf-8

# In[36]:

import requests
from bs4 import BeautifulSoup
res=requests.get("http://news.ltn.com.tw/list/newspaper/business/20170926")
soup=BeautifulSoup(res.text)


#print(list)
#print(list.select('a'))
for lis in soup.select('.list'):
    for item in lis.find_all('a',attrs='tit'):
        #if item['class']=='ph':
            print(item.text)
            b=item.get('href')
            tt=requests.get("http://news.ltn.com.tw/"+b)
            sp=BeautifulSoup(tt.text)
            for ss in sp.select('.text'):
                if ss.select('h4')!=[]:
                    print(ss.select('h4'))
                for p in ss.select('p'):
                    print(p.text)


# In[31]:




# In[25]:




# In[ ]:



