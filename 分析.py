# -*- coding: utf-8 -*-
"""
Created on Wed Dec 29 14:19:14 2021

@author: DELL
"""
import pandas as pd
import numpy as np
pd.set_option('display.max_rows',None)
pd.set_option('display.max_columns',None)
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
msno.matrix(data, labels=True)

## 6.1差异表达特征
# sheet3
a = pd.read_csv('E:\\sirebrowser\\OV\\两天\\7-卵巢癌干性相关基因筛选与鉴定\\5_预处理\\mirna+mrna_knn2.csv',engine='python',encoding='UTF-8-sig')
a = a.set_index('Sample',drop=True)
aa = a.iloc[3:,:]
#aa.to_csv('E:\\sirebrowser\\OV\\两天\\7-卵巢癌干性相关基因筛选与鉴定\\6_分析\\6_1-3.csv',index=0)

aa = aa.set_index('Sample',drop=True)
#nan = aa.isnull().sum(axis=0)
a = a.astype('float32')  
plt.figure(dpi=200)
sns.clustermap(data=a,method='average',metric='euclidean',row_cluster=True,
               col_cluster=False)
aa

## 6.2生存相关特征
# sheet1-cox
a = pd.read_csv('E:\\sirebrowser\\OV\\两天\\7-卵巢癌干性相关基因筛选与鉴定\\5_预处理\\mirna+mrna_z-score+clinical.csv',engine='python',encoding='UTF-8-sig')
a = a.set_index('Sample',drop=False)
data = a.values 
index1 = list(a.keys()) 
data = list(map(list, zip(*data))) 
data = pd.DataFrame(data, index=index1) 
data.iloc[0:3,0:4]
data.insert(0,'Sample',data.index)
data.columns = data.iloc[0,:]
data = data.drop('Sample',axis=0)
data = data.drop('stage',axis=1)
data = data.replace('dead',2)
data = data.replace('alive',1)

data.to_csv('E:\\sirebrowser\\OV\\两天\\7-卵巢癌干性相关基因筛选与鉴定\\6_分析\\6_2-1.csv', index=0)
data.iloc[0:3,17810:]

data.insert(17812,'status.1',data['status'])
data.insert(17813,'time.1',data['time'])
data = data.drop('status',axis=1)
data = data.drop('time',axis=1)
data.rename(columns={'status.1':'status','time.1':'time'},inplace=True)

data.to_csv('E:\\sirebrowser\\OV\\两天\\7-卵巢癌干性相关基因筛选与鉴定\\6_分析\\6_2-1(1).csv', index=0)

#from lifelines import CoxPHFitter 
#data['status'] = data['status'].replace(1,0)
#data['status'] = data['status'].replace(2,1)
#cox = data.set_index('Sample',drop=True)
#cph = CoxPHFitter()  #建立比例风险Cox模型
#cph.fit(cox, duration_col='time', event_col='status')  #模型拟合
#cph.print_summary()

# cox之后筛选 
targets = pd.read_csv('E:\\sirebrowser\\OV\\两天\\7-卵巢癌干性相关基因筛选与鉴定\\6_分析\\nolog\\diff_list.csv',engine='python')
p_value = pd.read_csv('E:\\sirebrowser\\OV\\两天\\7-卵巢癌干性相关基因筛选与鉴定\\6_分析\\nolog\\P_value.csv',engine='python')
hr = pd.read_csv('E:\\sirebrowser\\OV\\两天\\7-卵巢癌干性相关基因筛选与鉴定\\6_分析\\nolog\\HR.csv',engine='python')
ci = pd.read_csv('E:\\sirebrowser\\OV\\两天\\7-卵巢癌干性相关基因筛选与鉴定\\6_分析\\nolog\\CI.csv',engine='python')

# 在p_value中筛选targets
p_value.iloc[0:2,0:2]
p_value.info()
targets.iloc[1,0:2]
targets.info()
len(p_value.columns)
len(targets.columns)
lll = list(p_value.columns)
ttt = list(targets.columns)
for i in range(0,17809):
    if lll[i] in ttt:
        continue
    else:
        p_value.drop(columns=lll[i],inplace=True)

p_value.to_csv('E:\\sirebrowser\\OV\\两天\\7-卵巢癌干性相关基因筛选与鉴定\\6_分析\\nolog\\P_value_1.csv',index=0)

# 在hr中筛选targets
hr.iloc[0:2,0:2]
hr.info()
lll = list(hr.columns)
for i in range(0,17809): # 根据报错一直修改range的第二个值
    if lll[i] in ttt:
        continue
    else:
        hr = hr.drop(columns=lll[i],axis=1)

hr.to_csv('E:\\sirebrowser\\OV\\两天\\7-卵巢癌干性相关基因筛选与鉴定\\6_分析\\nolog\\HR_1.csv',index=0)

# 在ci中筛选targets
lll = list(ci.columns)
for i in range(0,17809): # 根据报错一直修改range的第二个值
    if lll[i] in ttt:
        continue
    else:
        ci = ci.drop(columns=lll[i],axis=1)

ci.to_csv('E:\\sirebrowser\\OV\\两天\\7-卵巢癌干性相关基因筛选与鉴定\\6_分析\\nolog\\CI_1.csv',index=0)

# sheet2-km
import pandas as pd
import numpy as np
from lifelines.datasets import load_waltons
from lifelines import KaplanMeierFitter
from lifelines.utils import median_survival_times
import matplotlib.pyplot as plt
from lifelines.statistics import logrank_test
fontt={'color': 'k',
      'size': 25,
      'family': 'Arial'}
fonty={'color': 'k',
      'size': 24,
      'family': 'Arial'}
font={'color': 'k',
      'size': 10,
      'family': 'Arial'}
## 平均值
a = pd.read_csv('E:\\sirebrowser\\OV\\两天\\7-卵巢癌干性相关基因筛选与鉴定\\6_分析\\km.csv',engine='python')
b = a.set_index('Sample')
b.iloc[142:144,17810:]

#i=2
q2 = {}
q3 = {}
q4 = {}
q2h = {}
q2l = {}
#i=17810
for i in range(2,17811): #17810
    print(b.columns[i],i)
    c = b.iloc[:,[0,1,i]] #i=2
    c = c.sort_values(by=b.columns[i],ascending=False)
    c_h = c[0:142]
    c_h['group'] = 'High'
    c_l = c[142:]
    c_l['group'] = 'Low'
    
    d = pd.concat([c_h,c_l],axis=0)
    e = d.iloc[:,[0,1,3]]
    kmf = KaplanMeierFitter()
    groups = e['group']
    ix1 = (groups == 'High')
    ix2 = (groups == 'Low')
    
    T = e['time']
    E = e['status']
    dem1 = (e['group'] == 'High')
    dem2 = (e['group'] == 'Low')
    results = logrank_test(T[dem1],T[dem2],E[dem1],E[dem2],alpha=.99)
   
#    q2[b.columns[i]] = '%.4f'%results.p_value
#    q3[b.columns[i]] = '%.4f'%c_h.iloc[141,2]
#    q4[b.columns[i]] = '%.4f'%c_l.iloc[0,2]

    if results.p_value <= 0.05:
        q2[b.columns[i]] = '%.4f'%results.p_value
        q3[b.columns[i]] = '%.4f'%c_h.iloc[141,2]
        q4[b.columns[i]] = '%.4f'%c_l.iloc[0,2]
        kmf.fit(e['time'][ix1], e['status'][ix1], label='High')
#        ax = kmf.plot(show_censors=True,ci_show=False,color='#3B49927F')
        q2h[b.columns[i]] = '%.4f'%kmf.median_survival_time_             

        kmf.fit(e['time'][ix2], e['status'][ix2], label='Low')
#        ax = kmf.plot(show_censors=True,ax=ax,ci_show=False,color='#BB00217F')
        q2l[b.columns[i]] = '%.4f'%kmf.median_survival_time_
        
#        plt.legend(prop={'family' : 'Arial', 'size'   : 24},handletextpad=0.5,frameon=False,labelspacing=0.1)
#        plt.tick_params(width=4)
#        ax.spines['bottom'].set_linewidth('2')
#        ax.spines['top'].set_linewidth('0')
#        ax.spines['left'].set_linewidth('2')
#        ax.spines['right'].set_linewidth('0')
#        plt.ylim(-0.08,1.08)
#        # plt.axvline(x=1072,c='k',ls='--',lw=0.5)
#        plt.title(b.columns[i], fontdict=fontt)
#        plt.text(0, -0.7,"P_value=%.4f"%results.p_value, fontdict=fonty)
#        plt.text(0, -0.9,"Q2u=%.4f"%c_h.iloc[141,2], fontdict=fonty)
#        plt.text(3000, -0.9,"Q2d=%.4f"%c_l.iloc[0,2], fontdict=fonty)
#    
#    
#        plt.xlabel('Timeline(days)', fontdict=fonty)
#        plt.ylabel('Cumulative survival (percentage)', fontdict=fonty)
#        plt.yticks(fontproperties = 'Arial', size = 24)
#        plt.xticks(fontproperties = 'Arial', size = 24,rotation=45)
#        plt.savefig(fname='E:\\sirebrowser\\OV\\两天\\7-卵巢癌干性相关基因筛选与鉴定\\6_分析\\2分位\\%s.png'%b.columns[i].replace('-','.').replace('|','.').replace('?','.'),figsize=[10,8],dpi=1000, bbox_inches='tight')
#        plt.close('all')
    else:
        continue

q22 = pd.DataFrame([q2]).T
q33 = pd.DataFrame([q3]).T
q44 = pd.DataFrame([q4]).T
q2hh = pd.DataFrame([q2h]).T
q2ll = pd.DataFrame([q2l]).T

#q2q = pd.concat([q22,q33,q44],axis=1)
q2hl = pd.concat([q22,q33,q44,q2hh,q2ll],axis=1)
q2hl.columns = ('p_value','up','down','high_median_survival_days','low_median_survival_days')
#q2hl.columns = ('high_median_survival_days','low_median_survival_days')
#q2q.to_csv('E:\\sirebrowser\\OV\\两天\\7-卵巢癌干性相关基因筛选与鉴定\\6_分析\\2分位\\2分位.csv')
#q2hl.to_csv('E:\\sirebrowser\\OV\\两天\\7-卵巢癌干性相关基因筛选与鉴定\\6_分析\\2分位\\2分位all.csv')
# 
# a = pd.read_csv('E:\\sirebrowser\\STAD\\miR\\分析\\rpm筛选按p排序.csv',engine='python')
# b = a.set_index('Sample')
# b.info()
# fig = plt.figure(figsize=(10,8))
q5 = {}
q6 = {}
q7 = {}
q4h = {}
q4l = {}

for i in range(2,17811):
    print(b.columns[i],i)
    c = b.iloc[:,[0,1,i]] #i=2
    c = c.sort_values(by=b.columns[i],ascending=False)
    c_h = c[0:71]
    c_h['group'] = 'High'
    c_l = c[213:]
    c_l['group'] = 'Low'
    
    d = pd.concat([c_h,c_l],axis=0)
    e = d.iloc[:,[0,1,3]]
    kmf = KaplanMeierFitter()
    groups = e['group']
    ix1 = (groups == 'High')
    ix2 = (groups == 'Low')
    
    T = e['time']
    E = e['status']
    dem1 = (e['group'] == 'High')
    dem2 = (e['group'] == 'Low')
    results = logrank_test(T[dem1],T[dem2],E[dem1],E[dem2],alpha=.99)
    
#    q5[b.columns[i]] = '%.4f'%results.p_value
#    q6[b.columns[i]] = '%.4f'%c_h.iloc[70,2]
#    q7[b.columns[i]] = '%.4f'%c_l.iloc[0,2]
    
    if results.p_value <= 0.05:
        q5[b.columns[i]] = '%.4f'%results.p_value
        q6[b.columns[i]] = '%.4f'%c_h.iloc[70,2]
        q7[b.columns[i]] = '%.4f'%c_l.iloc[0,2]
        kmf.fit(e['time'][ix1], e['status'][ix1], label='High')
        q4h[b.columns[i]] = '%.4f'%kmf.median_survival_time_ 
#        ax = kmf.plot(show_censors=True,ci_show=False,color='#3B49927F')
        kmf.fit(e['time'][ix2], e['status'][ix2], label='Low')
        q4l[b.columns[i]] = '%.4f'%kmf.median_survival_time_ 
#        ax = kmf.plot(show_censors=True,ax=ax,ci_show=False,color='#BB00217F')
        
#        plt.legend(prop={'family' : 'Arial', 'size'   : 24},handletextpad=0.5,frameon=False,labelspacing=0.1)
#        plt.tick_params(width=2)
#        ax.spines['bottom'].set_linewidth('2')
#        ax.spines['top'].set_linewidth('0')
#        ax.spines['left'].set_linewidth('2')
#        ax.spines['right'].set_linewidth('0')
#        plt.ylim(-0.08,1.08)
#        # plt.axvline(x=1072,c='k',ls='--',lw=0.5)
#        plt.title(b.columns[i], fontdict=fontt)
#        plt.text(0, -0.7,"P_value=%.4f"%results.p_value, fontdict=fonty)
#        plt.text(0, -0.9,"Q2u=%.4f"%c_h.iloc[70,2], fontdict=fonty)
#        plt.text(3000, -0.9,"Q2d=%.4f"%c_l.iloc[0,2], fontdict=fonty)
#        
#        plt.xlabel('Timeline(days)', fontdict=fonty)
#        plt.ylabel('Cumulative survival (percentage)', fontdict=fonty)
#        plt.yticks(fontproperties = 'Arial', size = 24)
#        plt.xticks(fontproperties = 'Arial', size = 24,rotation=45)
#        plt.savefig(fname='E:\\sirebrowser\\OV\\两天\\7-卵巢癌干性相关基因筛选与鉴定\\6_分析\\4分位\\%s.png'%b.columns[i].replace('-','.').replace('|','.').replace('?','.'),figsize=[10,8],dpi=1000, bbox_inches='tight')
#        plt.close('all')
    else:
        continue

q55 = pd.DataFrame([q5]).T
q66 = pd.DataFrame([q6]).T
q77 = pd.DataFrame([q7]).T
q4hh = pd.DataFrame([q4h]).T
q4ll = pd.DataFrame([q4l]).T

q4hl = pd.concat([q55,q66,q77,q4hh,q4ll],axis=1)
q4hl.columns = ('p_value','up','down','high_median_survival_days','low_median_survival_days')
#q4hl.to_csv('E:\\sirebrowser\\OV\\两天\\7-卵巢癌干性相关基因筛选与鉴定\\6_分析\\4分位\\4分位all.csv')

# 找出四分位和二分位的交集
lll = list(q4hl.index)
aaa = list(q2hl.index)
for i in range(0,1379): # 1020
    if lll[i] in aaa:
        continue
    else:
        q4hl.drop(index=lll[i],inplace=True)

lll = list(q4hl.index)
for i in range(0,1234): # 988 
    if aaa[i] in lll:
        continue
    else:
        q2hl.drop(index=aaa[i],inplace=True)
        
q24hl = pd.concat([q2hl,q4hl],axis=1)
q24hl.to_csv('E:\\sirebrowser\\OV\\两天\\7-卵巢癌干性相关基因筛选与鉴定\\6_分析\\nolog\\2+4分位p.csv')

# km与cox的交集
cox = pd.concat([p_value,hr,ci],axis=0,sort=False)
data1 = cox.values  
index1 = list(cox.keys())  
data1 = list(map(list, zip(*data1)))  
data1 = pd.DataFrame(data1, index=index1)  
data1.columns = ['p_value','hr','ci']
data1.insert(0,'feature',data1.index)
#data1['feature'] = data1['feature'].replace('-','.').replace('|','.').replace('?','.')
data1.to_csv('E:\\sirebrowser\\OV\\两天\\7-卵巢癌干性相关基因筛选与鉴定\\6_分析\\nolog\\p+hr_ci.csv')

cox = pd.read_csv('E:\\sirebrowser\\OV\\两天\\7-卵巢癌干性相关基因筛选与鉴定\\6_分析\\nolog\\p+hr_ci.csv',engine='python',encoding='UTF-8-sig')
cox = cox.set_index('feature',drop=False)

lll = list(cox['feature'])
aaa = list(q24hl.index)
for i in range(0,1668): # 1494 201
    if lll[i] in aaa:
        continue
    else:
        cox.drop(index=lll[i],inplace=True)

lll = list(cox['feature'])
len(aaa)
for i in range(0,509): # 345 201
    if aaa[i] in lll:
        continue
    else:
        q24hl.drop(index=aaa[i],inplace=True)
        
coxq = pd.concat([cox,q24hl],axis=1,sort=False)
coxq.to_csv('E:\\sirebrowser\\OV\\两天\\7-卵巢癌干性相关基因筛选与鉴定\\6_分析\\nolog\\cox+2+4分位p.csv',index=0)

## 6.3-6.11 相关基因
# pearson相关性
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
 
x = [0.5, 0.4, 0.6, 0.3, 0.6, 0.2, 0.7, 0.5]
y = [0.6, 0.4, 0.4, 0.3, 0.7, 0.2, 0.5, 0.6]
print(pearsonr(x, y))
print(spearmanr(x, y))
pearsonr(x, y)[0]
# 输出:(r, p)
# r:相关系数[-1，1]之间
# p:相关系数显著性
z = pd.read_csv('E:\\sirebrowser\\OV\\两天\\7-卵巢癌干性相关基因筛选与鉴定\\6_分析\\6_1-3.csv',engine='python')
z_score = z.values 
index1 = list(z.keys()) 
z_score = list(map(list, zip(*z_score))) 
z_score = pd.DataFrame(z_score, index=index1) 
z_score.iloc[0:2,0:2]
z_score.columns = z_score.iloc[0,:]
z_score = z_score.drop('Sample',axis=0)

r1 = z_score['PARP1|142_calculated']
r = pd.DataFrame(columns = ['name','pearson','pvalue'])
r['name'] = z_score.columns
for i in range(0,17809):
#    r['name'][i] = z_score.columns[i]
    r['pearson'][i] = pearsonr(r1,z_score.iloc[:,i])[0]
    r['pvalue'][i] = pearsonr(r1,z_score.iloc[:,i])[1]
r.to_csv('E:\\sirebrowser\\OV\\两天\\7-卵巢癌干性相关基因筛选与鉴定\\7_生存相关临床特征\\pearson\\CD44.csv',index=0)
r.to_csv('E:\\sirebrowser\\OV\\两天\\7-卵巢癌干性相关基因筛选与鉴定\\3-8生物信息学\PARP1.csv',index=0)

r1 = z_score['PROM1|8842_calculated']
r = pd.DataFrame(columns = ['name','pearson','pvalue'])
r['name'] = z_score.columns
for i in range(0,17809):
#    r['name'][i] = z_score.columns[i]
    r['pearson'][i] = pearsonr(r1,z_score.iloc[:,i])[0]
    r['pvalue'][i] = pearsonr(r1,z_score.iloc[:,i])[1]
r.to_csv('E:\\sirebrowser\\OV\\两天\\7-卵巢癌干性相关基因筛选与鉴定\\7_生存相关临床特征\\pearson\\PROM1.csv',index=0)

r1 = z_score['VIM|7431_calculated']
r = pd.DataFrame(columns = ['name','pearson','pvalue'])
r['name'] = z_score.columns
for i in range(0,17809):
#    r['name'][i] = z_score.columns[i]
    r['pearson'][i] = pearsonr(r1,z_score.iloc[:,i])[0]
    r['pvalue'][i] = pearsonr(r1,z_score.iloc[:,i])[1]
r.to_csv('E:\\sirebrowser\\OV\\两天\\7-卵巢癌干性相关基因筛选与鉴定\\7_生存相关临床特征\\pearson\\VIM(Vimentin).csv',index=0)

r1 = z_score['FN1|2335_calculated']
r = pd.DataFrame(columns = ['name','pearson','pvalue'])
r['name'] = z_score.columns
for i in range(0,17809):
#    r['name'][i] = z_score.columns[i]
    r['pearson'][i] = pearsonr(r1,z_score.iloc[:,i])[0]
    r['pvalue'][i] = pearsonr(r1,z_score.iloc[:,i])[1]
r.to_csv('E:\\sirebrowser\\OV\\两天\\7-卵巢癌干性相关基因筛选与鉴定\\7_生存相关临床特征\\pearson\\FN1(fibronectin 1).csv',index=0)

r1 = z_score['CDH2|1000_calculated']
r = pd.DataFrame(columns = ['name','pearson','pvalue'])
r['name'] = z_score.columns
for i in range(0,17809):
#    r['name'][i] = z_score.columns[i]
    r['pearson'][i] = pearsonr(r1,z_score.iloc[:,i])[0]
    r['pvalue'][i] = pearsonr(r1,z_score.iloc[:,i])[1]
r.to_csv('E:\\sirebrowser\\OV\\两天\\7-卵巢癌干性相关基因筛选与鉴定\\7_生存相关临床特征\\pearson\\CDH2(cadherin 2).csv',index=0)

r1 = z_score['CDH1|999_calculated']
r = pd.DataFrame(columns = ['name','pearson','pvalue'])
r['name'] = z_score.columns
for i in range(0,17809):
#    r['name'][i] = z_score.columns[i]
    r['pearson'][i] = pearsonr(r1,z_score.iloc[:,i])[0]
    r['pvalue'][i] = pearsonr(r1,z_score.iloc[:,i])[1]
r.to_csv('E:\\sirebrowser\\OV\\两天\\7-卵巢癌干性相关基因筛选与鉴定\\7_生存相关临床特征\\pearson\\CDH1(cadherin 1).csv',index=0)

r1 = z_score['MUC1|4582_calculated']
r = pd.DataFrame(columns = ['name','pearson','pvalue'])
r['name'] = z_score.columns
for i in range(0,17809):
#    r['name'][i] = z_score.columns[i]
    r['pearson'][i] = pearsonr(r1,z_score.iloc[:,i])[0]
    r['pvalue'][i] = pearsonr(r1,z_score.iloc[:,i])[1]
r.to_csv('E:\\sirebrowser\\OV\\两天\\7-卵巢癌干性相关基因筛选与鉴定\\7_生存相关临床特征\\pearson\\MUC1.csv',index=0)

r1 = z_score['LGR5|8549_calculated']
r = pd.DataFrame(columns = ['name','pearson','pvalue'])
r['name'] = z_score.columns
for i in range(0,17809):
#    r['name'][i] = z_score.columns[i]
    r['pearson'][i] = pearsonr(r1,z_score.iloc[:,i])[0]
    r['pvalue'][i] = pearsonr(r1,z_score.iloc[:,i])[1]
r.to_csv('E:\\sirebrowser\\OV\\两天\\7-卵巢癌干性相关基因筛选与鉴定\\7_生存相关临床特征\\pearson\\LGR5.csv',index=0)

r1 = z_score['SNAI1|6615_calculated']
r = pd.DataFrame(columns = ['name','pearson','pvalue'])
r['name'] = z_score.columns
for i in range(0,17809):
#    r['name'][i] = z_score.columns[i]
    r['pearson'][i] = pearsonr(r1,z_score.iloc[:,i])[0]
    r['pvalue'][i] = pearsonr(r1,z_score.iloc[:,i])[1]
r.to_csv('E:\\sirebrowser\\OV\\两天\\7-卵巢癌干性相关基因筛选与鉴定\\7_生存相关临床特征\\pearson\\SNAI1.csv',index=0)

# spearman相关性
from scipy.stats import spearmanr

r1 = z_score['CD44|960_calculated']
r = pd.DataFrame(columns = ['name','spearman','pvalue'])
r['name'] = z_score.columns
for i in range(0,17809):
#    r['name'][i] = z_score.columns[i]
    r['spearman'][i] = spearmanr(r1,z_score.iloc[:,i])[0]
    r['pvalue'][i] = spearmanr(r1,z_score.iloc[:,i])[1]
r.to_csv('E:\\sirebrowser\\OV\\两天\\7-卵巢癌干性相关基因筛选与鉴定\\7_生存相关临床特征\\spearman\\CD44.csv',index=0)

r1 = z_score['PROM1|8842_calculated']
r = pd.DataFrame(columns = ['name','spearman','pvalue'])
r['name'] = z_score.columns
for i in range(0,17809):
#    r['name'][i] = z_score.columns[i]
    r['spearman'][i] = spearmanr(r1,z_score.iloc[:,i])[0]
    r['pvalue'][i] = spearmanr(r1,z_score.iloc[:,i])[1]
r.to_csv('E:\\sirebrowser\\OV\\两天\\7-卵巢癌干性相关基因筛选与鉴定\\7_生存相关临床特征\\spearman\\PROM1.csv',index=0)

r1 = z_score['VIM|7431_calculated']
r = pd.DataFrame(columns = ['name','spearman','pvalue'])
r['name'] = z_score.columns
for i in range(0,17809):
#    r['name'][i] = z_score.columns[i]
    r['spearman'][i] = spearmanr(r1,z_score.iloc[:,i])[0]
    r['pvalue'][i] = spearmanr(r1,z_score.iloc[:,i])[1]
r.to_csv('E:\\sirebrowser\\OV\\两天\\7-卵巢癌干性相关基因筛选与鉴定\\7_生存相关临床特征\\spearman\\VIM(Vimentin).csv',index=0)

r1 = z_score['FN1|2335_calculated']
r = pd.DataFrame(columns = ['name','spearman','pvalue'])
r['name'] = z_score.columns
for i in range(0,17809):
#    r['name'][i] = z_score.columns[i]
    r['spearman'][i] = spearmanr(r1,z_score.iloc[:,i])[0]
    r['pvalue'][i] = spearmanr(r1,z_score.iloc[:,i])[1]
r.to_csv('E:\\sirebrowser\\OV\\两天\\7-卵巢癌干性相关基因筛选与鉴定\\7_生存相关临床特征\\spearman\\FN1(fibronectin 1).csv',index=0)

r1 = z_score['CDH2|1000_calculated']
r = pd.DataFrame(columns = ['name','spearman','pvalue'])
r['name'] = z_score.columns
for i in range(0,17809):
#    r['name'][i] = z_score.columns[i]
    r['spearman'][i] = spearmanr(r1,z_score.iloc[:,i])[0]
    r['pvalue'][i] = spearmanr(r1,z_score.iloc[:,i])[1]
r.to_csv('E:\\sirebrowser\\OV\\两天\\7-卵巢癌干性相关基因筛选与鉴定\\7_生存相关临床特征\\spearman\\CDH2(cadherin 2).csv',index=0)

r1 = z_score['CDH1|999_calculated']
r = pd.DataFrame(columns = ['name','spearman','pvalue'])
r['name'] = z_score.columns
for i in range(0,17809):
#    r['name'][i] = z_score.columns[i]
    r['spearman'][i] = spearmanr(r1,z_score.iloc[:,i])[0]
    r['pvalue'][i] = spearmanr(r1,z_score.iloc[:,i])[1]
r.to_csv('E:\\sirebrowser\\OV\\两天\\7-卵巢癌干性相关基因筛选与鉴定\\7_生存相关临床特征\\spearman\\CDH1(cadherin 1).csv',index=0)

r1 = z_score['MUC1|4582_calculated']
r = pd.DataFrame(columns = ['name','spearman','pvalue'])
r['name'] = z_score.columns
for i in range(0,17809):
#    r['name'][i] = z_score.columns[i]
    r['spearman'][i] = spearmanr(r1,z_score.iloc[:,i])[0]
    r['pvalue'][i] = spearmanr(r1,z_score.iloc[:,i])[1]
r.to_csv('E:\\sirebrowser\\OV\\两天\\7-卵巢癌干性相关基因筛选与鉴定\\7_生存相关临床特征\\spearman\\MUC1.csv',index=0)

r1 = z_score['LGR5|8549_calculated']
r = pd.DataFrame(columns = ['name','spearman','pvalue'])
r['name'] = z_score.columns
for i in range(0,17809):
#    r['name'][i] = z_score.columns[i]
    r['spearman'][i] = spearmanr(r1,z_score.iloc[:,i])[0]
    r['pvalue'][i] = spearmanr(r1,z_score.iloc[:,i])[1]
r.to_csv('E:\\sirebrowser\\OV\\两天\\7-卵巢癌干性相关基因筛选与鉴定\\7_生存相关临床特征\\spearman\\LGR5.csv',index=0)

r1 = z_score['SNAI1|6615_calculated']
r = pd.DataFrame(columns = ['name','spearman','pvalue'])
r['name'] = z_score.columns
for i in range(0,17809):
#    r['name'][i] = z_score.columns[i]
    r['spearman'][i] = spearmanr(r1,z_score.iloc[:,i])[0]
    r['pvalue'][i] = spearmanr(r1,z_score.iloc[:,i])[1]
r.to_csv('E:\\sirebrowser\\OV\\两天\\7-卵巢癌干性相关基因筛选与鉴定\\7_生存相关临床特征\\spearman\\SNAI1.csv',index=0)

# 提取正相关和负相关
#i = 'SNAI1'
pearson = ['CD44','PROM1','VIM(Vimentin)','FN1(fibronectin 1)','CDH2(cadherin 2)','CDH1(cadherin 1)','MUC1','LGR5','SNAI1']
for i in pearson:
    a = pd.read_csv(r'E:\\sirebrowser\\OV\\两天\\7-卵巢癌干性相关基因筛选与鉴定\\7_生存相关临床特征\\pearson\\%s.csv'%i,engine='python',encoding='UTF-8-sig')
    p = a[a['pvalue']<=0.05]
    p = p[p['pearson']>=0]
    p.to_csv(r'E:\\sirebrowser\\OV\\两天\\7-卵巢癌干性相关基因筛选与鉴定\\7_生存相关临床特征\\pearson\\%s_p.csv'%i,index=0)
    
    n = a[a['pvalue']<=0.05]
    n = n[n['pearson']<0]
    n.to_csv(r'E:\\sirebrowser\\OV\\两天\\7-卵巢癌干性相关基因筛选与鉴定\\7_生存相关临床特征\\pearson\\%s_n.csv'%i,index=0)

spearman = ['CD44','PROM1','VIM(Vimentin)','FN1(fibronectin 1)','CDH2(cadherin 2)','CDH1(cadherin 1)','MUC1','LGR5','SNAI1']
for i in spearman:
    a = pd.read_csv(r'E:\\sirebrowser\\OV\\两天\\7-卵巢癌干性相关基因筛选与鉴定\\7_生存相关临床特征\\spearman\\%s.csv'%i,engine='python',encoding='UTF-8-sig')
    p = a[a['pvalue']<=0.05]
    p = p[p['spearman']>=0]
    p.to_csv(r'E:\\sirebrowser\\OV\\两天\\7-卵巢癌干性相关基因筛选与鉴定\\7_生存相关临床特征\\spearman\\%s_p.csv'%i,index=0)
    
    n = a[a['pvalue']<=0.05]
    n = n[n['spearman']<0]
    n.to_csv(r'E:\\sirebrowser\\OV\\两天\\7-卵巢癌干性相关基因筛选与鉴定\\7_生存相关临床特征\\spearman\\%s_n.csv'%i,index=0)

# pearson和spearson互相检验
pp = ['CD44','PROM1','VIM(Vimentin)','FN1(fibronectin 1)','CDH2(cadherin 2)','CDH1(cadherin 1)','MUC1','LGR5','SNAI1']
for i in pp:
    a = pd.read_csv(r'E:\\sirebrowser\\OV\\两天\\7-卵巢癌干性相关基因筛选与鉴定\\7_生存相关临床特征\\pearson\\%s_n.csv'%i,engine='python',encoding='UTF-8-sig')
    b = pd.read_csv(r'E:\\sirebrowser\\OV\\两天\\7-卵巢癌干性相关基因筛选与鉴定\\7_生存相关临床特征\\spearman\\%s_n.csv'%i,engine='python',encoding='UTF-8-sig')
    a = a.set_index('name',drop=False)
    b = b.set_index('name',drop=False)
    aa = list(a['name'])
    bb = list(b['name'])
    for j in range(0,len(aa)):
        if aa[j] in bb:
            continue
        else:
            a.drop(index=aa[j],inplace=True)
        
    aa = list(a['name'])        
    for k in range(0,len(bb)):
        if bb[k] in aa:
            continue
        else:
            b.drop(index=bb[k],inplace=True) 
    c = pd.concat([a,b],axis=1,sort=False) 
    c.to_csv(r'E:\\sirebrowser\\OV\\两天\\7-卵巢癌干性相关基因筛选与鉴定\\7_生存相关临床特征\\pearson+spearman\\%s_n.csv'%i,index=0)
    
       
## 卡方检验
#a = pd.read_csv('E:\\sirebrowser\\OV\\两天\\7-卵巢癌干性相关基因筛选与鉴定\\6_分析\\6_1-3.csv',engine='python',encoding='UTF-8-sig')
#data = a.values 
#index1 = list(a.keys()) 
#data = list(map(list, zip(*data))) 
#data = pd.DataFrame(data, index=index1) 
#data.columns = data.iloc[0,:]
#data = data.drop('Sample',axis=0)
#data.iloc[0:2,0:2]   
#
#r1 = data['CD44|960_calculated']
#r2 = data['hsa-miR-148a-3p|MIMAT0000243']
#r2s = r2.sort_values(ascending=False)
#
#r1 = r1.sort_values(ascending=False)
#r3 = pd.concat([r1,r2],axis=1,sort=False)
#    
#n1 = r3[r3['CD44|960_calculated']>=r3.iloc[141,0]]
#n2 = r3[r3['CD44|960_calculated']<r3.iloc[141,0]]
#
#n11 = n1[n1['hsa-miR-148a-3p|MIMAT0000243']>=r2s.iloc[141]]
#n12 = n1[n1['hsa-miR-148a-3p|MIMAT0000243']<r2s.iloc[141]]
#
#n21 = n2[n2['hsa-miR-148a-3p|MIMAT0000243']>=r2s.iloc[141]]
#n22 = n2[n2['hsa-miR-148a-3p|MIMAT0000243']<r2s.iloc[141]]

#################### 代码优化 ##############################
lll = list(p_value.columns)
ttt = list(targets.columns)
for i in range(0,17808):
    if lll[i] in ttt:
        continue
    else:
        p_value.drop(columns=lll[i],inplace=True)

#################### 结果输出 ##############################
## 生存相关
a = pd.read_csv('E:\\sirebrowser\\OV\\两天\\7-卵巢癌干性相关基因筛选与鉴定\\6_分析\\6_1-3.csv',engine='python',encoding='UTF-8-sig')
a = a.set_index('Sample',drop=False)
#b = pd.read_csv('E:\\sirebrowser\\OV\\两天\\7-卵巢癌干性相关基因筛选与鉴定\\6_分析\\p+hr_ci.csv',engine='python',encoding='UTF-8-sig')
c = pd.read_csv('E:\\sirebrowser\\OV\\两天\\7-卵巢癌干性相关基因筛选与鉴定\\6_分析\\2分位\\2分位all.csv',engine='python',encoding='UTF-8-sig')
c.rename(columns={'feature':'Sample'},inplace=True)
c = c.set_index('Sample',drop=True)
d = pd.read_csv('E:\\sirebrowser\\OV\\两天\\7-卵巢癌干性相关基因筛选与鉴定\\6_分析\\4分位\\4分位all.csv',engine='python',encoding='UTF-8-sig')
d.rename(columns={'feature':'Sample'},inplace=True)
d = d.set_index('Sample',drop=True)


p = pd.read_csv('E:\\sirebrowser\\OV\\两天\\7-卵巢癌干性相关基因筛选与鉴定\\6_分析\\P_value.csv',engine='python',encoding='UTF-8-sig')
hr = pd.read_csv('E:\\sirebrowser\\OV\\两天\\7-卵巢癌干性相关基因筛选与鉴定\\6_分析\\HR.csv',engine='python',encoding='UTF-8-sig')
ci = pd.read_csv('E:\\sirebrowser\\OV\\两天\\7-卵巢癌干性相关基因筛选与鉴定\\6_分析\\CI.csv',engine='python',encoding='UTF-8-sig')
r = pd.concat([p,hr,ci],axis=0)
b = r.values 
index1 = list(r.keys()) 
b = list(map(list, zip(*b))) 
b = pd.DataFrame(b, index=index1) 
b.columns = ['p_value','hr','95%ci']
b.insert(0,'Sample',b.index)
b = b.set_index('Sample',drop=True)
b.index = a.index
b.iloc[0:2,0:3]
b.to_csv('E:\\sirebrowser\\OV\\两天\\7-卵巢癌干性相关基因筛选与鉴定\\6_分析\\p+hr+ci_all.csv',index=0)


result = pd.concat([a,b,c,d],axis=1,sort=False)
result.to_csv('E:\\sirebrowser\\OV\\两天\\7-卵巢癌干性相关基因筛选与鉴定\\6_分析\\cox+2+4.csv',index=0)

## 标志物相关
a = pd.read_csv('E:\\sirebrowser\\OV\\两天\\7-卵巢癌干性相关基因筛选与鉴定\\6_分析\\6_1-3.csv',engine='python',encoding='UTF-8-sig')
a = a.set_index('Sample',drop=False)

pearson = ['CD44','PROM1','SNAI1','VIM(Vimentin)','FN1(fibronectin 1)','CDH2(cadherin 2)','LGR5','MUC1','CDH1(cadherin 1)']
for i in pearson:
    p = pd.read_csv(r'E:\\sirebrowser\\OV\\两天\\7-卵巢癌干性相关基因筛选与鉴定\\7_生存相关临床特征\\pearson\\%s.csv'%i,engine='python',encoding='UTF-8-sig')
    p.rename(columns={'name':'Sample'},inplace=True)
    p = p.set_index('Sample',drop=True) 
    a = pd.concat([a,p],axis=1,sort=False)

spearman = ['CD44','PROM1','SNAI1','VIM(Vimentin)','FN1(fibronectin 1)','CDH2(cadherin 2)','LGR5','MUC1','CDH1(cadherin 1)']
for i in spearman:
    p = pd.read_csv(r'E:\\sirebrowser\\OV\\两天\\7-卵巢癌干性相关基因筛选与鉴定\\7_生存相关临床特征\\spearman\\%s.csv'%i,engine='python',encoding='UTF-8-sig')
    p.rename(columns={'name':'Sample'},inplace=True)
    p = p.set_index('Sample',drop=True) 
    a = pd.concat([a,p],axis=1,sort=False)    
    
a.to_csv('E:\\sirebrowser\\OV\\两天\\7-卵巢癌干性相关基因筛选与鉴定\\7_生存相关临床特征\\标志物相关.csv',index=0)

# 标志物相关统计
import itertools
import copy
list1 = ['CD44','PROM1','SNAI1','VIM(Vimentin)','FN1(fibronectin 1)','CDH2(cadherin 2)','LGR5']
list2 = []
#i=2
#j=1
k=5
for i in range(2,len(list1)+1):
    iter = itertools.combinations(list1,i)
    list2.append(list(iter))
    for k in range(0,6):
        for j in range(0,len(list2[k])):
            a = pd.read_csv(r'E:\\sirebrowser\\OV\\两天\\7-卵巢癌干性相关基因筛选与鉴定\\7_生存相关临床特征\\pearson\\%s_p.csv'%list2[k][j][0],engine='python',encoding='UTF-8-sig')
            a = a.set_index('name',drop=False)
            b = pd.read_csv(r'E:\\sirebrowser\\OV\\两天\\7-卵巢癌干性相关基因筛选与鉴定\\7_生存相关临床特征\\pearson\\%s_p.csv'%list2[k][j][1],engine='python',encoding='UTF-8-sig')
            b = b.set_index('name',drop=False)
            c = pd.read_csv(r'E:\\sirebrowser\\OV\\两天\\7-卵巢癌干性相关基因筛选与鉴定\\7_生存相关临床特征\\pearson\\%s_p.csv'%list2[k][j][2],engine='python',encoding='UTF-8-sig')
            c = c.set_index('name',drop=False)
            d = pd.read_csv(r'E:\\sirebrowser\\OV\\两天\\7-卵巢癌干性相关基因筛选与鉴定\\7_生存相关临床特征\\pearson\\%s_p.csv'%list2[k][j][3],engine='python',encoding='UTF-8-sig')
            d = d.set_index('name',drop=False)
            e = pd.read_csv(r'E:\\sirebrowser\\OV\\两天\\7-卵巢癌干性相关基因筛选与鉴定\\7_生存相关临床特征\\pearson\\%s_p.csv'%list2[k][j][4],engine='python',encoding='UTF-8-sig')
            e = e.set_index('name',drop=False)
            f = pd.read_csv(r'E:\\sirebrowser\\OV\\两天\\7-卵巢癌干性相关基因筛选与鉴定\\7_生存相关临床特征\\pearson\\%s_p.csv'%list2[k][j][5],engine='python',encoding='UTF-8-sig')
            f = f.set_index('name',drop=False)
            g = pd.read_csv(r'E:\\sirebrowser\\OV\\两天\\7-卵巢癌干性相关基因筛选与鉴定\\7_生存相关临床特征\\pearson\\%s_p.csv'%list2[k][j][6],engine='python',encoding='UTF-8-sig')
            g = g.set_index('name',drop=False)
            aa = list(a['name'])
            bb = list(b['name'])
            cc = list(c['name'])
            dd = list(d['name'])
            ee = list(e['name'])
            ff = list(f['name'])
            gg = list(g['name'])
            #
            for i in range(0,len(aa)):
                if aa[i] in bb:
                    continue
                else:
                    a.drop(index=aa[i],inplace=True)
            aa = list(a['name'])
            for i in range(0,len(bb)):
                if bb[i] in aa:
                    continue
                else:
                    b.drop(index=bb[i],inplace=True)
            #
            if len(a) != 0: 
                cc = list(c['name'])
                for i in range(0,len(aa)):
                    if aa[i] in cc:
                        continue
                    else:
                        a.drop(index=aa[i],inplace=True)
                bb = list(b['name'])
                for i in range(0,len(bb)):
                    if bb[i] in cc:
                        continue
                    else:
                        b.drop(index=bb[i],inplace=True)
                aa = list(a['name'])
                for i in range(0,len(cc)):
                    if cc[i] in aa:
                        continue
                    else:
                        c.drop(index=cc[i],inplace=True)   
                #
                if len(a) != 0: 
                    aa = list(a['name'])       
                    for i in range(0,len(aa)):
                        if aa[i] in dd:
                            continue
                        else:
                            a.drop(index=aa[i],inplace=True)
                    bb = list(b['name'])
                    for i in range(0,len(bb)):
                        if bb[i] in dd:
                            continue
                        else:
                            b.drop(index=bb[i],inplace=True)
                    cc = list(c['name'])
                    for i in range(0,len(cc)):
                        if cc[i] in dd:
                            continue
                        else:
                            c.drop(index=cc[i],inplace=True)        
                    aa = list(a['name'])
                    for i in range(0,len(dd)):
                        if dd[i] in aa:
                            continue
                        else:
                            d.drop(index=dd[i],inplace=True)
                    #
                    if len(a) != 0: 
                        aa = list(a['name'])       
                        for i in range(0,len(aa)):
                            if aa[i] in ee:
                                continue
                            else:
                                a.drop(index=aa[i],inplace=True)
                        bb = list(b['name'])
                        for i in range(0,len(bb)):
                            if bb[i] in ee:
                                continue
                            else:
                                b.drop(index=bb[i],inplace=True)
                        cc = list(c['name'])
                        for i in range(0,len(cc)):
                            if cc[i] in ee:
                                continue
                            else:
                                c.drop(index=cc[i],inplace=True)   
                        dd = list(d['name'])
                        for i in range(0,len(dd)):
                            if dd[i] in ee:
                                continue
                            else:
                                d.drop(index=dd[i],inplace=True) 
                        aa = list(a['name'])
                        for i in range(0,len(ee)):
                            if ee[i] in aa:
                                continue
                            else:
                                e.drop(index=ee[i],inplace=True)
                        #
                        if len(a) != 0: 
                            aa = list(a['name'])       
                            for i in range(0,len(aa)):
                                if aa[i] in ff:
                                    continue
                                else:
                                    a.drop(index=aa[i],inplace=True)
                            bb = list(b['name'])
                            for i in range(0,len(bb)):
                                if bb[i] in ff:
                                    continue
                                else:
                                    b.drop(index=bb[i],inplace=True)
                            cc = list(c['name'])
                            for i in range(0,len(cc)):
                                if cc[i] in ff:
                                    continue
                                else:
                                    c.drop(index=cc[i],inplace=True)   
                            dd = list(d['name'])
                            for i in range(0,len(dd)):
                                if dd[i] in ff:
                                    continue
                                else:
                                    d.drop(index=dd[i],inplace=True) 
                            ee = list(e['name'])
                            for i in range(0,len(ee)):
                                if ee[i] in ff:
                                    continue
                                else:
                                    e.drop(index=ee[i],inplace=True) 
                            aa = list(a['name'])
                            for i in range(0,len(ff)):
                                if ff[i] in aa:
                                    continue
                                else:
                                    f.drop(index=ff[i],inplace=True)
                            #
                            if len(a) != 0: 
                                aa = list(a['name'])       
                                for i in range(0,len(aa)):
                                    if aa[i] in gg:
                                        continue
                                    else:
                                        a.drop(index=aa[i],inplace=True)
                                bb = list(b['name'])
                                for i in range(0,len(bb)):
                                    if bb[i] in gg:
                                        continue
                                    else:
                                        b.drop(index=bb[i],inplace=True)
                                cc = list(c['name'])
                                for i in range(0,len(cc)):
                                    if cc[i] in gg:
                                        continue
                                    else:
                                        c.drop(index=cc[i],inplace=True)   
                                dd = list(d['name'])
                                for i in range(0,len(dd)):
                                    if dd[i] in gg:
                                        continue
                                    else:
                                        d.drop(index=dd[i],inplace=True) 
                                ee = list(e['name'])
                                for i in range(0,len(ee)):
                                    if ee[i] in gg:
                                        continue
                                    else:
                                        e.drop(index=ee[i],inplace=True) 
                                ff = list(f['name'])
                                for i in range(0,len(ff)):
                                    if ff[i] in gg:
                                        continue
                                    else:
                                        f.drop(index=ff[i],inplace=True)
                                aa = list(a['name'])
                                for i in range(0,len(gg)):
                                    if gg[i] in aa:
                                        continue
                                    else:
                                        g.drop(index=gg[i],inplace=True)
                                #        
                                a.rename(columns={'name':r'%s'%list2[k][j][0]},inplace=True)
                                b.rename(columns={'name':r'%s'%list2[k][j][1]},inplace=True)
                                c.rename(columns={'name':r'%s'%list2[k][j][2]},inplace=True)
                                d.rename(columns={'name':r'%s'%list2[k][j][3]},inplace=True)
                                e.rename(columns={'name':r'%s'%list2[k][j][4]},inplace=True)
                                f.rename(columns={'name':r'%s'%list2[k][j][5]},inplace=True)
                                g.rename(columns={'name':r'%s'%list2[k][j][6]},inplace=True)
                                if len(a) != 0: 
                                    re = pd.concat([a,b,c,d,e,f,g],axis=1,sort=False) 
                                    re.to_csv(r'E:\\sirebrowser\\OV\\两天\\7-卵巢癌干性相关基因筛选与鉴定\\7_生存相关临床特征\\交集7\\%s_p+%s_p+%s_p+%s_p+%s_p+%s_p+%s_p.csv'%(list2[k][j][0],list2[k][j][1],list2[k][j][2],list2[k][j][3],list2[k][j][4],list2[k][j][5],list2[k][j][6]),index=0)
            
                                    # 判断负相关
                                    j1 = copy.deepcopy(re)
                                    j2 = copy.deepcopy(re)
                                    
                                    m = pd.read_csv('E:\\sirebrowser\\OV\\两天\\7-卵巢癌干性相关基因筛选与鉴定\\7_生存相关临床特征\\pearson\\MUC1_n.csv',engine='python',encoding='UTF-8-sig')
                                    m = m.set_index('name',drop=False)
                                    cd = pd.read_csv('E:\\sirebrowser\\OV\\两天\\7-卵巢癌干性相关基因筛选与鉴定\\7_生存相关临床特征\\pearson\\CDH1(cadherin 1)_n.csv',engine='python',encoding='UTF-8-sig')
                                    cd = cd.set_index('name',drop=False)   
                                    
                                    aa = list(j1.index)
                                    bb = list(m.index)
                                    for i in range(0,len(aa)):
                                        if aa[i] in bb:
                                            continue
                                        else:
                                            j1.drop(index=aa[i],inplace=True)
                                    aa = list(j1.index)
                                    for i in range(0,len(bb)):
                                        if bb[i] in aa:
                                            continue
                                        else:
                                            m.drop(index=bb[i],inplace=True)
                                    if len(m) != 0: 
                                        re1 = pd.concat([j1,m],axis=1,sort=False) 
                                        re1.to_csv(r'E:\\sirebrowser\\OV\\两天\\7-卵巢癌干性相关基因筛选与鉴定\\7_生存相关临床特征\\交集7\\%s_p+%s_p+%s_p+%s_p+%s_p+%s_p+%s_p+MUC1_n.csv'%(list2[k][j][0],list2[k][j][1],list2[k][j][2],list2[k][j][3],list2[k][j][4],list2[k][j][5],list2[k][j][6]),index=0)
                                    else:
                                        continue
                                    
                                    aa = list(j2.index)
                                    bb = list(cd.index)
                                    for i in range(0,len(aa)):
                                        if aa[i] in bb:
                                            continue
                                        else:
                                            j2.drop(index=aa[i],inplace=True)
                                    aa = list(j2.index)
                                    for i in range(0,len(bb)):
                                        if bb[i] in aa:
                                            continue
                                        else:
                                            cd.drop(index=bb[i],inplace=True) 
                                    if len(cd) != 0:         
                                        re2 = pd.concat([j2,cd],axis=1,sort=False) 
                                        re2.to_csv(r'E:\\sirebrowser\\OV\\两天\\7-卵巢癌干性相关基因筛选与鉴定\\7_生存相关临床特征\\交集7\\%s_p+%s_p+%s_p+%s_p+%s_p+%s_p+%s_p+CDH1(cadherin 1)_n.csv'%(list2[k][j][0],list2[k][j][1],list2[k][j][2],list2[k][j][3],list2[k][j][4],list2[k][j][5],list2[k][j][6]),index=0)
                                        
                                        m = pd.read_csv('E:\\sirebrowser\\OV\\两天\\7-卵巢癌干性相关基因筛选与鉴定\\7_生存相关临床特征\\pearson\\MUC1_n.csv',engine='python',encoding='UTF-8-sig')
                                        m = m.set_index('name',drop=False)
                                        aa = list(re2.index)
                                        bb = list(m.index)
                                        for i in range(0,len(aa)):
                                            if aa[i] in bb:
                                                continue
                                            else:
                                                re2.drop(index=aa[i],inplace=True)
                                        aa = list(re2.index)
                                        for i in range(0,len(bb)):
                                            if bb[i] in aa:
                                                continue
                                            else:
                                                m.drop(index=bb[i],inplace=True)
                                        if len(m) != 0: 
                                            re3 = pd.concat([re2,m],axis=1,sort=False) 
                                            re3.to_csv(r'E:\\sirebrowser\\OV\\两天\\7-卵巢癌干性相关基因筛选与鉴定\\7_生存相关临床特征\\交集7\\%s_p+%s_p+%s_p+%s_p+%s_p+%s_p+%s_p+MUC1_n+CDH1(cadherin 1)_n.csv'%(list2[k][j][0],list2[k][j][1],list2[k][j][2],list2[k][j][3],list2[k][j][4],list2[k][j][5],list2[k][j][6]),index=0)
                                        else:
                                            continue
                                    else:
                                        continue
                                else:
                                    continue
                            else:
                                continue
                        else:
                            continue
                    else:
                        continue
                else:
                    continue
            else:
                continue

print(list2)
list2[5][20][1]
len(list2[0])    
  
## 标志物相关与生存相关交集
import os
a = pd.read_csv('E:\\sirebrowser\\OV\\两天\\7-卵巢癌干性相关基因筛选与鉴定\\6_分析\\生存相关交集.csv',engine='python',encoding='UTF-8-sig')

#打印出一个目录下所有的末级文件名称
#i=2
for i in range(2,8):
    for root,dirs,files in os.walk(r"E:\\sirebrowser\\OV\\两天\\7-卵巢癌干性相关基因筛选与鉴定\\7_生存相关临床特征\\交集%s"%i):
        print(files)
        for j in range(0,len(files)):
#            files[0]
#            j=0
            a = pd.read_csv('E:\\sirebrowser\\OV\\两天\\7-卵巢癌干性相关基因筛选与鉴定\\6_分析\\生存相关交集log2.csv',engine='python',encoding='UTF-8-sig')
            a = a.set_index('feature',drop=False)
            b = pd.read_csv(r'E:\\sirebrowser\\OV\\两天\\7-卵巢癌干性相关基因筛选与鉴定\\7_生存相关临床特征\\交集%s\\%s'%(i,files[j]),engine='python',encoding='UTF-8-sig')
            b = b.set_index(b.iloc[:,0],drop=True)
            aa = list(a.index)
            bb = list(b.index)
            for k in range(0,len(aa)):
                if aa[k] in bb:
                    continue
                else:
                    a.drop(index=aa[k],inplace=True)
            aa = list(a.index)
            for g in range(0,len(bb)):
                if bb[g] in aa:
                    continue
                else:
                    b.drop(index=bb[g],inplace=True)
            if len(a) != 0: 
                re = pd.concat([a,b],axis=1,sort=False) 
                re.to_csv(r'E:\\sirebrowser\\OV\\两天\\7-卵巢癌干性相关基因筛选与鉴定\\7_生存相关临床特征\\交集%s生存相关\\%s'%(i,files[j]),index=0)
            else:
                continue
    
for i in range(2,8):
    for root,dirs,files in os.walk(r"E:\\sirebrowser\\OV\\两天\\7-卵巢癌干性相关基因筛选与鉴定\\7_生存相关临床特征\\2交集%s"%i):
        for j in range(0,len(files)):
#            files[0]
#            j=0
            a = pd.read_csv('E:\\sirebrowser\\OV\\两天\\7-卵巢癌干性相关基因筛选与鉴定\\6_分析\\生存相关交集.csv',engine='python',encoding='UTF-8-sig')
            a = a.set_index('feature',drop=False)
            b = pd.read_csv(r'E:\\sirebrowser\\OV\\两天\\7-卵巢癌干性相关基因筛选与鉴定\\7_生存相关临床特征\\2交集%s\\%s'%(i,files[j]),engine='python',encoding='UTF-8-sig')
            b = b.set_index(b.iloc[:,0],drop=True)
            aa = list(a.index)
            bb = list(b.index)
            for k in range(0,len(aa)):
                if aa[k] in bb:
                    continue
                else:
                    a.drop(index=aa[k],inplace=True)
            aa = list(a.index)
            for g in range(0,len(bb)):
                if bb[g] in aa:
                    continue
                else:
                    b.drop(index=bb[g],inplace=True)
            if len(a) != 0: 
                re = pd.concat([a,b],axis=1,sort=False) 
                re.to_csv(r'E:\\sirebrowser\\OV\\两天\\7-卵巢癌干性相关基因筛选与鉴定\\7_生存相关临床特征\\2交集%s生存相关\\%s'%(i,files[j]),index=0)
            else:
                continue    
    
    