# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 16:35:19 2020

@author: ludov
"""

import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats as stat

file = 'E:/000_PAPER/CatWalk/CATWALK_PROFILES.xlsx'
savedir = 'E:/000_PAPER/CatWalk'

dataset = pd.read_excel(file,header=0,index_col=0,sheet_name='All')

fig, ax = plt.subplots(1,3,sharex=True,sharey=True,figsize=(18,9))
ax[0].set_title('CUFF individual trials') ; ax[0].set_ylabel('Log Balance Index')
ax[1].set_title('SHAM individual trials')
ax[1].set_title('WT individual trials')

days = dataset.columns[:-1]

allCuff, Sham1, Sham2, allCuffLabels, Sham1Labels, Sham2Labels = [],[],[],[],[],[]

allWT, WTlabels = [],[]

for animal in dataset.index[:24]:

    profile = dataset.loc[animal,'baseline':'post_op_33'].values
    
    condition = dataset.loc[animal,'Condition']
    
    if profile[0] > 0:
        profile = profile-profile[0]
        
    else:
        profile = profile + np.abs(profile[0])


    if condition == 'CUFF':
        ax[0].plot(profile,label=animal)
        allCuff.append(profile)
        allCuffLabels.append(animal)

#    else:
#        if 'S' in animal:
#            ax[1].plot(profile,label=animal)
#            Sham2.append(profile)
#            Sham2Labels.append(animal)
#        else:
#            ax[1].plot(profile,label=animal,linestyle='--')
#            Sham1.append(profile)
#            Sham1Labels.append(animal)
#            
#    if 'WT' in animal:
#        ax[2].plot(profile,label=animal,linestyle='-')
#        allWT.append(profile)
#        WTlabels.append(animal)      

    if condition == 'SHAM':   
        if 'S' in animal:
            ax[1].plot(profile,label=animal)
            Sham2.append(profile)
            Sham2Labels.append(animal)
        else:
            ax[1].plot(profile,label=animal,linestyle='--')
            Sham1.append(profile)
            Sham1Labels.append(animal)
            
    if condition == 'WT':
        ax[2].plot(profile,label=animal,linestyle='-')
        allWT.append(profile)
        WTlabels.append(animal) 
        

        

for i in range(3):
    ax[i].legend(loc='best')
    ax[i].plot(np.zeros(len(profile)),color='black',linestyle='--')
    ax[i].set_xticks(range(len(days)))
    ax[i].set_xticklabels(days,rotation='45')
    ax[i].set_xlabel('Day#')
      
#turn normed data to df
cuffs = pd.DataFrame(allCuff, index=allCuffLabels,columns=days)
sham1 = pd.DataFrame(Sham1, index=Sham1Labels,columns=days)
sham2 = pd.DataFrame(Sham2, index=Sham2Labels,columns=days)
wts = pd.DataFrame(allWT, index=WTlabels,columns=days)


#Plot average with SEM
fig2, axx = plt.subplots(1,2)

axx[0].set_title('All groups')
axx[0].plot(np.nanmean(cuffs.values,axis=0),color='orange',label='Cuff')
axx[0].plot(np.nanmean(sham1.values,axis=0),color='black',label='Sham1')
axx[0].plot(np.nanmean(sham2.values,axis=0),color='gray',label='Sham2')
axx[0].plot(np.nanmean(wts.values,axis=0),color='skyblue',label='WT')

cuffSEM = np.nanstd(cuffs.values,axis=0)/np.sqrt(cuffs.shape[0])

sham1SEM = np.nanstd(sham1.values,axis=0)/np.sqrt(sham1.shape[0])

sham2SEM = np.nanstd(sham2.values,axis=0)/np.sqrt(sham2.shape[0])

wtSEM = np.nanstd(wts.values,axis=0)/np.sqrt(wts.shape[0])

axx[0].fill_between(range(len(profile)),
                 np.nanmean(cuffs.values,axis=0)+cuffSEM,
                 np.nanmean(cuffs.values,axis=0)-cuffSEM,
                 color='orange',alpha=0.2)

axx[0].fill_between(range(len(profile)),
                 np.nanmean(sham1.values,axis=0)+sham1SEM,
                 np.nanmean(sham1.values,axis=0)-sham1SEM,
                 color='black',alpha=0.2)

axx[0].fill_between(range(len(profile)),
                 np.nanmean(sham2.values,axis=0)+sham2SEM,
                 np.nanmean(sham2.values,axis=0)-sham2SEM,
                 color='gray',alpha=0.2)

axx[0].fill_between(range(len(profile)),
                 np.nanmean(wts.values,axis=0)+wtSEM,
                 np.nanmean(wts.values,axis=0)-wtSEM,
                 color='skyblue',alpha=0.2)

#Plot except AI sham animal, seems to be an outlier
axx[1].set_title('All groups, except AI sham animal')
axx[1].plot(np.nanmean(cuffs.values,axis=0),color='orange',label='Cuff')
axx[1].plot(np.nanmean(sham1.loc[sham1.index != 'AI',:].values,axis=0),color='black',label='Sham1')
axx[1].plot(np.nanmean(sham2.values,axis=0),color='gray',label='Sham2')
axx[1].plot(np.nanmean(wts.values,axis=0),color='skyblue',label='WT')


cuffSEM = np.nanstd(cuffs.values,axis=0)/np.sqrt(cuffs.shape[0])

sham1SEM = np.nanstd(sham1.loc[sham1.index != 'AI',:].values,axis=0)/np.sqrt(sham1.loc[sham1.index != 'AI',:].shape[0])

sham2SEM = np.nanstd(sham2.values,axis=0)/np.sqrt(sham2.shape[0])

axx[1].fill_between(range(len(profile)),
                 np.nanmean(cuffs.values,axis=0)+cuffSEM,
                 np.nanmean(cuffs.values,axis=0)-cuffSEM,
                 color='orange',alpha=0.2)

axx[1].fill_between(range(len(profile)),
                 np.nanmean(sham1.loc[sham1.index != 'AI',:].values,axis=0)+sham1SEM,
                 np.nanmean(sham1.loc[sham1.index != 'AI',:].values,axis=0)-sham1SEM,
                 color='black',alpha=0.2)

axx[1].fill_between(range(len(profile)),
                 np.nanmean(sham2.values,axis=0)+sham2SEM,
                 np.nanmean(sham2.values,axis=0)-sham2SEM,
                 color='gray',alpha=0.2)

axx[1].fill_between(range(len(profile)),
                 np.nanmean(wts.values,axis=0)+wtSEM,
                 np.nanmean(wts.values,axis=0)-wtSEM,
                 color='skyblue',alpha=0.2)


for i in range(2):
    axx[i].legend(loc='best')
    axx[i].plot(np.zeros(len(profile)),color='black',linestyle='--')
    axx[i].set_xticks(range(len(days)))
    axx[i].set_xticklabels(days,rotation='45')
    axx[i].set_xlabel('Day#')
       
       
#Save datasheet for stats and co 
finalDf = pd.concat((cuffs,sham1,sham2,wts))

finalDf.to_excel('{}/NORMED_PROFILES.xlsx'.format(savedir))
