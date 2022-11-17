
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 11 12:53:38 2022

@author: SDonnel

Notational clarification
    status: 0=operational but non-responsive; 1=all clear; 2=destroyed
    NOTE: assets do not turn to 0, this is reserved for a future study
    
    Terminology has changed, however variable names have not:
        center of gravity-> centroid
        clustering-> grouping
        cluster-> tile
        
        
"""


#%% Packages
import numpy as np
import warnings
import pandas as pd
import matplotlib.pyplot as plt
import math
import random
import time



#%%Definitions -- made too late but better than never at all

def distance(x1,y1,x2,y2):
    distcalc=np.sqrt((x1-x2)**2+(y1-y2)**2)
    return distcalc


def addlabels(x,y):
    for i in range(len(x)):
        plt.text(i, y[i], y[i], ha = 'center')


def truncate(number, digits) -> float:
    stepper = 10.0 ** digits
    return math.trunc(stepper * number) / stepper




#%%Dataset Creation
#set seed
random.seed(386)

#parameter to determine the number of assets versus targets desired
a_t_prop=0.5
#asset ratios
s_prop=0.40
st_prop=0.40
c2_prop=0.20

#set number of players
total_players=25
#set size of map
fieldlength=150


#Design selection: 
#LtoR = 1   assets inner/targets outer =2  targets inner/assets outer =3   Chaos =4
design=1

#extra parameters for designs 2/3
centerpoint=[0.5*fieldlength, 0.5*fieldlength]
arearadius=30





##make the dataframe:
#Types

assetscheck=0
while assetscheck==0:
    typelist=[]
    for i in range(total_players):
        randnum=random.random()
        if randnum<a_t_prop*s_prop:
            typelist.append('sensor')
        elif randnum<a_t_prop*s_prop+a_t_prop*st_prop:
            typelist.append('strike')
        elif randnum<a_t_prop*s_prop+a_t_prop*st_prop+a_t_prop*c2_prop:
            typelist.append('c2')        
        else:
            typelist.append('target')
            
        if typelist.count('strike')>0 and typelist.count('sensor')>0 and typelist.count('c2') >0 and typelist.count('target')>0:
            assetscheck=1

    
#status -- all begin active
statuslist=[1]*total_players    
#(x,y) of locations -- dependant on the design type
xlist=[]
ylist=[]
for i in range(total_players):
    if design == 1:
        #asstes left targets right
        if typelist[i] == 'target':
            xlist.append(random.randrange(0.70*fieldlength, fieldlength))
            ylist.append(random.randrange(0,fieldlength))
        else:
            xlist.append(random.randrange(0, 0.3*fieldlength))
            ylist.append(random.randrange(0,fieldlength))
    elif design == 2:
        if typelist[i] == 'target':
            #random point outside circle
            goodpoint=0
            while goodpoint==0:
                xtemp=random.randrange(0,fieldlength)
                ytemp=random.randrange(0,fieldlength)
                if (xtemp-centerpoint[0])**2+(ytemp-centerpoint[1])**2>=arearadius**2:
                    goodpoint=1
            xlist.append(xtemp)
            ylist.append(ytemp)
        else:
            #random point inside circle
            r=arearadius*np.sqrt(random.random())
            temptheta= random.random()*2*math.pi
            xlist.append(centerpoint[0]+r*math.cos(temptheta))
            ylist.append(centerpoint[1]+r*math.sin(temptheta))
    elif design == 3:
        if typelist[i] == 'target':
            #random point inside circle
            r=arearadius*np.sqrt(random.random())
            temptheta= random.random()*2*math.pi
            xlist.append(centerpoint[0]+r*math.cos(temptheta))
            ylist.append(centerpoint[1]+r*math.sin(temptheta))
        else:
            #random point outside circle+
            goodpoint=0
            while goodpoint==0:
                xtemp=random.randrange(0,fieldlength)
                ytemp=random.randrange(0,fieldlength)
                if (xtemp-centerpoint[0])**2+(ytemp-centerpoint[1])**2>=arearadius**2:
                    goodpoint=1
            xlist.append(xtemp)
            ylist.append(ytemp)                    
    elif design == 4:
        #chaos everywhere
        xlist.append(random.randrange(0,fieldlength))
        ylist.append(random.randrange(0,fieldlength))


#make the dataframe
collected_data= {'type': typelist, 
                 'x': xlist,
                 'y': ylist,
                 'status': statuslist}
df_raw = pd.DataFrame(collected_data)

#%% Data Import, Clean-up, and Display

df=df_raw.sort_values('type')
df=df.reset_index(drop='True')

#do a total count (active or not) for naming purposes
numc2 = df['type'].value_counts().c2
numstrike = df['type'].value_counts().strike
numsensor = df['type'].value_counts().sensor
numtarget = df['type'].value_counts().target

names=[]
clean_names=[] #solely for plotting

for i in range(0,numc2):
    names.append('c2_'+str(i+1))
    clean_names.append('c'+str(i+1))
for i in range(0,numsensor):
    names.append('sensor_'+str(i+1))
    clean_names.append('s'+str(i+1))
for i in range(0,numstrike):
    names.append('strike_'+str(i+1))  
    clean_names.append('st'+str(i+1))
for i in range(0,numtarget):
    names.append('target_'+str(i+1))    
    clean_names.append('t'+str(i+1))
    
df['names']=names


colors={'strike':'purple', 'sensor':'blue','c2':'green', 'target':'red'}

# =============================================================================
# #Plot of the 'longer' named actors
# annotations=df['names']
# 
# df.plot.scatter(x='x',
#                 y='y',
#                 c=df['type'].map(colors))
# 
# for i, label in enumerate(annotations):
#     plt.annotate(label, (df['x'][i]+1, df['y'][i]+1))
#     
# plt.xlim(0,fieldlength)
# plt.ylim(0,fieldlength)     
# 
# plt.show()
# 
# =============================================================================

#plot of cleaner names (s=sensor, st=strike, c=s2, t=target)
df.plot.scatter(x='x',
                y='y',
                c=df['type'].map(colors))

for i, label in enumerate(clean_names):
    plt.annotate(label, (df['x'][i]+1, df['y'][i]+1))
    
plt.xlim(0,fieldlength)
plt.ylim(0,fieldlength)     
plt.title('Iniital Position')

plt.show()

#establish variables for tracking
starting_position=[df.x,df.y] #just for plotting 
tracking_dict={}

for i in range(len(df)):
    tracking_dict[df.names[i]]={'x': [], 'y': []}
    tracking_dict[df.names[i]]['x'].append(df.x[i])
    tracking_dict[df.names[i]]['y'].append(df.y[i])

targets=np.array(df[df.type=='target'].names)
target_dict={}

for i in range(len(targets)):
    target_dict[targets[i]]={'Iteration killed':[], 'Cluster':[], 'Sensor': [], 'Strike':[]}

#%% Parameters 
#ranges of influence
c2range=25
strikerange=14
sensorrange=13
#max travel in period(iteration)
travellimit_C2=32
travellimit_strike=30
travellimit_sensor=43

#Indivdual Agency (lambda)
lamda=0.95
# Whole Cluster Movement -- group Agency (Phi)
movement_rate=0.51

#Individual asset movement allowed after cluster move (influenced by proportional ratio)
adjustlimit_c2=lamda*travellimit_C2
adjustlimit_strike=lamda*travellimit_strike
adjustlimit_sensor=lamda*travellimit_sensor

#cluster movement--limit imposed on cluster sync move-- based on what is left from adjustment allotment
clustermovelimit=min(travellimit_C2, travellimit_strike-adjustlimit_strike, travellimit_sensor-adjustlimit_sensor)

NM_crunch=0
directpathcount_strike=0
directpathcount_sensor=0


#Creating Clusters

begin_time=time.time()
#warnigns are turned off to no clutter output. Dataframe copy flags warning, but no error occurs 
pd.options.mode.chained_assignment = None  
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

iteration=0

while ((df['type']=='target') & (df['status']==1)).sum() > 0:
    
    # Creating Clusters
    
    iteration+=1
    
    
    #assets avilable needed to know the number of clusters formed
    activec2 = ((df['type']=='c2') & (df['status']==1)).sum()
    activestrike = ((df['type']=='strike') & (df['status']==1)).sum()
    activesensor = ((df['type']=='sensor') & (df['status']==1)).sum()
    numclusters = min(activec2, activestrike, activesensor) #lowest possible, distance will determine actual number 
    
    df_active_c2=df[(df['status']==1) & (df['type']=='c2')]
    df_active_c2=df_active_c2.reset_index(drop='True')
    
    
    #copy dataframe, this will update all information for that iteration-- then be purged
    df_clustering=df.copy()
    
    #distances between c2 used in clusters and all other assets
    for i in range(numclusters):
        distances=[]
        for j in range(len(df)):
            tempdist=np.sqrt((df_active_c2['x'][i]-df['x'][j])**2 + (df_active_c2['y'][i]-df['y'][j])**2)
            distances.append(tempdist)
        new_column=df_active_c2['names'][i]
        df_clustering= df_clustering.assign(**{new_column: distances})
    
    #Grouping Algorithm
    #assigning clusters
    #first, no clusters
    df_clustering['cluster']=0
    
    #preprocessing to determine any C2/cluster in eligible
    badc2=[]
    for i in range(numclusters):
        if df_clustering[(df_clustering.type=='sensor') & (df_clustering.status==1)].iloc[:,i+5].min() > c2range+ adjustlimit_sensor \
            or df_clustering[(df_clustering.type=='strike') & (df_clustering.status==1)].iloc[:,i+5].min() > c2range+ adjustlimit_strike:
            badc2.append(i+5)
        
    #how do we determine which ones cluster to keep if all are bad?
    if len(badc2)==numclusters:
        badc2.pop(0) #if all clusters deems unworkable, clusters will be made on the first index active c2
        numclusters=1
    else: numclusters= numclusters-len(badc2)
        
        
    df_clustering.drop(df_clustering.columns[badc2], axis=1, inplace = True)
    
    
    happyclusters=0 #flag that cluster has an active c2 paired with active st/s that can adjust if necessary
    while happyclusters==0 and numclusters!=0: #either we have viable clusters, or all assets move in sync in a single cluster
    
        df_clustering['cluster']=0
        
        #assign c2 to its own cluster
        for i in range(numclusters):
            c2index=df_clustering[(df_clustering.type=='c2') & (df_clustering.cluster==0) & (df_clustering.status==1)].iloc[:,i+5].idxmin()
            df_clustering['cluster'][c2index]=i+1
    
        #Find nearest active st/s not in cluster---grab
        for i in range(numclusters):
            strikeindex=df_clustering[(df_clustering.type=='strike') & (df_clustering.cluster==0) & (df_clustering.status==1)].iloc[:,i+5].idxmin()
            sensorindex=df_clustering[(df_clustering.type=='sensor') & (df_clustering.cluster==0) & (df_clustering.status==1)].iloc[:,i+5].idxmin()
            df_clustering['cluster'][strikeindex]=i+1
            df_clustering['cluster'][sensorindex]=i+1
        
            #distance check
            #greedy path--- let everyone be greedy --> can we remove the first greedy path then?
        for k in range (numclusters):
            i=numclusters-k-1 #works in reverse by stealing from lower indexed clusters
            
            #strike grab--greedy       
            #find current strike
            if ((df_clustering.type=='strike') & (df_clustering.cluster==i+1)).sum() >0: #if you still have a strike, find out where it is and determine it close enough
                strikeindex=df_clustering[(df_clustering.type=='strike') & (df_clustering.cluster==i+1) & (df_clustering.status==1)].iloc[:,i+5].idxmin()
                if (df_clustering.iloc[strikeindex,i+5]>adjustlimit_strike+c2range): 
                    df_clustering.cluster[strikeindex]=0     
            if ((df_clustering.type=='strike') & (df_clustering.cluster==i+1)).sum() == 0:   #adjustmentlimit+c2range?       
                for j in range(len(df_clustering)): #attempt to steal 
                    if df_clustering['cluster'][j] < i+1 and df_clustering.type[j]=='strike' and \
                        df_clustering['status'][j]==1 and df_clustering.iloc[j,i+5]<adjustlimit_strike+c2range:#adjustmentlimit+c2range?
                        df_clustering['cluster'][j]= i+1 #claim the fitting asset                    
                        break
                    
            #sensor grab--greedy        
            if ((df_clustering.type=='sensor') & (df_clustering.cluster==i+1)).sum() >0: #if you still have a sensor, find out where it is and determine it close enough
                sensorindex=df_clustering[(df_clustering.type=='sensor') & (df_clustering.cluster==i+1) & (df_clustering.status==1)].iloc[:,i+5].idxmin()
                if (df_clustering.iloc[sensorindex,i+5]>adjustlimit_sensor+c2range): 
                    df_clustering.cluster[sensorindex]=0     
            if ((df_clustering.type=='sensor') & (df_clustering.cluster==i+1)).sum() == 0:   #adjustmentlimit+c2range?       
                for j in range(len(df_clustering)): #attempt to steal 
                    if df_clustering['cluster'][j] < i+1 and df_clustering.type[j]=='sensor' and \
                        df_clustering['status'][j]==1 and df_clustering.iloc[j,i+5]<adjustlimit_sensor+c2range:#adjustmentlimit+c2range?
                        df_clustering['cluster'][j]= i+1 #claim the fitting asset                    
                        break    
    
                    
    
    
        #iterate through again and grab the closest remaining asset you need --accept
        for i in range(numclusters):
            if (((df_clustering.type=='strike') & (df_clustering.cluster==i+1)).sum()==0):
                addstrike=df_clustering[(df_clustering.type=='strike') & (df_clustering.cluster==0) & (df_clustering.status==1)].iloc[:,i+5].idxmin()
                #distance check
                if df_clustering.iloc[addstrike,i+5]<adjustlimit_strike+c2range:         #adjustmentlimit+c2range?   
                    df_clustering.cluster[addstrike]=i+1
                else:
                    df_clustering.cluster[addstrike]=0  #is this necessary?
            if (((df_clustering.type=='sensor') & (df_clustering.cluster==i+1)).sum()==0):
                addsensor=df_clustering[(df_clustering.type=='sensor') & (df_clustering.cluster==0) & (df_clustering.status==1)].iloc[:,i+5].idxmin()
                if df_clustering.iloc[addsensor,i+5]<adjustlimit_sensor+c2range:            #adjustmentlimit+range?
                    df_clustering.cluster[addsensor]=i+1
                else:
                    df_clustering.cluster[addsensor]=0   #is this necessary?
                    
        #happiness checker     
        happinesschecker=0        
        for i in range(numclusters):
            if (df_clustering.cluster==i+1).sum()==3:
                happinesschecker+=1
        if happinesschecker==numclusters:
            happyclusters=1
        else: 
            numclusters-=1  #if happiness does not occur, drop clusters by 1            
    
    
    #after exiting the big while loop, make sure numclusters is >0 if it got that low:    
    if numclusters==0:
        numclusters=1
    
    
    #assign remaining assets:
    for i in range(len(df_clustering)):
       if df_clustering.cluster[i]==0 and df_clustering.type[i] != 'target':
           temparray=df_clustering.iloc[i,5:5+numclusters]
           cluster_num=np.where(np.array(temparray)==np.amin(np.array(temparray)))[0][0]
           df_clustering.cluster[i]=cluster_num+1
          
    """
    # Plotting of the clusters
    
    for j in range(numclusters):
        cluster_plot=df_clustering[df_clustering['cluster']==j+1]
        cluster_plot=cluster_plot.reset_index(drop='True')
        
        annotations=cluster_plot['names']
        
        
        cluster_plot.plot.scatter(x='x',
                        y='y',
                        c=cluster_plot['type'].map(colors))
        
        for i, label in enumerate(annotations):
            plt.annotate(label, (cluster_plot['x'][i]+1, cluster_plot['y'][i]+1))
            
        plt.xlim(0,fieldlength)
        plt.ylim(0,fieldlength)    
        
    
        title='cluster ' + str(j+1) + '---Iteration ' + str(iteration)   
        plt.title(title)
        
        
        plt.show()
        """   
        
    #Tile Movement Algorithm    
    # Distance between clusters and remaining targets
        
    #center of gravity (centroid) for each cluster--using positions of only active units
    COG_x = []
    COG_y = []
    
    for i in range(numclusters):
        center_x = df_clustering.loc[(df_clustering['cluster'] == i+1)& (df_clustering['status']==1), 'x'].sum()/((df_clustering['cluster']==i+1)& (df_clustering['status']==1)).sum()
        COG_x.append(center_x)
        center_y =  df_clustering.loc[(df_clustering['cluster'] == i+1)& (df_clustering['status']==1), 'y'].sum()/((df_clustering['cluster']==i+1)& (df_clustering['status']==1)).sum()
        COG_y.append(center_y)
            

    #name clusters
    clusters=[]
    for i in range(numclusters):
        clusters.append('cluster_'+str(i+1))  
    
    ctdist_df=pd.DataFrame(clusters)
    
    #distance between clusters and targets    
    current_targets=df[df['type']=='target'][df['status']==1]
    current_targets=current_targets.reset_index(drop='True')
    activetargets=(current_targets['status']==1).sum()
    
    
    
    for j in range(activetargets):
        distances_t=[]
        for i in range(len(ctdist_df)):
            tempdist=np.sqrt((COG_x[i]-current_targets['x'][j])**2 + (COG_y[i]-current_targets['y'][j])**2)
            distances_t.append(tempdist)
        new_column=current_targets['names'][j]
        ctdist_df= ctdist_df.assign(**{new_column: distances_t})
        
        
    closesttarget=ctdist_df.iloc[:,1:].idxmin(axis = 1)
    
    
    #Limitation to Movement
    #move cluster as a unit a certain percentage of the way, or max distance. 
    
    #Need a wayto keep track of movement of assets during a priod
    df_clustering['distance']=0
    for i in range(len(df_clustering)):
        if df_clustering.type[i] == 'c2':
            df_clustering.distance[i] = min(travellimit_C2, clustermovelimit +adjustlimit_c2)
        elif df_clustering.type[i] =='sensor':
            df_clustering.distance[i] = min(travellimit_sensor, clustermovelimit+adjustlimit_sensor)
        elif df_clustering.type[i]=='strike':
            df_clustering.distance[i] = min(travellimit_strike, clustermovelimit+adjustlimit_strike)
        else: df_clustering.distance[i]=0
        
    #fix for integer assignment    
    df_clustering.distance = df_clustering.distance.astype(float)
        
  
    
    disttotarget_COG=[]
    for i in range(numclusters):
        disttotarget_COG.append(ctdist_df.min(axis=1)[i])
    
    #adjust movement rate basd on distance allowed
    clusterrate=[]
    for i in range(numclusters):
        if movement_rate*disttotarget_COG[i]<clustermovelimit:
            clusterrate.append(movement_rate)
        else:
            clusterrate.append(clustermovelimit/disttotarget_COG[i])
    
    
    clusterpaths_x=[]
    clusterpaths_y=[]
    
    for i in range(numclusters):
        TOI=df.loc[df['names'] == closesttarget[i]].index.tolist()[0]
        ypath=(df['y'][np.array(TOI)]-COG_y[i])
        xpath=(df['x'][np.array(TOI)]-COG_x[i])
        clusterpaths_y.append(ypath)
        clusterpaths_x.append(xpath)        
        
           
    
    
    df_clustering['new_x']=0
    df_clustering['new_y']=0
    df_clustering['new_x'] = df_clustering['new_x'].astype(float)
    df_clustering['new_y'] = df_clustering['new_y'].astype(float)
    
    
    for i in range(numclusters):
        for j in range(len(df_clustering)):
            if df_clustering['cluster'][j]==i+1:
                df_clustering['new_x'][j]=df_clustering['x'][j]+clusterrate[i]*clusterpaths_x[i]
                df_clustering['new_y'][j]=df_clustering['y'][j]+clusterrate[i]*clusterpaths_y[i]       
            elif df_clustering['cluster'][j]== 0 or df_clustering['cluster'][j]==-1:
                df_clustering['new_x'][j]=df_clustering['x'][j]
                df_clustering['new_y'][j]=df_clustering['y'][j]    
    
    #update the distance limits
    for i in range(numclusters):
        for j in range(len(df_clustering)):
            if df_clustering.cluster[j]==i+1:
                df_clustering.distance[j]=df_clustering.distance[j]-np.sqrt((df_clustering.new_x[j]-df_clustering.x[j])**2 + (df_clustering.new_y[j]-df_clustering.y[j])**2)
                if abs(df_clustering.distance[j])<  0.005:  #because of rounding errors
                    df_clustering.distance[j]=0
                    
    #The Nelder-Mead Adjustment-- To force cluster movement            
    stagnation_check=[]            
    for i in range(numclusters):
        alldisttraveled=[]
        for j in range(len(df_clustering)):       
            if df_clustering.cluster[j]==i+1:
                alldisttraveled.append(distance(df_clustering.x[j],df_clustering.y[j],df_clustering.new_x[j],df_clustering.new_y[j]))
        if sum(alldisttraveled)<0.01*clustermovelimit:
            stagnation_check.append(movement_rate)
            NM_crunch+=1
        else:
            stagnation_check.append(0)  
       
    ##pinch all assets to COG/target
    #need to find path between asset and COG/target
    xpath=[]
    ypath=[]
    dist_asset_COG=[]
    clusterrate=[]
    distancemoved=[]
    newxhold=[]
    newyhold=[]
    for i in range(len(df_clustering)):
        if df_clustering.type[i] != 'target':
            xpath.append(COG_x[df_clustering.cluster[i]-1]-df_clustering.new_x[i])
            ypath.append(COG_y[df_clustering.cluster[i]-1]-df_clustering.new_y[i])
            #find distance between COG and player
            dist_asset_COG.append(distance(df_clustering.new_x[i], df_clustering.new_y[i], COG_x[df_clustering.cluster[i]-1], COG_y[df_clustering.cluster[i]-1]))     
            #update movement rate (using cluster rate bc its already defined)
            distancemoved.append(clustermovelimit-distance(df_clustering.x[i],df_clustering.y[i],df_clustering.new_x[i],df_clustering.new_y[i]))
            if stagnation_check[df_clustering.cluster[i]-1]*dist_asset_COG[i] < distancemoved[i]:       
                clusterrate.append(stagnation_check[df_clustering.cluster[i]-1])
            else: 
                clusterrate.append(distancemoved[i]/dist_asset_COG[i])
            #determine new point
            newxhold.append(df_clustering.new_x[i])
            newyhold.append(df_clustering.new_y[i])
            df_clustering['new_x'][i]=df_clustering['new_x'][i]+clusterrate[i]*xpath[i]
            df_clustering['new_y'][i]=df_clustering['new_y'][i]+clusterrate[i]*ypath[i]   
            #update distance limits
            df_clustering.distance[i]=df_clustering.distance[i]-distance(newxhold[i],newyhold[i],df_clustering.new_x[i],df_clustering.new_y[i])
            #overflow fix
            if abs(df_clustering.distance[i])<  0.005:  #because of rounding errors
                df_clustering.distance[i]=0
     

                
    #Adjustment movements
    #Update remaining distances to adjustment only
    
    for i in range(len(df_clustering)):
        if df_clustering.type[i] == 'c2':
            df_clustering.distance[i] = min(df_clustering.distance[i], adjustlimit_c2)
        elif df_clustering.type[i] =='sensor':
            df_clustering.distance[i] = min(df_clustering.distance[i], adjustlimit_sensor)
        elif df_clustering.type[i]=='strike':
            df_clustering.distance[i] = min(df_clustering.distance[i], adjustlimit_strike)
        else: df_clustering.distance[i]=0            
        
    for m in range(len(df_clustering)):
        if df_clustering.distance[m]<0.001:
            df_clustering.distance[m]=0          
                
    #Individual Movement Algorithm                
    
    # Strike Adjustment Movement
    #adjust to within range of nearest active C2
    for i in range(numclusters): #do for all clusters
        temp_df_c2=df_clustering[(df_clustering['cluster']==i+1) & (df_clustering.type=='c2') & df_clustering.status==1] #find all active c2s in cluster
        temp_df_c2=temp_df_c2.reset_index(drop='True')
        TOI=df.loc[df['names'] == closesttarget[i]].index.tolist()[0]
        for j in range(len(df_clustering)):
            closestc2=[]
            if df_clustering.type[j]=='strike' and df_clustering.cluster[j]==i+1 and df_clustering.status[j]==1:
                for k in range(len(temp_df_c2)):
                    if temp_df_c2.cluster[k]==i+1:
                        tempdist= np.sqrt((df_clustering.new_x[j]-temp_df_c2.new_x[k])**2 + (df_clustering.new_y[j]-temp_df_c2.new_y[k])**2)
                        closestc2.append(tempdist)
                    else: #if c2 not in cluster set large value so not appealing
                        tempdist=9999
                        closestc2.append(tempdist)
                c2index=closestc2.index(min(closestc2))
                c2_strikedist=min(closestc2)
                if c2_strikedist>c2range: #move closer to C2
                    if c2_strikedist-c2range<df_clustering.distance[j]:   
                        travel_ratio=1-(c2range/c2_strikedist)
                    else:
                        travel_ratio= df_clustering.distance[j]/c2_strikedist
                    xpath=(temp_df_c2.new_x[c2index]-df_clustering.new_x[j])
                    ypath=(temp_df_c2.new_y[c2index]-df_clustering.new_y[j]) 
                    newerx=xpath*travel_ratio + df_clustering['new_x'][j]  
                    newery=ypath*travel_ratio + df_clustering['new_y'][j] 
                    #update distance traveled
                    df_clustering.distance[j]=df_clustering.distance[j]-np.sqrt((df_clustering.new_x[j]-newerx)**2 + (df_clustering.new_y[j]-newery)**2)
                    #first move complete
                    df_clustering['new_x'][j]=xpath*travel_ratio + df_clustering['new_x'][j]                  
                    df_clustering['new_y'][j]=ypath*travel_ratio + df_clustering['new_y'][j]  
                    
                for m in range(len(df_clustering)):
                    if df_clustering.distance[m]<0.001:
                        df_clustering.distance[m]=0   
                    
                    
                #now get within range of target (if applicable)
                st_t_dist= np.sqrt((df_clustering.new_x[j]-df_clustering.new_x[TOI])**2 + (df_clustering.new_y[j]-df_clustering.new_y[TOI])**2)
                if st_t_dist>strikerange and df_clustering.distance[j]>0: #only move again if not in range of target
                    c2_st_dist=np.sqrt((df_clustering.new_x[j]-temp_df_c2.new_x[c2index])**2 + (df_clustering.new_y[j]-temp_df_c2.new_y[c2index])**2)
                    c2_t_dist=np.sqrt((df_clustering.new_x[TOI]-temp_df_c2.new_x[c2index])**2 + (df_clustering.new_y[TOI]-temp_df_c2.new_y[c2index])**2)
                    
                    if c2_t_dist<c2range:
                        directpathcount_strike+=1
                        finalmove_ratio=min(df_clustering.distance[j], st_t_dist-strikerange)/st_t_dist
                        ypath=df_clustering.new_y[TOI]-df_clustering.new_y[j]#just b-line towards target as much as possible
                        xpath=df_clustering.new_x[TOI]-df_clustering.new_x[j]
                        df_clustering.new_y[j]=df_clustering.new_y[j]+ypath*finalmove_ratio
                        df_clustering.new_x[j]=df_clustering.new_x[j]+xpath*finalmove_ratio
                        #update its move                   
                        df_clustering.distance[j]=df_clustering.distance[j]-min(c2range, df_clustering.distance[j], st_t_dist-strikerange)                    
                    
                    elif c2_st_dist<0.001:  #if this occurs with two assets on top of eachother the trig WILL NOT WORK
                        finalmove_ratio=min(df_clustering.distance[j], st_t_dist-strikerange, c2range)/st_t_dist
                        ypath=df_clustering.new_y[TOI]-df_clustering.new_y[j]#just b-line towards target as much as possible
                        xpath=df_clustering.new_x[TOI]-df_clustering.new_x[j]
                        df_clustering.new_y[j]=df_clustering.new_y[j]+ypath*finalmove_ratio
                        df_clustering.new_x[j]=df_clustering.new_x[j]+xpath*finalmove_ratio
                        #update its move                   
                        df_clustering.distance[j]=df_clustering.distance[j]-min(c2range, df_clustering.distance[j], st_t_dist-strikerange)
        
                        
                    else:
                        #find theta
                        thetanum=(c2_st_dist)**2+(st_t_dist)**2-(c2_t_dist)**2
                        thetadenom= 2*c2_st_dist*st_t_dist
                        theta=math.acos(thetanum/thetadenom)
                        #find second angle
                        theta2=math.asin((math.sin(theta)*c2_st_dist)/c2range)
                        theta3=math.radians(180-math.degrees(theta)-math.degrees(theta2))
                        golddist=c2range*math.sin(theta3)/math.sin(theta)
                        if golddist <0:
                            golddist=0
                        #now move that far, or whatever distance you can left
                        finalmove_ratio=min(golddist, df_clustering.distance[j], st_t_dist-strikerange)/st_t_dist
                        ypath=df_clustering.new_y[TOI]-df_clustering.new_y[j]
                        xpath=df_clustering.new_x[TOI]-df_clustering.new_x[j]
                        df_clustering.new_y[j]=df_clustering.new_y[j]+ypath*finalmove_ratio
                        df_clustering.new_x[j]=df_clustering.new_x[j]+xpath*finalmove_ratio
                           
                        #update its move                   
                        df_clustering.distance[j]=df_clustering.distance[j]-min(golddist, df_clustering.distance[j], st_t_dist-strikerange)
        
     
    #Sensor Adjustment Movement 
    #adjust to within range of nearest active C2
    for i in range(numclusters): #do for all clusters
        temp_df_c2=df_clustering[(df_clustering['cluster']==i+1) & (df_clustering.type=='c2') & df_clustering.status==1] #find all active c2s in cluster
        temp_df_c2=temp_df_c2.reset_index(drop='True')
        TOI=df.loc[df['names'] == closesttarget[i]].index.tolist()[0]
        for j in range(len(df_clustering)):
            closestc2=[]
            if df_clustering.type[j]=='sensor' and df_clustering.cluster[j]==i+1 and df_clustering.status[j]==1:
                for k in range(len(temp_df_c2)):
                    if temp_df_c2.cluster[k]==i+1:
                        tempdist= np.sqrt((df_clustering.new_x[j]-temp_df_c2.new_x[k])**2 + (df_clustering.new_y[j]-temp_df_c2.new_y[k])**2)
                        closestc2.append(tempdist)
                    else: #if c2 not in cluster set large value so not appealing
                        tempdist=9999
                        closestc2.append(tempdist)
                c2index=closestc2.index(min(closestc2))
                c2_s_dist=min(closestc2)
                if c2_s_dist>c2range:
                    if c2_s_dist-c2range<df_clustering.distance[j]:   
                        travel_ratio=1-(c2range/c2_s_dist)
                    else:
                        travel_ratio= df_clustering.distance[j]/c2_s_dist
                    xpath=(temp_df_c2.new_x[c2index]-df_clustering.new_x[j])
                    ypath=(temp_df_c2.new_y[c2index]-df_clustering.new_y[j]) 
                    newerx=xpath*travel_ratio + df_clustering['new_x'][j]  
                    newery=ypath*travel_ratio + df_clustering['new_y'][j] 
                    #update distance traveled
                    df_clustering.distance[j]=df_clustering.distance[j]-np.sqrt((df_clustering.new_x[j]-newerx)**2 + (df_clustering.new_y[j]-newery)**2)
                    #first move complete
                    df_clustering['new_x'][j]=xpath*travel_ratio + df_clustering['new_x'][j]                  
                    df_clustering['new_y'][j]=ypath*travel_ratio + df_clustering['new_y'][j]   
                    
                for m in range(len(df_clustering)):
                    if df_clustering.distance[m]<0.001:
                        df_clustering.distance[m]=0   
                    
                    
                #now get within range of target (if applicable)
                
                s_t_dist= np.sqrt((df_clustering.new_x[j]-df_clustering.new_x[TOI])**2 + (df_clustering.new_y[j]-df_clustering.new_y[TOI])**2)
                if s_t_dist>sensorrange and df_clustering.distance[j]>0: #only move again if not in range of target
                    c2_s_dist=np.sqrt((df_clustering.new_x[j]-temp_df_c2.new_x[c2index])**2 + (df_clustering.new_y[j]-temp_df_c2.new_y[c2index])**2)
                    c2_t_dist=np.sqrt((df_clustering.new_x[TOI]-temp_df_c2.new_x[c2index])**2 + (df_clustering.new_y[TOI]-temp_df_c2.new_y[c2index])**2)
                    if c2_t_dist<c2range:
                        directpathcount_sensor+=1
                        finalmove_ratio=min(df_clustering.distance[j], s_t_dist-sensorrange)/s_t_dist
                        ypath=df_clustering.new_y[TOI]-df_clustering.new_y[j]#just b-line towards target as much as possible
                        xpath=df_clustering.new_x[TOI]-df_clustering.new_x[j]
                        df_clustering.new_y[j]=df_clustering.new_y[j]+ypath*finalmove_ratio
                        df_clustering.new_x[j]=df_clustering.new_x[j]+xpath*finalmove_ratio
                        #update its move                   
                        df_clustering.distance[j]=df_clustering.distance[j]-min(c2range, df_clustering.distance[j], s_t_dist-sensorrange)                    
                                   
                    
                    
                    elif c2_s_dist<0.001:  #if this occurs with two assets on top of eachother the trig WILL NOT WORK
                        finalmove_ratio=min(df_clustering.distance[j], s_t_dist-sensorrange, c2range)/s_t_dist
                        ypath=df_clustering.new_y[TOI]-df_clustering.new_y[j]#just b-line towards target as much as possible
                        xpath=df_clustering.new_x[TOI]-df_clustering.new_x[j]
                        df_clustering.new_y[j]=df_clustering.new_y[j]+ypath*finalmove_ratio
                        df_clustering.new_x[j]=df_clustering.new_x[j]+xpath*finalmove_ratio
                        
                        #update its move
                        
                        df_clustering.distance[j]=df_clustering.distance[j]-min(c2range, df_clustering.distance[j], s_t_dist-sensorrange)                      
                        
                    else:
                        #find theta
                        thetanum=(c2_s_dist)**2+(s_t_dist)**2-(c2_t_dist)**2
                        thetadenom= 2*c2_s_dist*s_t_dist
                        theta=math.acos(thetanum/thetadenom)
                        #find second angle
                        theta2=math.asin((math.sin(theta)*c2_s_dist)/c2range)
                        theta3=math.radians(180-math.degrees(theta)-math.degrees(theta2))
                        golddist=c2range*math.sin(theta3)/math.sin(theta)
                        if golddist <0:
                            golddist=0
                        #now move that far, or whatever distance you can left
                        finalmove_ratio=min(golddist, df_clustering.distance[j], s_t_dist-sensorrange)/s_t_dist
                        ypath=df_clustering.new_y[TOI]-df_clustering.new_y[j]
                        xpath=df_clustering.new_x[TOI]-df_clustering.new_x[j]
                        df_clustering.new_y[j]=df_clustering.new_y[j]+ypath*finalmove_ratio
                        df_clustering.new_x[j]=df_clustering.new_x[j]+xpath*finalmove_ratio  
    
                        #update its move
                    
                        df_clustering.distance[j]=df_clustering.distance[j]-min(golddist, df_clustering.distance[j], s_t_dist-sensorrange)                      
                        
    #Update the tracking dictionary
    
    
    for i in range(len(df_clustering)):
        tracking_dict[df_clustering.names[i]]['x'].append(df_clustering.new_x[i])
        tracking_dict[df_clustering.names[i]]['y'].append(df_clustering.new_y[i])
    
    #Check to see if appropiate assets are within range of target
    
    df_targets =df_clustering[(df_clustering.type=='target') & (df_clustering.status==1)] #find all active targets
    df_targets=df_targets.reset_index(drop='True')
    
    
    
    for i in range(len(df_targets)):
        for j in range(numclusters):
            strikeflag=0
            sensorflag=0
            for k in range(len(df_clustering)):
                if df_clustering.type[k]=='strike' and df_clustering.cluster[k]==j+1 and df_clustering.status[k]==1:
                    strike_dist=np.sqrt((df_clustering.new_x[k]-df_targets.new_x[i])**2 + (df_clustering.new_y[k]-df_targets.new_y[i])**2)
                    if strike_dist<strikerange:
                        strikeflag=1
                        strikename=df_clustering.names[k]
                elif df_clustering.type[k]=='sensor' and df_clustering.cluster[k]==j+1 and df_clustering.status[k]==1:
                    sensor_dist=np.sqrt((df_clustering.new_x[k]-df_targets.new_x[i])**2 + (df_clustering.new_y[k]-df_targets.new_y[i])**2)
                    if sensor_dist<sensorrange:
                        sensorflag=1  
                        sensorname=df_clustering.names[k]
            if (strikeflag==1) & (sensorflag==1):
                target_to_remove=df_targets.names[i]
                target_index=df_clustering.index[df['names'] == target_to_remove]
                df_clustering.status[target_index]=2
                df.status[target_index]=2  
                #update target dictionary
                target_dict[target_to_remove]['Iteration killed'].append(iteration)
                target_dict[target_to_remove]['Cluster'].append(j+1)
                target_dict[target_to_remove]['Sensor'].append(sensorname)
                target_dict[target_to_remove]['Strike'].append(strikename)
    
                
                
    # end of iteration
    #end the iteration by resetting the locations of assets to original dataframe
    
    #need to put each asset in its own array
    
    df['x']= df_clustering['new_x']
    df['y']= df_clustering['new_y']
    
    
    
    
    

    del df_clustering            
    del df_targets
    del df_active_c2
    
    # plot the graph of new locations
    
    
    """
    df.plot.scatter(x='x',
                    y='y',
                    c=df['type'].map(colors))
    
    for i, label in enumerate(clean_names):
        plt.annotate(label, (df['x'][i]+1, df['y'][i]+1))
        
    plt.xlim(0,fieldlength)
    plt.ylim(0,fieldlength)     
    
    title='End Iteration ' + str(iteration)   
    plt.title(title)
    
    plt.show()
    """
    
    
elapsed=time.time()-begin_time
    
    



#End of Operation analysis

#To plotn asset flight paths
#The path plotter:
df_targets=df[(df.type=='target')] 
df_targets=df_targets.reset_index(drop='True')

c2names=clean_names[0:(df.type=='c2').sum()]
sensornames=clean_names[(df.type=='c2').sum():(df.type=='sensor').sum()+(df.type=='c2').sum()]
strikenames=clean_names[(df.type=='sensor').sum()+(df.type=='c2').sum():(df.type=='sensor').sum()+(df.type=='c2').sum()+(df.type=='strike').sum()]
targetnames=clean_names[(df.type=='sensor').sum()+(df.type=='c2').sum()+(df.type=='strike').sum():]

#plot the c2's paths
df_c2 =df[(df.type=='c2')] 
df_c2=df_c2.reset_index(drop='True')

for i in range((df.type=='c2').sum()):
    tempkey= df_c2.names[i]
    plt.scatter(tracking_dict[tempkey]['x'],tracking_dict[tempkey]['y'])
    plt.plot(tracking_dict[tempkey]['x'],tracking_dict[tempkey]['y'])
plt.scatter(df_targets.x, df_targets.y, c='r')    
 
for i, label in enumerate(targetnames):
    plt.annotate(label, (df_targets.x[i]+1, df_targets.y[i]+1))
for i, label in enumerate(c2names):
    plt.annotate(label, (starting_position[0][i]+1, starting_position[1][i]+1))


plt.xlim(0,fieldlength)
plt.ylim(0,fieldlength)     

plt.title('c2')
plt.show()


#plot the sensor's paths
df_sensor =df[(df.type=='sensor')] 
df_sensor=df_sensor.reset_index(drop='True')

for i in range((df.type=='sensor').sum()):
    tempkey= df_sensor.names[i]
    plt.scatter(tracking_dict[tempkey]['x'],tracking_dict[tempkey]['y'])
    plt.plot(tracking_dict[tempkey]['x'],tracking_dict[tempkey]['y'])
plt.scatter(df_targets.x, df_targets.y, c='r')    
 
for i, label in enumerate(targetnames):
    plt.annotate(label, (df_targets.x[i]+1, df_targets.y[i]+1))
for i, label in enumerate(sensornames):
    plt.annotate(label, (starting_position[0][i+(df.type=='c2').sum()]+1, starting_position[1][i+(df.type=='c2').sum()]+1))


plt.xlim(0,fieldlength)
plt.ylim(0,fieldlength)     

plt.title('Sensors')
plt.show()

#plot the strike's paths
df_strike =df[(df.type=='strike')] 
df_strike=df_strike.reset_index(drop='True')

for i in range((df.type=='strike').sum()):
    tempkey= df_strike.names[i]
    plt.scatter(tracking_dict[tempkey]['x'],tracking_dict[tempkey]['y'])
    plt.plot(tracking_dict[tempkey]['x'],tracking_dict[tempkey]['y'])
plt.scatter(df_targets.x, df_targets.y, c='r')    
 
for i, label in enumerate(targetnames):
    plt.annotate(label, (df_targets.x[i]+1, df_targets.y[i]+1))
for i, label in enumerate(strikenames):
    plt.annotate(label, (starting_position[0][i+(df.type=='c2').sum()+(df.type=='sensor').sum()]+1, starting_position[1][i+(df.type=='c2').sum()+(df.type=='sensor').sum()]+1))


plt.xlim(0,fieldlength)
plt.ylim(0,fieldlength)     

plt.title('Strikes')
plt.show()    



#calc distance traveled for all assets
disttraveled_total=[]
for j in range(len(df)):
    total=0
    for i in range(iteration):
        tempdist=distance(tracking_dict[df.names[j]]['x'][i], tracking_dict[df.names[j]]['y'][i],tracking_dict[df.names[j]]['x'][i+1],tracking_dict[df.names[j]]['y'][i+1])
        total=total+tempdist
    disttraveled_total.append(total)

numtargets= len(df[df['type']=='target'])
    
d={'player': np.array(df.names)[0:len(df)-numtargets], 'distance_traveled': disttraveled_total[0:len(df)-numtargets]}
df_dist = pd.DataFrame(data=d)   
  
#determine average distances
totc2=df[df.type=='c2'].type.count()
tots=df[df.type=='sensor'].type.count()
totst=df[df.type=='strike'].type.count()

c2dist_avg=df_dist.distance_traveled[0:totc2].sum()/df[df.type=='c2'].type.count()
sdist_avg=df_dist.distance_traveled[totc2:totc2+tots].sum()/df[df.type=='sensor'].type.count()
stdist_avg=df_dist.distance_traveled[totc2+tots:len(df_dist)].sum()/df[df.type=='strike'].type.count()

ddist={'Asset': ['c2','sensor','strike'], 'Average distance': [c2dist_avg,sdist_avg, stdist_avg]}
df_avgdist=pd.DataFrame(data=ddist)



#calculate number of engagements
numengagements=0
for i in range(len(target_dict)):
    keyname='target_'+str(i+1)
    numengagements+= len(target_dict[keyname]['Iteration killed'])



#price of anarachy (engagements)

POA_engagement = numengagements/len(target_dict)

#print outs for reference
print('Time Elapsed in seconds: '+str(elapsed))
print('Number of iterations for operation: '+ str(iteration))
print('Number of NM-like shrinks occured: '+ str(NM_crunch))
print('Total distance traveled: '+ str(np.array(disttraveled_total[0:len(df)-numtargets]).sum()))
print('Total number of engagements: '+ str(numengagements))
print('POA using engagements: '+str(POA_engagement))
print(str(df_avgdist))
