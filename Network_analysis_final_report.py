##############################original code   
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  7 16:01:39 2020

@author: Asus
"""
import pickle
import requests
import lxml
from bs4 import BeautifulSoup
from urllib.request import urlopen
import os
import pandas as pd
import numpy as np
import networkx as nx
from modularity_maximization import partition
from modularity_maximization.utils import get_modularity
from sknetwork.clustering import Louvain, BiLouvain, modularity, bimodularity
from sknetwork.data import karate_club, painters, movie_actor
from sknetwork.visualization import svg_graph, svg_digraph, svg_bigraph
import numpy.linalg
import statsmodels.formula.api as sm
import matplotlib.pyplot as plt


os.chdir("C:/Colorado/Network analysis/Final project")
import csv    
headers = {'User-Agent':'Mozilla/5.0',
               'Authorization': 'token 2e72b74a5487de5bcb90e98523db483c2bac0183',
               'Content-Type':'application/json',
               'Accept':'application/vnd.github.v3.star+json'
               }


organization_list=pd.read_csv('organization_list.csv')

data= []
for i in range(len(organization_list)):
    j=0
    url='https://api.github.com/orgs/'+str(organization_list['Organization'][i])+'/members?page='+str(j)+'&per_page=100'    ###member url for the organization
    member=[]
    temp= requests.get(url,headers=headers).json()
    for org in temp:
        org.update({'Organization':str(organization_list['Organization'][i])})  ##append the organization name
    data=data+(temp)

    while len(temp) !=0:            
            j=j+1
            url='https://api.github.com/orgs/'+str(organization_list['Organization'][i])+'/members?page='+str(j)+'&per_page=100'
            temp= requests.get(url,headers=headers).json()
            for org in temp:
                org.update({'Organization':str(organization_list['Organization'][i])})
            
            data=data+(temp)     ###The list of all members


data=[dict(t) for t in {tuple(d.items()) for d in data}]   ###drop duplicates

### drop microsoft
for i in data:
    if "Microsoft" in i.values():
        data.remove(i)    

tocsv=data
keys = tocsv[0].keys()               
with open('member.csv', 'w', encoding='utf8', newline='')  as output_file:
    dict_writer = csv.DictWriter(output_file, keys)
    dict_writer.writeheader()
    dict_writer.writerows(tocsv) 


data=pd.read_csv('member.csv')  
data=data.drop_duplicates(subset=['id'])   ####drop duplicates
data=data.reset_index(drop=True)


member_id=data['id']
member_id=set(member_id)


#######update the knowledge flow for each member
updated_data=[{} for i in range(len(data))]

for i in range(0,len(updated_data)):
    ####get the information of each member
    member_url=data['url'][i]
    member_data=requests.get(member_url,headers=headers).json()    
    updated_data[i].update(member_data)
    updated_data[i].update({'Organization':data['Organization'][i]})
    

    ##### record the ID of members that are starred by the focal member 
    index=0
    star_url=data['starred_url'][i].split('{/owner}{/repo}')[0]+'?page='+str(index)+'&per_page=100'
    star_data=requests.get(star_url,headers=headers).json()    ###record the information of repos that are starred by the member
    star_list=[]
    
    while len(star_data) !=0:
        for j in range(len(star_data)):
            if star_data[j]['repo']['owner'] is not None:     ###check for Nonetype              
                starred_ID=star_data[j]['repo']['owner']['id']    ### starred repo owner's ID
                star_list.append(starred_ID)
        index +=1
        star_url=data['starred_url'][i].split('{/owner}{/repo}')[0]+'?page='+str(index)+'&per_page=100'
        star_data=requests.get(star_url,headers=headers).json()
        
    star_list=set(star_list)
    starred_member=list(star_list.intersection(member_id))   ### ID of starred member
    temp={'starred_member':starred_member}
    updated_data[i].update(temp)
    






    
tocsv=updated_data
keys = tocsv[0].keys()               
with open('data-17.csv', 'w', encoding='utf8', newline='')  as output_file:
    dict_writer = csv.DictWriter(output_file, keys)
    dict_writer.writeheader()
    dict_writer.writerows(tocsv)    
    
    
network_data= pd.read_csv("data_11_28.csv",encoding = "ISO-8859-1")
network_data=network_data.dropna(subset=['login'])
network_data['id']=data['id']

c=network_data['starred_member']
c=c.dropna()
c=c.tolist()

###destring the edge list
for i in range(len(network_data)):
        c[i]=eval(c[i])

edgelist=[]


for i in range(len(network_data)):
    if len(c[i]) !=0:
        for j in range(len(c[i])):
            if data['id'][i] !=c[i][j]:
                temp=(data['id'][i],c[i][j])
                edgelist.append(temp)
            
source=pd.DataFrame() 
source['a']=[None for i in range(0,len(edgelist))]   
source['b']=[None for i in range(0,len(edgelist))]     
for i in range(len(edgelist)):
    source['a'][i]=edgelist[i][0]
    source['b'][i]=edgelist[i][1]

source.to_csv('edgelist-2020-11-18.csv')








    
######network analysis
G = nx.DiGraph()
G.add_nodes_from(network_data['id'])  
G.add_edges_from(edgelist) 

####drop isolated nodes
isolated_node=list(nx.isolates(G))
    
for i in range(len(network_data)):
    if network_data['id'][i] in isolated_node:
        network_data=network_data.drop(index=i)
        
network_data=network_data.reset_index(drop=True)

###import the language attribute
language=pd.read_csv('org_rank.csv')

network_data=network_data.join(language.set_index('Organization'), on='Organization')
network_data.rename(columns={'Unnamed: 2':'language'}, inplace=True)

network_data.to_csv('member_data.csv')

G.remove_nodes_from(list(nx.isolates(G)))


####add attribute to the network
attribute_list=['hireable','Organization','language']

for i in attribute_list:
    Attribute=pd.DataFrame() 
    Attribute['ID']=network_data['id']
    Attribute[i]=network_data[i]
    temp=dict(Attribute.values.tolist())
    nx.set_node_attributes(G, temp, str(i))




##############################################3
###### 
###centrality

#######calculate the page rank
pagerank= nx.pagerank(G)

pagerank_data=pd.DataFrame(pagerank.items())

####calculate the Katz Centrality

katzrank=nx.katz_centrality(G,0.02)

katzrank_data=pd.DataFrame(katzrank.items())

####calculate the indegree centrality
indegreerank=nx.in_degree_centrality(G)

indegreerank_data=pd.DataFrame(indegreerank.items())

data=pd.read_csv('member_data.csv')
data['pagerank']= pagerank_data[1]
data['katzrank']= katzrank_data[1]
data['indegreerank']= indegreerank_data[1]

#####the impact of blog on stargazers
data['blog_dummy']=data['blog']
data['blog_dummy'] = data['blog_dummy'].fillna(0)


for i in range(len(data['blog_dummy'])):
    if type(data['blog_dummy'][i]) ==str:
        data['blog_dummy'][i]=1

data.groupby('blog_dummy')['indegreerank'].mean()
data.groupby('blog_dummy')['katzrank'].mean()
data.groupby('blog_dummy')['pagerank'].mean()


####the impact of recruitment incentive on stargazaers
data['hireable_dummy']=data['hireable']
data['hireable_dummy'] = data['hireable_dummy'].fillna(0)


for i in range(len(data['hireable_dummy'])):
    if data['hireable_dummy'][i] ==True:
        data['hireable_dummy'][i]=1
        

data.groupby('hireable_dummy')['indegreerank'].mean()
data.groupby('hireable_dummy')['katzrank'].mean()
data.groupby('hireable_dummy')['pagerank'].mean()



###Top 50 developers with the highest rank 
def ranklist(rank,data,ranktype,reverse):
    rank_list={k: v for k, v in sorted(rank.items(), key=lambda item: item[1],reverse=reverse)[:50]}
    developer_ID=list(rank_list.keys())

    developer_list= data[0:0]
    developer_list[ranktype]=np.nan

    for i in developer_ID:    
        temp=data[(data['id']==i)]
        temp[ranktype]=rank_list[i]
        developer_list=developer_list.append(temp)

        
    return developer_list


pagerank_list=ranklist(pagerank,network_data,'Pagerank',True)    
pagerank_list=pagerank_list.reset_index(drop=True)

katzrank_list=ranklist(katzrank,network_data,'katzrank',True) 
katzrank_list=katzrank_list.reset_index(drop=True)
 

indegreerank_list=ranklist(indegreerank,network_data,'indegreerank',True)  
indegreerank_list=indegreerank_list.reset_index(drop=True)


pagerank_name=list(pagerank_list['name'])

katzrank_name=list(katzrank_list['name'])


indegreerank_name=list(indegreerank_list['name'])

###common user in all the ranks
user_list=pd.DataFrame(set(pagerank_name) & set(katzrank_name) & set(indegreerank_name),columns=['name'])

def attribute(data,ranklist,rank,variable):
    data[rank]=np.nan
    for i in range(len(data)):
        j=0
        while not ranklist[variable][j]==data[variable][i]:
            j+=1
    
        data[rank][i]=ranklist[rank][j]  

    return data
    

user_list=attribute(user_list,pagerank_list,'Pagerank','name')

user_list=attribute(user_list,katzrank_list,'katzrank','name')

user_list=attribute(user_list,indegreerank_list,'indegreerank','name')

user_list=attribute(user_list,indegreerank_list,'followers','name')

user_list=attribute(user_list,indegreerank_list,'Organization','name')

user_list.to_csv('userlist.csv')



####Top organization
pagerank_org=data.groupby('Organization')['pagerank'].mean()
pagerank_org.to_csv('pagerank_org.csv')
pagerank_org=pagerank_org.rank(ascending=False)
pagerank_org=pagerank_org.to_frame()


katzrank_org=data.groupby('Organization')['katzrank'].mean()
katzrank_org.to_csv('katzrank_org.csv')
katzrank_org=katzrank_org.rank(ascending=False)
katzrank_org=katzrank_org.to_frame()



indegreerank_org=data.groupby('Organization')['indegreerank'].mean()
indegreerank_org.to_csv('indegreerank_org.csv')
indegreerank_org=indegreerank_org.rank(ascending=False)
indegreerank_org=indegreerank_org.to_frame()



org_rank=pagerank_org.join(katzrank_org)
org_rank=org_rank.join(indegreerank_org)
org_rank=org_rank.mean(axis=1)
org_rank.to_csv('org_rank.csv')



#####pattern detection
adjacency = nx.adjacency_matrix(G)
louvain = Louvain()
labels = louvain.fit_transform(adjacency)
labels_unique, counts = np.unique(labels, return_counts=True)

optimal_modularity=modularity(adjacency, labels)




#####modularity of the attribute
organization=network_data['Organization']
organization=organization.to_numpy()
organization_label= pd.factorize(organization)[0] 

organization_modularity=modularity(adjacency, organization_label)   

hireable=network_data['hireable']
hireable=hireable.to_numpy()
hireable_label=pd.factorize(hireable)[0] 
hireable_modularity=modularity(adjacency, hireable_label)   


language=network_data['language']
language=language.to_numpy()
language_label=pd.factorize(language)[0] 
language_modularity=modularity(adjacency, language_label)  



####regression on the followership
pagerank_reg = sm.ols(formula="pagerank ~ blog_dummy+hireable_dummy+followers+public_repos", data=data).fit()
print(pagerank_reg.summary())


plt.rc('figure', figsize=(12, 7))
plt.text(0.01, 0.05, str(pagerank_reg.summary()), {'fontsize': 10}, fontproperties = 'monospace')
plt.axis('off')
plt.tight_layout()
plt.savefig('output.png')

##############################
katzrank_reg = sm.ols(formula="katzrank ~ blog_dummy+hireable_dummy+followers+public_repos", data=data).fit()
print(pagerank_reg.summary())

plt.rc('figure', figsize=(12, 7))
plt.text(0.01, 0.05, str(katzrank_reg.summary()), {'fontsize': 10}, fontproperties = 'monospace') 
plt.axis('off')
plt.tight_layout()
plt.savefig('output.png')

###############################
indegreerank_reg = sm.ols(formula="indegreerank ~ blog_dummy+hireable_dummy+followers+public_repos", data=data).fit()
print(pagerank_reg.summary())


plt.rc('figure', figsize=(12, 7))
plt.text(0.01, 0.05, str(indegreerank_reg.summary()), {'fontsize': 10}, fontproperties = 'monospace')
plt.axis('off')
plt.tight_layout()
plt.savefig('output.png')
 








