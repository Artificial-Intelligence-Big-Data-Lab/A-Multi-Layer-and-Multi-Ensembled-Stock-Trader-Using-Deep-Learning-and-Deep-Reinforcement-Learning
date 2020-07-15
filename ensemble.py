import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def majority_voting(df):
    
    local_df = df.copy()
    x=local_df.loc[:,'iteration0':'iteration24']
    local_df['ensemble']=x.mode(axis=1).iloc[:, 0]
    local_df = local_df.drop(local_df.columns.difference(['ensemble']), axis=1)
    return local_df
    
def ensemble(type, ensembleFolderName):

    dollSum=0
    rewSum=0
    posSum=0
    negSum=0
    covSum=0 
    numSum=0

    values=[]
    columns = ["Experiment","#Wins","#Losses","Dollars","Coverage","Accuracy"]
    
    sp500=pd.read_csv("./dataset/jpm/test_data.csv",index_col='date_time')
      
    df=pd.read_csv("./Output/ensemble/"+ensembleFolderName+"/ensemble_"+type+".csv",index_col='date_time')
        
    df=majority_voting(df)
      
    num=0
    rew=0
    pos=0
    neg=0
    doll=0
    cov=0

  
    #Lets iterate through each date and decision 
    for date, i in df.iterrows():
     
        #If the date in the predictions is in the index of sp500 (which is also a date) 
        if date in sp500.index:
             
            num+=1
              
            #If the output is 1 (long)
            if (i['ensemble']==1):
                   
                #If the close - open is positive at that day, we have earning money. Positives are equal to 1. Otherwise, no incrementation 
                pos+= 1 if (float(sp500.at[date,'delta_next_day'])) > 0 else 0

                #If close - open is negative at that day, we are losing money. Negatives are equal to 1. Otherwise, no incrementation 
                neg+= 1 if (float(sp500.at[date,'delta_next_day'])) < 0 else 0

                #Lets calculate the reward (positive or negative)
                rew+=float(sp500.at[date,'delta_next_day'])
                    
                #In dollars, we just multiply by the sp500 points by the differences 
                doll+=float(sp500.at[date,'delta_next_day'])

                #There is coverage (of course) 
                cov+=1

            #The same stuff happens for short.
            elif (i['ensemble']==2):
     
                pos+= 1 if float(sp500.at[date,'delta_next_day']) < 0 else 0
                neg+= 1 if float(sp500.at[date,'delta_next_day']) > 0 else 0
                    
                rew+=-float(sp500.at[date,'delta_next_day'])
                cov+=1
                doll+=-float(sp500.at[date,'delta_next_day'])
                    


    values.append([str(1),str(round(pos,2)),str(round(neg,2)),str(round(doll,2)),str(round(cov/num,2)),(str(round(pos/cov,2)) if (cov>0) else "")])
        
    #Now lets sum walk by walk 
    dollSum+=doll
    rewSum+=rew
    posSum+=pos
    negSum+=neg
    covSum+=cov
    numSum+=num

    
    #Now lets summarize everything showing the sum of values 
    values.append(["sum",str(round(posSum,2)),str(round(negSum,2)),str(round(dollSum,2)),str(round(covSum/numSum,2)),(str(round(posSum/covSum,2)) if (covSum>0) else "")])
    
    return values,columns
    
    
################
