#Callbacks are functions used to give a feedback about each epoch calculated metrics
from rl.callbacks import Callback

class ValidationCallback(Callback):

    def __init__(self):
        #Initially, the metrics are zero
        self.episodes = 0
        self.rewardSum = 0
        self.accuracy = 0
        self.coverage = 0
        self.short = 0
        self.long = 0
        self.shortAcc =0
        self.longAcc =0
        self.longPrec =0
        self.shortPrec =0
        self.marketRise =0
        self.marketFall =0

    def reset(self):
        #The metrics are also zero when the epoch ends
        self.episodes = 0
        self.rewardSum = 0
        self.accuracy = 0
        self.coverage = 0
        self.short = 0
        self.long = 0
        self.shortAcc =0
        self.longAcc =0
        self.longPrec =0
        self.shortPrec =0
        self.marketRise =0
        self.marketFall =0
        
    #all information is given by the environment: action, reward and market
    #Then, when the episode ends, metrics are calculated
    def on_episode_end(self, action, reward, market):
        
        #After the episode ends, increments the episodes 
        self.episodes+=1

        #Increments the reward
        self.rewardSum+=reward

        #If the action is not a hold, there is coverage because the agent decided 
        self.coverage+=1 if (action != 0) else 0

        #increments the accuracy if the reward is positive (we have a hit)
        self.accuracy+=1 if (reward >= 0 and action != 0) else 0
       
        
        #Increments the counter for short if the action is a short (id 2)
        self.short +=1 if(action == 2) else 0
        
        #Increments the counter for long if the action is a long (id 1)
        self.long +=1 if(action == 1) else 0
        
        #We will also calculate the accuracy for a given action. Here, it increments
        #the accuracy for short if the action is short and the reward is positive
        self.shortAcc +=1 if(action == 2 and reward >=0) else 0
        
        #Increments the accuracy for long if the action is long and the reward is positive
        self.longAcc +=1 if(action == 1 and reward >=0) else 0
        
        #If the market increases, increments the marketRise variable. If the prediction is 1 (long), increments the precision for long
        if(market>0):
            self.marketRise+=1
            self.longPrec+=1 if(action == 1) else 0

        #If market decreases, increments the marketFall. If the prediction is 2 (short), increments the precision for short   
        elif(market<0):
            self.marketFall+=1
            self.shortPrec+=1 if(action == 2) else 0

    #Function to show the metrics of the episode  
    def getInfo(self):
        #Start setting them to zero
        acc = 0
        cov = 0
        short = 0
        long = 0
        longAcc = 0
        shortAcc = 0
        longPrec = 0
        shortPrec = 0
        
        #If there is coverage, we will calculate the accuracy only related to when decisions were made. 
        #In other words, we dont calculate accuracy for hold operations
        if self.coverage > 0:
            acc = self.accuracy/self.coverage
        
        #Now, we calculate the mean coverage, short and long operations from the episodes
        if self.episodes > 0:
            cov = self.coverage/self.episodes
            short = self.short/self.episodes
            long = self.long/self.episodes

        #Calculate the mean accuracy for short operations. 
        #That is, the number of total short correctly predicted (self.shortAcc) 
        #divided by the total of shorts predicted (self.short)
        # #We need to correct this     
        if self.short > 0:
            shortAcc = self.shortAcc/self.short
        
        #Calculate the mean accuracy for long operations. 
        #That is, the number of total short correctly predicted (long.shortAcc) 
        #divided by the total of longs predicted (long.short)
        if self.long > 0:
            longAcc = self.longAcc/self.long


        if self.marketRise > 0:
            longPrec = self.longPrec/self.marketRise

        if self.marketFall > 0:
            shortPrec = self.shortPrec/self.marketFall

        #Returns the metrics to the user    
        return self.episodes,cov,acc,self.rewardSum,long,short,longAcc,shortAcc,longPrec,shortPrec
