#Environment used for spenv 
#gym is the library of videogames used by reinforcement learning
import gym
from gym import spaces
#Numpy is the library to deal with matrices
import numpy
#Pandas is the library used to deal with the CSV dataset
import pandas
#datetime is the library used to manipulate time and date
from datetime import datetime
#Library created by Tonio to merge data used as feature vectors
#from MergedDataStructure import MergedDataStructure
#Callback is the library used to show metrics 
import Callback


class SpEnv(gym.Env):
    #Just for the gym library. In a continuous environment, you can do infinite decisions. 
    #We dont want this because we have just three possible actions.
    continuous = False

    #Observation window is the time window regarding the "hourly" dataset 
    #ensemble variable tells to save or not the decisions at each walk

    def __init__(self, data, callback = None, ensamble = None, columnName = "iteration-1"):
        #Declare the episode as the first episode
        self.episode=1

        # opening the dataset      
        self.data=data

        #Load the data
        self.output=False

        #ensamble is the table of validation and testing
        #If its none, you will not save csvs of validation and testing    
        if(ensamble is not None): # managing the ensamble output (maybe in the wrong way)
            self.output=True
            self.ensamble=ensamble
            self.columnName = columnName
            
            #self.ensemble is a big table (before file writing) containing observations as lines and epochs as columns
            #each column will contain a decision for each epoch at each date. It is saved later.
            #We read this table later in order to make ensemble decisions at each epoch
            self.ensamble[self.columnName]=0

        #Declare low and high as vectors with -inf values 
        self.low = numpy.array([-numpy.inf])
        self.high = numpy.array([+numpy.inf])

        #Define the space of actions as 3
        #the action space is now 2 (hold and long)
        #self.action_spaces = space.Discrete(2) 
        self.action_space = gym.spaces.Box(low=numpy.array([0]),high= numpy.array([2]), dtype=numpy.int)
      
               
        #low and high are the minimun and maximum accepted values for this problem
        #Tonio used random values
        #We dont know what are the minimum and maximum values of Close-Open, so we put these values
        self.observation_space = spaces.Box(self.low, self.high, dtype=numpy.float32)

        self.currentObservation = 0
        #Defines that the environment is not done yet
        self.done = False
        #The limit is the number of open values in the dataset (could be any other value)
        self.limit = len(data)      
        
        #Initiates the values to be returned by the environment
        self.reward = None
        self.possibleGain = 0
        self.openValue = 0
        self.closeValue = 0
        self.callback=callback

    #This is the action that is done in the environment. 
    #Receives the action and returns the state, the reward and if its done 
    def step(self, action):
    
        #assert self.action_space.contains(action)

        #Initiates the reward, weeklist and daylist
        self.reward=0
    
        #Calculate the reward in percentage of growing/decreasing
        self.possibleGain = self.data.iloc[self.currentObservation]['delta_next_day']
        
        #Calculate the reward in percentage of growing/decreasing
        self.possibleGain = self.data.iloc[self.currentObservation]['delta_next_day']

        #If action is a LONG, calculate the reward
        #If action is a long, calculate the reward
        if(action == 1):
        #The reward must be subtracted by the cost of transaction
        #action=1
            self.reward = self.possibleGain

        #If action is a short, calculate the reward
        elif(action==2):
            self.reward = (-self.possibleGain)

        #If action is a hold, no reward
        elif(action==0):
            self.reward = 0
                   
        #Finish episode 
        self.done=True

        
        #Call the callback for the episode
        if(self.callback!=None and self.done):
            self.callback.on_episode_end(action,self.reward,self.possibleGain)
            
        #If its validation or test, save the outputs in the ensemble file that will be ensembled later    
        if(self.output):
            self.ensamble.at[self.data.iloc[self.currentObservation]['date_time'],self.columnName]=action
                   
        self.episode+=1   
        self.currentObservation+=1
        
        if(self.currentObservation>=self.limit):
            self.currentObservation=0
             
        #Return the state, reward and if its done or not
        return self.getObservation(), self.reward, self.done, {}
        
    #function done when the episode finishes
    #reset will prepare the next state (feature vector) and give it to the agent
    def reset(self):
    
        self.done = False
        self.reward = None
        self.possibleGain = 0
       
        return self.getObservation()
        

    def getObservation(self):

        predictionList = []
        predictionList=numpy.array(self.data.iloc[self.currentObservation]["prediction_0":"prediction_999"])
     
        
        return predictionList.ravel()
    
    def resetEnv(self):
        self.currentObservation=0
        self.episode=1
