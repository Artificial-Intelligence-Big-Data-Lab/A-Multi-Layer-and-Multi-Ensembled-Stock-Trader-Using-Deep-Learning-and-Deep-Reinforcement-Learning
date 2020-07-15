#Imports the SPEnv library, which will perform the Agent actions themselves
from SpEnv import SpEnv

#Callback used to print the results at each episode
from Callback import ValidationCallback

#Keras library for the NN considered
from keras.models import Sequential

#Keras libraries for layers, activations and optimizers used
from keras.layers import Dense, Activation, Flatten
from keras.layers.advanced_activations import LeakyReLU, PReLU
from keras.optimizers import *

#RL Agent 
from rl.agents.dqn import DQNAgent
from rl.memory import SequentialMemory
from rl.policy import EpsGreedyQPolicy
from keras_radam import RAdam

#Mathematical operations used later
from math import floor

#Library to manipulate the dataset in a csv file
import pandas as pd

#Library used to manipulate time
import datetime
import os

import numpy
numpy.random.seed(0)

class DeepQTrading:
    
    #Class constructor
    #model: Keras model considered
    #explorations_iterations: a vector containing (i) probability of random predictions; (ii) how many iterations will be 
    #run by the algorithm (we run the algorithm several times-several iterations)  
    #outputFile: name of the file to print metrics of the training
    #ensembleFolderName: name of the file to print predictions
    #optimizer: optimizer to run 
        
    def __init__(self, model, nbActions, explorations_iterations, outputFile, ensembleFolderName, optimizer="adamax"):
        
        self.ensembleFolderName=ensembleFolderName
        self.policy = EpsGreedyQPolicy()
        self.explorations_iterations=explorations_iterations
        self.nbActions=nbActions
        self.model=model
        #Define the memory
        self.memory = SequentialMemory(limit=10000, window_length=1)
        #Instantiate the agent with parameters received
        self.agent = DQNAgent(model=self.model, policy=self.policy,  nb_actions=self.nbActions, memory=self.memory, nb_steps_warmup=200, target_model_update=1e-1, enable_double_dqn=True,enable_dueling_network=True)
        
        #Compile the agent with the optimizer given as parameter
        if optimizer=="adamax":        
                self.agent.compile(Adamax(), metrics=['mae'])
        if optimizer=="adadelta":        
                self.agent.compile(Adadelta(), metrics=['mae'])
        if optimizer=="sgd":        
                self.agent.compile(SGD(), metrics=['mae'])
        if optimizer=="rmsprop":        
                self.agent.compile(RMSprop(), metrics=['mae'])
        if optimizer=="nadam":        
                self.agent.compile(Nadam(), metrics=['mae'])
        if optimizer=="adagrad":        
                self.agent.compile(Adagrad(), metrics=['mae'])
        if optimizer=="adam":        
                self.agent.compile(Adam(), metrics=['mae'])
        if optimizer=="radam":        
                self.agent.compile(RAdam(total_steps=5000, warmup_proportion=0.1, min_lr=1e-5), metrics=['mae'])

        #Save the weights of the agents in the q.weights file
        #Save random weights
        self.agent.save_weights("q.weights", overwrite=True)

        #Load data
        self.train_data= pd.read_csv('./dataset/jpm/train_data.csv')
        self.validation_data=pd.read_csv('./dataset/jpm/train_data.csv')
        self.test_data=pd.read_csv('./dataset/jpm/test_data.csv')
                
        #Call the callback for training, validation and test in order to show results for each iteration 
        self.trainer=ValidationCallback()
        self.validator=ValidationCallback()
        self.tester=ValidationCallback()
        self.outputFileName=outputFile

    def run(self):
        #Initiates the environments, 
        trainEnv=validEnv=testEnv=" "
         
        if not os.path.exists(self.outputFileName):
             os.makedirs(self.outputFileName)

        file_name=self.outputFileName+"/results-agent-training.csv"
        
        self.outputFile=open(file_name, "w+")
        #write the first row of the csv
        self.outputFile.write(
            "Iteration,"+
            "trainAccuracy,"+
            "trainCoverage,"+
            "trainReward,"+
            "trainLong%,"+
            "trainShort%,"+
            "trainLongAcc,"+
            "trainShortAcc,"+
            "trainLongPrec,"+
            "trainShortPrec,"+

            "validationAccuracy,"+
            "validationCoverage,"+
            "validationReward,"+
            "validationLong%,"+
            "validationShort%,"+
            "validationLongAcc,"+
            "validationShortAcc,"+
            "validLongPrec,"+
            "validShortPrec,"+
                
            "testAccuracy,"+
            "testCoverage,"+
            "testReward,"+
            "testLong%,"+
            "testShort%,"+
            "testLongAcc,"+
            "testShortAcc,"+
            "testLongPrec,"+
            "testShortPrec\n")      
        
            
        #Prepare the training and validation files for saving them later 
        ensambleValid=pd.DataFrame(index=self.validation_data[:].ix[:,'date_time'].drop_duplicates().tolist())
        ensambleTest=pd.DataFrame(index=self.test_data[:].ix[:,'date_time'].drop_duplicates().tolist())
            
        #Put the name of the index for validation and testing
        ensambleValid.index.name='date_time'
        ensambleTest.index.name='date_time'
            
        #Explorations are epochs considered, or how many times the agent will play the game.  
        for eps in self.explorations_iterations:

            #policy will use eps[0] (explorations), so the randomness of predictions (actions) will happen with eps[0] of probability 
            self.policy.eps = eps[0]
                
            #there will be 25 iterations or eps[1] in explorations_iterations)
            for i in range(0,eps[1]):
                    
                del(trainEnv)
                #Define the training, validation and testing environments with their respective callbacks
                trainEnv = SpEnv(data=self.train_data, callback=self.trainer)
                
                del(validEnv)
                validEnv=SpEnv(data=self.validation_data,ensamble=ensambleValid,callback=self.validator,columnName="iteration"+str(i))
                
                del(testEnv)  
                testEnv=SpEnv(data=self.test_data, callback=self.tester,ensamble=ensambleTest,columnName="iteration"+str(i))

                #Reset the callback
                self.trainer.reset()
                self.validator.reset()
                self.tester.reset()

                #Reset the training environment
                trainEnv.resetEnv()
                
                #Train the agent
                #The agent receives as input one environment
                self.agent.fit(trainEnv,nb_steps=len(self.train_data),visualize=False,verbose=0)
                
                #Get the info from the train callback    
                (_,trainCoverage,trainAccuracy,trainReward,trainLongPerc,                                              trainShortPerc,trainLongAcc,trainShortAcc,trainLongPrec,trainShortPrec)=self.trainer.getInfo()
                
                print("Iteration " + str(i+1) + " TRAIN:  accuracy: " + str(trainAccuracy)+ " coverage: " + str(trainCoverage)+ " reward: " + str(trainReward))
                             
                #Reset the validation environment
                validEnv.resetEnv()               
                #Test the agent on validation data
                self.agent.test(validEnv,nb_episodes=len(self.validation_data),visualize=False,verbose=0)
                
                #Get the info from the validation callback
                (_,validCoverage,validAccuracy,validReward,validLongPerc,validShortPerc,
validLongAcc,validShortAcc,validLongPrec,validShortPrec)=self.validator.getInfo()
                #Print callback values on the screen
                print("Iteration " +str(i+1) + " VALIDATION:  accuracy: " + str(validAccuracy)+ " coverage: " + str(validCoverage)+ " reward: " + str(validReward))

                #Reset the testing environment
                testEnv.resetEnv()
                #Test the agent on testing data
                self.agent.test(testEnv,nb_episodes=len(self.test_data),visualize=False,verbose=0)
                #Get the info from the testing callback
                (_,testCoverage,testAccuracy,testReward,testLongPerc,testShortPerc,
testLongAcc,testShortAcc,testLongPrec,testShortPrec)=self.tester.getInfo()
                #Print callback values on the screen
                print("Iteration " +str(i+1) + " TEST:  acc: " + str(testAccuracy)+ " cov: " + str(testCoverage)+ " rew: " + str(testReward))
                print(" ")
                    
                #write the metrics in a text file
                self.outputFile.write(
                    str(i)+","+
                    str(trainAccuracy)+","+
                    str(trainCoverage)+","+
                    str(trainReward)+","+
                    str(trainLongPerc)+","+
                    str(trainShortPerc)+","+
                    str(trainLongAcc)+","+
                    str(trainShortAcc)+","+
                    str(trainLongPrec)+","+
                    str(trainShortPrec)+","+
                       
                    str(validAccuracy)+","+
                    str(validCoverage)+","+
                    str(validReward)+","+
                    str(validLongPerc)+","+
                    str(validShortPerc)+","+
                    str(validLongAcc)+","+
                    str(validShortAcc)+","+
                    str(validLongPrec)+","+
                    str(validShortPrec)+","+
                       
                    str(testAccuracy)+","+
                    str(testCoverage)+","+
                    str(testReward)+","+
                    str(testLongPerc)+","+
                    str(testShortPerc)+","+
                    str(testLongAcc)+","+
                    str(testShortAcc)+","+
                    str(testLongPrec)+","+
                    str(testShortPrec)+"\n")

        #Close the file                
        self.outputFile.close()

        if not os.path.exists("./Output/ensemble/"+self.ensembleFolderName):
             os.makedirs("./Output/ensemble/"+self.ensembleFolderName)

        ensambleValid.to_csv("./Output/ensemble/"+self.ensembleFolderName+"/ensemble_valid.csv")
        ensambleTest.to_csv("./Output/ensemble/"+self.ensembleFolderName+"/ensemble_test.csv")


    #Function to end the Agent
    def end(self):
        print("FINISHED")
