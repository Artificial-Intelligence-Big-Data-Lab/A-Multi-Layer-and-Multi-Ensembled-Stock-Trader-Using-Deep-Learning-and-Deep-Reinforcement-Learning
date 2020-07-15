"""""
        This is the code of Reinforcement learning applied on outputs of an ensemble of classifiers
        There is an ensemble of 1000 CNNs that will output predictions for each day
        Therefore, our RL metalearner will be applied on these 1000 outputs
        
        We call it as the following:
        
        python main.py <number_of_actions> <number_of_explorations> <activation> <output_file> <optimizer>
        
        ex: python3 main.py 3 0.3 selu teste-rmsprop-0.3-selu rmsprop

        where:
                <number_of_actions>: number of actions done by the agent. 
                <number_of_explorations>: in the RL training, this is the probability that the action taken is random or it obeys the Q-values found previously 
                <activation>: activation function of the double q-network layer we use as RL agent 
                <output_file>: where results will be written 
                <optimizer>: optimization approach of the RL network

       Authors: Anselmo Ferreira, Alessandro Sebastian Podda and Andrea Corriga
       
       Please dont hesitate to cite our Applied Intelligence paper when using it for your research ;-)  
  
"""""

#os library is used to define the GPU to be used by the code, needed only in cerain situations (Better not to use it, use only if the main gpu is Busy)
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID";
os.environ["CUDA_VISIBLE_DEVICES"]="0";

#This is the class call for the Agent which will perform the experiment
from DeepQTrading import DeepQTrading

#Date library to manipulate time in the source code
import datetime

#Keras library to define the NN to be used
from keras.models import Sequential

#Layers used in the NN considered
from keras.layers import Dense, Activation, Flatten
from keras.layers import advanced_activations

#Activation Layers used in the source code
from keras.layers.advanced_activations import LeakyReLU, PReLU, ReLU

#Optimizer used in the NN
from keras.optimizers import Adam

#Libraries used for the Agent considered
from rl.agents.dqn import DQNAgent
from rl.memory import SequentialMemory
from rl.policy import EpsGreedyQPolicy

#Library used for showing the exception in the case of error 
import sys

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.3
set_session(tf.Session(config=config))


#There are three actions possible in the stock market
#Hold(id 0): do nothing.
#Long(id 1): It predicts that the stock market value will raise at the end of the day. 
#So, the action performed in this case is buying at the beginning of the day and sell it at the end of the day (aka long).
#Short(id 2): It predicts that the stock market value will decrease at the end of the day.
#So, the action that must be done is selling at the beginning of the day and buy it at the end of the day (aka short). 

#This is a simple NN considered. It is composed of:
#One flatten layer to get 1000 dimensional vectors as input
#One dense layer with 35 neurons with a given activation
#One final Dense Layer with the number of actions considered and linear activation

model = Sequential()
model.add(Flatten(input_shape=(1,1000))) 
if(sys.argv[4]=="relu"):
    model.add(Dense(35,activation='relu'))    
if(sys.argv[4]=="sigmoid"):
    model.add(Dense(35,activation='sigmoid'))    
if(sys.argv[4]=="linear"):
    model.add(Dense(35,activation='linear'))
if(sys.argv[4]=="tanh"):
    model.add(Dense(35,activation='tanh'))
if(sys.argv[4]=="selu"):
    model.add(Dense(35,activation='selu'))
model.add(LeakyReLU(alpha=.001))
model.add(Dense(int(sys.argv[1])))
model.add(Activation('linear'))



#Define the DeepQTrading class with the following parameters:
#explorations: operations are random with a given probability, and 25 iterations.
#in this case, iterations parameter is used because the agent acts on daily basis, so its better to repeat the experiments several
#times. 
#outputFile: where the results will be written

#sys.argv[1]: number of actions
#sys.argv[2]: probability of performing explorations
#sys.argv[3]: initializer
#sys.argv[4]: folder name where experiments results will be written
#sys.argv[5]: optimizer

dqt = DeepQTrading(
    model=model,
    nbActions=int(sys.argv[1]),
    explorations_iterations=[(round(float(sys.argv[2])),25)],
    outputFile="./Output/csv/" + sys.argv[4],
    ensembleFolderName=sys.argv[4],
    optimizer=sys.argv[5]
    )

dqt.run()

dqt.end()


