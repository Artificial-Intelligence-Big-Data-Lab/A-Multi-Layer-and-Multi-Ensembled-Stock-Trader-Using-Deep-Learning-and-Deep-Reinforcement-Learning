from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
from math import floor
from ensemble import ensemble

from matplotlib.gridspec import GridSpec

outputFile=str(sys.argv[2])+".pdf"
numFiles=int(sys.argv[3])
#Number of epochs in the algorithm
numEpochs=35
numPlots=10

pdf=PdfPages(outputFile)

#Configure the size of the picture that will be plotted
#Configure the size of the picture that will be plotted
plt.figure(figsize=((numEpochs/10)*(2),9*5))

#Open the file that was saved on folder csv/walks, containing information about each iteration in that walk 
#Lets show a summary of each walk
#For each walk, one column is plotted in a final pdf file
for i in range(1,numFiles+1):

    document = pd.read_csv("./Output/csv/"+ sys.argv[1]+"/results-agent-training.csv")
    plt.subplot(numPlots,numFiles,0*numFiles + i)
    #Draw information in that file. First of all, lets plot accuracy
    plt.plot(document.ix[:, 'Iteration'].tolist(),document.ix[:, 'testAccuracy'].tolist(),'r',label='Test')
    plt.plot(document.ix[:, 'Iteration'].tolist(),document.ix[:, 'trainAccuracy'].tolist(),'b',label='Train')
    plt.plot(document.ix[:, 'Iteration'].tolist(),document.ix[:, 'validationAccuracy'].tolist(),'g',label='Validation')
    plt.xticks(range(0,numEpochs,4))
    plt.yticks(np.arange(0, 1, step=0.1))
    plt.ylim(-0.05,1.05)
    plt.axhline(y=0, color='k', linestyle='-')
    plt.legend()
    plt.grid()
    plt.title('Accuracy')

    #Lets draw information about coverage, read from the csv file located at csv/walks
    plt.subplot(numPlots,numFiles,1*numFiles + i)
    plt.plot(document.ix[:, 'Iteration'].tolist(),document.ix[:, 'testCoverage'].tolist(),'r',label='Test')
    plt.plot(document.ix[:, 'Iteration'].tolist(),document.ix[:, 'trainCoverage'].tolist(),'b',label='Train')
    plt.plot(document.ix[:, 'Iteration'].tolist(),document.ix[:, 'validationCoverage'].tolist(),'g',label='Validation')
    plt.xticks(range(0,numEpochs,4))
    plt.yticks(np.arange(0, 1, step=0.1))
    plt.ylim(-0.05,1.05)
    plt.axhline(y=0, color='k', linestyle='-')
    plt.legend()
    plt.grid()
    plt.title('Coverage')

    # Information about reward
    plt.subplot(numPlots,numFiles,2*numFiles + i )
    plt.plot(document.ix[:, 'Iteration'].tolist(),document.ix[:, 'trainReward'].tolist(),'b',label='Train')
    plt.plot(document.ix[:, 'Iteration'].tolist(),document.ix[:, 'validationReward'].tolist(),'g',label='Validation')
    plt.plot(document.ix[:, 'Iteration'].tolist(),document.ix[:, 'testReward'].tolist(),'r',label='Test')
    plt.xticks(range(0,numEpochs,4))
    plt.axhline(y=0, color='k', linestyle='-')
    plt.legend()
    plt.grid()
    plt.title('Reward')
    
    #Percentages of long
    plt.subplot(numPlots,numFiles,3*numFiles + i )
    plt.plot(document.ix[:, 'Iteration'].tolist(),document.ix[:, 'trainLong%'].tolist(),'b',label='Train')
    plt.plot(document.ix[:, 'Iteration'].tolist(),document.ix[:, 'validationLong%'].tolist(),'g',label='Validation')
    plt.plot(document.ix[:, 'Iteration'].tolist(),document.ix[:, 'testLong%'].tolist(),'r',label='Test')  
    plt.xticks(range(0,numEpochs,4))
    plt.yticks(np.arange(0, 1, step=0.1))    
    plt.ylim(-0.05,1.05)
    plt.axhline(y=0, color='k', linestyle='-')
    plt.legend()
    plt.grid()
    plt.title('Long %')
    
    #Percentages of short
    plt.subplot(numPlots,numFiles,4*numFiles + i )
    plt.plot(document.ix[:, 'Iteration'].tolist(),document.ix[:, 'trainShort%'].tolist(),'b',label='Train')
    plt.plot(document.ix[:, 'Iteration'].tolist(),document.ix[:, 'validationShort%'].tolist(),'g',label='Validation')
    plt.plot(document.ix[:, 'Iteration'].tolist(),document.ix[:, 'testShort%'].tolist(),'r',label='Test')
    plt.xticks(range(0,numEpochs,4))
    plt.yticks(np.arange(0, 1, step=0.1))
    plt.ylim(-0.05,1.05)
    plt.axhline(y=0, color='k', linestyle='-')
    plt.legend()
    plt.grid()
    plt.title('Short %')
    

    #Coverage
    plt.subplot(numPlots,numFiles,5*numFiles + i )
    plt.plot(document.ix[:, 'Iteration'].tolist(),list(map(lambda x: 1-x,document.ix[:, 'trainCoverage'].tolist())),'b',label='Train')
    plt.plot(document.ix[:, 'Iteration'].tolist(),list(map(lambda x: 1-x,document.ix[:, 'validationCoverage'].tolist())),'g',label='Validation')
    plt.plot(document.ix[:, 'Iteration'].tolist(),list(map(lambda x: 1-x,document.ix[:, 'testCoverage'].tolist())),'r',label='Test')
    plt.xticks(range(0,numEpochs,4))
    plt.yticks(np.arange(0, 1, step=0.1))
    plt.ylim(-0.05,1.05)
    plt.axhline(y=0, color='k', linestyle='-')
    plt.legend()
    plt.grid()
    plt.title('Hold %')
    

    #Accuracy of longs
    plt.subplot(numPlots,numFiles,6*numFiles + i )
    plt.plot(document.ix[:, 'Iteration'].tolist(),document.ix[:, 'trainLongAcc'].tolist(),'b',label='Train')
    plt.plot(document.ix[:, 'Iteration'].tolist(),document.ix[:, 'validationLongAcc'].tolist(),'g',label='Validation')
    plt.plot(document.ix[:, 'Iteration'].tolist(),document.ix[:, 'testLongAcc'].tolist(),'r',label='Test')
    plt.xticks(range(0,numEpochs,4))
    plt.yticks(np.arange(0, 1, step=0.1))
    plt.ylim(-0.05,1.05)
    plt.axhline(y=0, color='k', linestyle='-')
    plt.legend()
    plt.grid()
    plt.title('Long Accuracy')
    
    #Accuracy of shorts
    plt.subplot(numPlots,numFiles,7*numFiles + i )
    plt.plot(document.ix[:, 'Iteration'].tolist(),document.ix[:, 'trainShortAcc'].tolist(),'b',label='Train')
    plt.plot(document.ix[:, 'Iteration'].tolist(),document.ix[:, 'validationShortAcc'].tolist(),'g',label='Validation')
    plt.plot(document.ix[:, 'Iteration'].tolist(),document.ix[:, 'testShortAcc'].tolist(),'r',label='Test')
    plt.xticks(range(0,numEpochs,4))
    plt.yticks(np.arange(0, 1, step=0.1))
    plt.ylim(-0.05,1.05)
    plt.axhline(y=0, color='k', linestyle='-')
    plt.legend()
    plt.grid()
    plt.title('Short Accuracy')

    
    #Precisions of long
    plt.subplot(numPlots,numFiles,8*numFiles + i )
    plt.plot(document.ix[:, 'Iteration'].tolist(),document.ix[:, 'trainLongPrec'].tolist(),'b',label='Train')
    plt.plot(document.ix[:, 'Iteration'].tolist(),document.ix[:, 'validLongPrec'].tolist(),'g',label='Validation')
    plt.plot(document.ix[:, 'Iteration'].tolist(),document.ix[:, 'testLongPrec'].tolist(),'r',label='Test')
    plt.xticks(range(0,numEpochs,4))
    plt.yticks(np.arange(0, 1, step=0.1))
    plt.ylim(-0.05,1.05)
    plt.axhline(y=0, color='k', linestyle='-')
    plt.legend()
    plt.grid()
    plt.title('Long Precision')
    
    #Precisions of short
    plt.subplot(numPlots,numFiles,9*numFiles + i )
    plt.plot(document.ix[:, 'Iteration'].tolist(),document.ix[:, 'trainShortPrec'].tolist(),'b',label='Train')
    plt.plot(document.ix[:, 'Iteration'].tolist(),document.ix[:, 'validShortPrec'].tolist(),'g',label='Validation')
    plt.plot(document.ix[:, 'Iteration'].tolist(),document.ix[:, 'testShortPrec'].tolist(),'r',label='Test')
    plt.xticks(range(0,numEpochs,4))
    plt.yticks(np.arange(0, 1, step=0.1))
    plt.ylim(-0.05,1.05)
    plt.axhline(y=0, color='k', linestyle='-')
    plt.legend()
    plt.grid()
    plt.title('Short Precision')

plt.suptitle("Experiment RL metalearner\n"
            +"Model: 35 neurons single layer\n"
            +"Input: 1000 predictions of CNNs\n"
            +"Memory-Window Length: 10000-1\n"
            +"Other changes: Does Short, Hold and Long\n"
            +"Explorations:" +sys.argv[4] +"."
            ,size=19    
            ,weight=20
            ,ha='left'
            ,x=0.1
            ,y=0.99)

pdf.savefig()


#Now, lets try the ensemble
i=1

###########-------------------------------------------------------------------|Tabella Full Ensemble|-------------------
x=2
y=1
plt.figure(figsize=(x*3.5,y*3.5))

plt.subplot(y,y,1)
plt.axis('off')
val,col=ensemble("test", sys.argv[1])
t=plt.table(cellText=val, colLabels=col, fontsize=20, loc='center')
t.auto_set_font_size(False)
t.set_fontsize(6)
plt.title("Final Results")
#plt.suptitle("MAJORITY VOTING")
pdf.savefig()
###########--------------------------------------------------------------------------------------------------------------------
pdf.close()



