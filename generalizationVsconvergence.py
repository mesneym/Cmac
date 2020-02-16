############################################
# This file explores the relationship 
# between Time to convergence and 
# Generalization number,g
###########################################

from Cmac import Cmac
import numpy as np
import matplotlib.pyplot as plt

###########################################
#       Training and Test Data
###########################################
x = np.random.uniform(0,10,(70,1))
y = np.sin(x)

trainData = np.column_stack((x,y))

x = np.random.uniform(0,10,(30,1))
y = np.sin(x)

testDataExpected = np.column_stack((x,y))


#############################################
#   Generalization vs Time to Convergence
#############################################

for g in range(3,25):
    a = Cmac(g,35,35)
    accuracyTableD = a.train(trainData[:,0],trainData[:,1])
    testDataResults = np.zeros((len(x),2))
    for i in range(len(x)):
        y = a.prediction(x[i])
        testDataResults[i]=[x[i],y]

    idx = testDataResults[:,0].argsort()
    testDataResults = testDataResults[idx]
    
    #########################################
    ## plotting graph
    #########################################

    ###########################
    # Accuracy Table Discrete
    ###########################
    fig2, ax2 = plt.subplots()
    ax2.plot(accuracyTableD[:,0],accuracyTableD[:,1],'-o')
    ax2.set_title("Accuracy of Training(generalization = {})".format(g))
    ax2.set_ylabel("Percentage wrong")
    ax2.set_xlabel("Num of training samples")
    # plt.savefig("/home/ak/Cmac/Data/ConvergenceVsGeneralization/Acc {}.png".format(g))
    plt.show()




