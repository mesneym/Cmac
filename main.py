import os
from Cmac import Cmac
import numpy as np
import matplotlib.pyplot as plt


###########################################
#         True Signal
###########################################
Tx = np.arange(0,10,0.1)
Ty = np.sin(Tx)

###########################################
#            Training and Test Data
###########################################
x = np.random.uniform(0,10,(70,1))
y = np.sin(x)

trainData = np.column_stack((x,y))

x = np.random.uniform(0,10,(30,1))
y = np.sin(x)

testDataExpected = np.column_stack((x,y))

idx = testDataExpected[:,0].argsort()
testDataExpected = testDataExpected[idx]

###########################################
#        Training and Testing Cmac
###########################################
g = 5
w = 35
ac = 35
a = Cmac(g,w,ac)

##################################
#         Discrete
##################################
a.train(trainData[:,0],trainData[:,1])
testDataResults = np.zeros((len(testDataExpected),2))

#######################
# Test Data Prediction
#######################
for i in range(len(testDataExpected)):
    y = a.prediction(testDataExpected[i,0])
    testDataResults[i]=[testDataExpected[i,0],y]

idx = testDataResults[:,0].argsort()
testDataResults = testDataResults[idx]

#######################
# Calculating accuracy
#######################
count = 0.0
accuracyDiscrete = 0
for i in range(len(testDataExpected)):
    if(abs(testDataResults[i,1] - testDataExpected[i,1])<= 0.1):
        count += 1

accuracyDiscrete = count/len(testDataExpected) * 100


###################################
#           Continuous
###################################
# set d  to 1 to indicate continuous cmac
a.train(trainData[:,0],trainData[:,1],d=1)
testDataResultsC = np.zeros((len(testDataExpected),2))

#######################
# Test Data Prediction
#######################
for i in range(len(testDataExpected)):
    y = a.prediction(testDataExpected[i,0])
    testDataResultsC[i]=[testDataExpected[i,0],y]

idx = testDataResultsC[:,0].argsort()
testDataResultsC = testDataResultsC[idx]

#######################
# Calculating accuracy
#######################
count = 0.0
accuracyContinous = 0
for i in range(len(testDataExpected)):
    if(abs(testDataResults[i,1] - testDataExpected[i,1])<= 0.1):
        count += 1
accuracyContinous = count/len(testDataExpected) * 100


#############################################
#            Plotting graphs
#############################################

###########################
# Discrete cmac 
###########################
fig1, ax1 = plt.subplots()
ax1.plot(Tx,Ty,'r-',label = 'Signal')
ax1.plot(testDataResults[:,0],testDataResults[:,1],'-o',label = 'Cmac Output from Test Data')
ax1.plot(testDataExpected[:,0],testDataExpected[:,1],'g^',label = 'Expected Output from Test Data')
ax1.set_title("Discrete CMAC(generalization= {})".format(g))
ax1.set_xlabel("X-label for axis 1")
ax1.set_ylabel("sin(x)")
plt.legend(loc = "upper right")
plt.text(3,0.5, 'accuracy of test data =  {}'.format(np.round(accuracyDiscrete,2)))
strFile = "./Data/Accuracy/discreteAccuracy.png"
if os.path.isfile(strFile):
   os.remove(strFile)   
plt.savefig('./Data/Accuracy/discreteAccuracy.png')

########################
# Continuous Cmac
########################
fig3, ax3 = plt.subplots()
ax3.plot(Tx,Ty,'r-',label = 'Signal')
ax3.plot(testDataResultsC[:,0],testDataResultsC[:,1],'-o',label = 'Cmac Output from Test Data')
ax3.plot(testDataExpected[:,0],testDataExpected[:,1],'g^',label = 'Expected Output from Test Data')
ax3.set_title("Continuous CMAC (generalization=%i)" %g)
ax3.set_xlabel("X-label for axis 1")
ax3.set_ylabel("sin(x)")
plt.legend(loc = "upper right")
plt.text(3,0.5, 'accuracy of test data =  {}'.format(np.round(accuracyContinous,2)))
strFile = "./Data/Accuracy/continuousAccuracy.png"
if os.path.isfile(strFile):
   os.remove(strFile)   
plt.savefig('./Data/Accuracy/continousAccuracy.png')
plt.show()


