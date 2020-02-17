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

#Training data
trainData = np.column_stack((x,y))

x = np.random.uniform(0,10,(30,1))
y = np.sin(x)

testDataExpected = np.column_stack((x,y))

#Sorting testDataExpected
idx = testDataExpected[:,0].argsort()
testDataExpected = testDataExpected[idx]

###########################################
#        Training and Testing Cmac
###########################################
g = 5   # generalization number
w = 35  # number of weights
ac = 35 # resolution(number of grids for
        # look up table) 

a = Cmac(g,w,ac)

##################################
#         Discrete
##################################
#train data with discrete cmac
a.train(trainData[:,0],trainData[:,1])
testDataResults = np.zeros((len(testDataExpected),2))

#######################
# Test Data Prediction
#######################
for i in range(len(testDataExpected)):
    y = a.prediction(testDataExpected[i,0])
    testDataResults[i]=[testDataExpected[i,0],y]

#sort testDataResults 
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

#sort testDataResults
idx = testDataResultsC[:,0].argsort()
testDataResultsC = testDataResultsC[idx]

#######################
# Calculating accuracy
#######################
count = 0.0
accuracyContinous = 0
for i in range(len(testDataExpected)):
    if(abs(testDataResultsC[i,1] - testDataExpected[i,1])<= 0.05):
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
strFile = "./Results/discreteAccuracy.png"
if os.path.isfile(strFile):
   os.remove(strFile)   

fig1.set_size_inches(8, 5)
plt.savefig('./Results/discreteAccuracy.png')

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
strFile = "./Results/continuousAccuracy.png"
if os.path.isfile(strFile):
   os.remove(strFile)   

fig3.set_size_inches(8, 5)
plt.savefig('./Results/continousAccuracy.png')
plt.show()


