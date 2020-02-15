from Cmac import Cmac
import numpy as np
import matplotlib.pyplot as plt


###########################################
#         True Signal
###########################################


###########################################
#            Training and Test Data
###########################################
x = np.random.uniform(0,10,(70,1))
y = np.sin(x)

trainData = np.column_stack((x,y))

x = np.random.uniform(0,10,(30,1))
y = np.sin(x)

testDataExpected = np.column_stack((x,y))

###########################################
#            Training Cmac
###########################################
a = Cmac(5,35,35)

###############
#  Discrete
###############
accuracyTableD = a.train(trainData[:,0],trainData[:,1])

testDataResults = np.zeros((len(x),2))

for i in range(len(x)):
    y = a.prediction(x[i])
    testDataResults[i]=[x[i],y]

idx = testDataResults[:,0].argsort()
testDataResults = testDataResults[idx]


###############
#  Continuous
###############
accuracyTableC = a.train(trainData[:,0],trainData[:,1],d=1)

testDataResultsC = np.zeros((len(x),2))

for i in range(len(x)):
    y = a.prediction(x[i])
    testDataResultsC[i]=[x[i],y]

idx = testDataResultsC[:,0].argsort()
testDataResultsC = testDataResultsC[idx]



#############################################
#            Plotting graphs
#############################################

###################
#  Discrete Cmac
###################
fig1, ax1 = plt.subplots()
ax1.plot(trainData[:,0],trainData[:,1],'ro',label = 'Training Data')
ax1.plot(testDataResults[:,0],testDataResults[:,1],'-o',label = 'Cmac Output from Test Data')
ax1.plot(testDataExpected[:,0],testDataExpected[:,1],'g^',label = 'Expected Output from Test Data')
ax1.set_title("Discrete CMAC Results")
ax1.set_xlabel("X-label for axis 1")
ax1.set_ylabel("sin(x)")
plt.legend(loc = "upper right")

###########################
# Accuracy Table Discrete
###########################
fig2, ax2 = plt.subplots()
ax2.plot(accuracyTableD[:,0],accuracyTableD[:,1],'-o')
ax2.set_title("Accuracy of Training")
ax2.set_ylabel("Percentage wrong")
ax2.set_xlabel("Num of training samples")

########################
# Continuous Cmac
########################
fig3, ax3 = plt.subplots()
ax3.plot(trainData[:,0],trainData[:,1],'ro',label = 'Training Data')
ax3.plot(testDataResultsC[:,0],testDataResultsC[:,1],'-o',label = 'Cmac Output from Test Data')
ax3.plot(testDataExpected[:,0],testDataExpected[:,1],'g^',label = 'Expected Output from Test Data')
ax3.set_title("Continuous CMAC Results")
ax3.set_xlabel("X-label for axis 1")
ax3.set_ylabel("sin(x)")
plt.legend(loc = "upper right")


#############################
# Accuracy Table Continuous
#############################
fig4, ax4 = plt.subplots()
ax4.plot(accuracyTableC[:,0],accuracyTableC[:,1],'-o')
ax4.set_title("Accuracy of Training")
ax4.set_ylabel("Percentage wrong")
ax4.set_xlabel("Num of training samples")
plt.show()

