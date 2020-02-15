from Cmac import Cmac
import numpy as np
import matplotlib.pyplot as plt


###########################################
#         True Signal
###########################################


###########################################
#       Training and Test Data
###########################################
x = np.random.uniform(0,10,(70,1))
y = np.sin(x)

trainData = np.column_stack((x,y))

x = np.random.uniform(0,10,(30,1))
y = np.sin(x)

testDataExpected = np.column_stack((x,y))


###########################################
#          Training Cmac
###########################################
a = Cmac(5,35,35)

###############
#  Discrete
###############
a.train(x,y)
testDataResults = np.zeros((len(x),2))

for i in range(len(x)):
    y = a.prediction(x[i])
    testDataResults[i]=[x[i],y]


#############################################
#            Plotting graphs
#############################################
fig1, ax1 = plt.subplots()
ax1.plot(trainData[:,0],trainData[:,1],'ro',label = 'Training Data')
ax1.plot(testDataResults[:,0],testDataResults[:,1],'go',label = 'Cmac Output from Test Data')
ax1.plot(testDataExpected[:,0],testDataExpected[:,1],'b^',label = 'Expected Output from Test Data')

ax1.set_title("CMAC Results")
ax1.set_xlabel("X-label for axis 1")
ax1.set_ylabel("sin(x)")

plt.legend(loc = "upper right")
plt.show()
