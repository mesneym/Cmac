import numpy as np
import matplotlib.pyplot as plt

x = np.random.uniform(0,10,(70,1))
y = np.sin(x)

trainData = np.column_stack((x,y))

x = np.random.uniform(0,10,(30,1))
y = np.sin(x)

testDataExpected = np.column_stack((x,y))


# Train and test cmac code goes here
testDataResults = np.column_stack((x,y))
#



fig1, ax1 = plt.subplots()
ax1.plot(trainData[:,0],trainData[:,1],'ro',label = 'Training Data')
ax1.plot(testDataResults[:,0],testDataResults[:,1],'go',label = 'Expected Output from Test Data')
ax1.plot(testDataExpected[:,0],testDataExpected[:,1],'b^',label = 'Cmac Output from Test Data')

ax1.set_title("CMAC Results")
ax1.set_xlabel("X-label for axis 1")
ax1.set_ylabel("sin(x)")


plt.legend(loc = "upper right")
plt.show()
