import numpy as np
import matplotlib.pyplot as plt

x = np.random.uniform(0,10,(70,1))
y = np.sin(x)

trainData = np.column_stack((x,y))

x = np.random.uniform(0,10,(30,1))
y = np.sin(x)


# Train and test cmac code goes here

testData = np.column_stack((x,y))

#


fig1, ax1 = plt.subplots()
ax1.plot(trainData[:,0],trainData[:,1],'ro',label = 'Training Data')
ax1.set_title("CMAC Results")
ax1.set_xlabel("X-label for axis 1")
ax1.set_ylabel("sin(x)")



ax1.plot(testData[:,0],testData[:,1],'go',label = 'Test Data')
plt.legend(loc = "upper right")
plt.show()
