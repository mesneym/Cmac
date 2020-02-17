############################################
# This file explores the relationship 
# between Time to convergence and 
# Generalization number,g
###########################################

import os
from Cmac import Cmac
import numpy as np
import matplotlib.pyplot as plt

###########################################
#       Training and Test Data
###########################################
x = np.random.uniform(0,10,(70,1))
y = np.sin(x)

trainData = np.column_stack((x,y))

#############################################
#   Generalization vs Time to Convergence
#############################################

fig, ax = plt.subplots(nrows=4, ncols=4)

g = 3
for row in ax:
    for col in row:
        a = Cmac(g,35,35)
        accuracyTableD = np.array(a.train(trainData[:,0],trainData[:,1]))

        # plotting graph
        col.plot(accuracyTableD[:,0],accuracyTableD[:,1],'b-')
        col.set_title("Accuracy of Training(generalization = {})".format(g))
        col.set_ylabel("Percentage wrong")
        col.set_xlabel("Num of iterations")
        g += 1

fig.subplots_adjust(hspace=0.5)
fig.set_size_inches(20, 10)
# manager = plt.get_current_fig_manager()
# manager.resize(*manager.window.maxsize())

strFile = "/home/ak/Cmac/Results/convergenceVsgeneralization.png"
if os.path.isfile(strFile):
   os.remove(strFile)   

plt.savefig("/home/ak/Cmac/Results/convergenceVsgeneralization.png", bbox_inches='tight')
plt.show()

