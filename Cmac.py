import numpy as np
class Cmac:
    def __init__(self,g,a,o,i=0,d=0):
        self.generalization = g
        self.AssociationCell = np.zeros([a,1])
        self.outputs = np.zeros([o,1])
        self.weights = np.zeros([a*o,1]) if(i==0) else np.zeros([a*o,a*o])
        self.inputDimension = 1 if(i == 0) else 2
        self.discreteCmac = 0 if(d==0) else 1


    def activateOrDeactivateACells(self,x,v=0):
        for i in x:
            if(self.inputDimension == 1):
                address = self.__getAddress([i])
                for j in address:
                    self.AssociationCell[j] = v

            elif(self.inputDimension == 2):
                address = self.__getAddress(i)
                for j in address:
                    self.AssociationCell[j[0],j[1]]= v
        
    def __getAddress(self,x):
        l = []
        if(len(x)!=self.inputDimension):
            print("Input must be of specified dimension")
            return l

        for i in range(self.generalization):
            if(self.inputDimension == 1):
                l.append(x[0]+i)
            elif(self.inputDimension == 2):
                l.append([x[0]+i,x[1]+i])

        return l

    def getACell(self):
        return self.AssociationCell

    def __modifyWeights(self,x):
        pass

    def train(self,x,y):
        pass




# a = Cmac(4,35,1,0)

# x = [2, 7]
# a.activateOrDeactivateACells(x,1)
# h =np.copy(a.getACell())
# a.activateOrDeactivateACells(x,0)

# x = [1]
# a.activateOrDeactivateACells(x,3)
# g =np.copy(a.getACell())
# a.activateOrDeactivateACells(x,0)

# x = [0]
# a.activateOrDeactivateACells(x,4)
# q = np.copy(a.getACell())
# a.activateOrDeactivateACells(x,0)
# print(np.c_[h,g,q])

