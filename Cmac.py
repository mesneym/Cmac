import numpy as np
class Cmac:
    def __init__(self,g,a,res):
        self.generalization = g if(g%2 == 1) else g-1
        self.AssociationCells = np.zeros([a,1])

        #increasing weights by generalization number for
        #association cells at extremes 
        self.weights = np.zeros([a+g,1]) 

        self.resolution = res


    def __kernel(self,x1,x2):
        return 1/abs(x1-x2)
        # return 1/(x1-x2)**2
        # return 1/np.exp((x1-x2)**2/4)



    def prediction(self,x,xmin=0,xmax=10):
        if(x<min(x,xmin) or x> max(x,xmax)):
            raise Exception('Input must be within specified range')

        quantization = int(np.floor(self.resolution*(x-xmin)/(xmax-xmin)) + 
                       np.floor(self.generalization/2))
        quant = int(np.floor(self.resolution*(x-xmin)/(xmax-xmin)))
        low = int(quantization -np.floor(self.generalization/2))
        high= int(quantization +np.floor(self.generalization/2))
        neighborOfx = (low,high)

        # print('value of x is {}'.format(x))
        # print('neighbor of x is {}'.format(neighborOfx))
        # print('quantization of x is{}'.format(quantization))
        # print("===============")
        # print(" ")
        result = 0
        for i in range(neighborOfx[0],neighborOfx[1]+1):
            result += self.weights[i] 

        return result


    def __updateWeights(self,x,learningRate,error,xmin,xmax):
        quantization = int(np.floor(self.resolution*(x-xmin)/(xmax-xmin)) + 
                       np.floor(self.generalization/2))
        quant = int(np.floor(self.resolution*(x-xmin)/(xmax-xmin)))
        low = int(quantization -np.floor(self.generalization/2))
        high= int(quantization +np.floor(self.generalization/2))
        neighborOfx = (low,high)

        
        for i in range(neighborOfx[0],neighborOfx[1]+1):
            self.weights[i] -= learningRate*error*self.__kernel(quantization,self.weights[i])



    def train(self,x,y,learningRate=0.02,iterations=2000,accuracy=0.01,d = 0, xmin=0 ,xmax =10):
        for i in range(len(x)):
            for j in range(iterations):
                error = y[i] - self.prediction(x[i],xmin,xmax) 
                if(error < accuracy):
                    break
                self.__updateWeights(x[i],learningRate,error, xmin,xmax)
        print(self.weights)

                

        
# a = Cmac(5,35,35)
# x = np.array([0, 1, 2])
# x.reshape(len(x),1)
# y = np.cos(x) 

# a.train(x,y)
# print(a.prediction(x[2]))
