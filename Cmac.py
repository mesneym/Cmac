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
        # if(x1 == x2):
            # return 1
        # return 1/((x1-x2)**2)
        # return 1/abs(x1-x2)
        return 1/np.exp((x1-x2)**2/4)

    def __quantizeInput(self,x,xmin =0, xmax = 10):
        if(x<min(x,xmin) or x> max(x,xmax)):
            raise Exception('Input must be within specified range')

        quantization = int(np.floor(self.resolution*(x-xmin)/(xmax-xmin)) + 
                       np.floor(self.generalization/2))
        low = int(quantization - np.floor(self.generalization/2))
        high= int(quantization + np.floor(self.generalization/2))

        return [low,high,quantization]


    def prediction(self,x,xmin=0,xmax=10,d = 0):
        neighborOfx = [0,0]
        neighborOfx[0],neighborOfx[1],quantization = self.__quantizeInput(x)

        result = 0
        if(d == 0):
            result = self.weights[neighborOfx[0]:neighborOfx[1]+1].sum()
        else:
            if(neighborOfx[1]+1 < len(self.weights)):
                result = 0.5*self.weights[neighborOfx[1]+1]  \
                         +0.5*self.weights[neighborOfx[0]] \
                         +self.weights[neighborOfx[0]+1:neighborOfx[1]+1].sum()
            else:
                 result = self.weights[neighborOfx[0]:neighborOfx[1]+1].sum()
        return result


    def __updateWeights(self,x,learningRate,error,xmin,xmax,d):
        neighborOfx = [0,0]
        neighborOfx[0],neighborOfx[1],quantization = self.__quantizeInput(x)

        if(d == 0): # Discrete Cmac
            for i in range(neighborOfx[0],neighborOfx[1]+1):
                self.weights[i] += learningRate*error*self.__kernel(quantization,i)
        
        else:    # continuous Cmac
            for i in range(neighborOfx[0],neighborOfx[1]+2):
                if(i == neighborOfx[0]):
                    self.weights[i] += learningRate*error*self.__kernel(quantization,i)*0.5
                elif(i< len(self.weights)):
                    if(i == neighborOfx[1]+1):
                        self.weights[i] += learningRate*error*self.__kernel(quantization,i)*0.5
                    else:
                        self.weights[i] += learningRate*error*self.__kernel(quantization,i)



    def train(self,x,y,learningRate=0.01,iterations=100,accuracy=0.01,d = 0, xmin=0 ,xmax =10):
        accuracyTable = np.zeros((len(x),2))
        for j in range(iterations):
            for i in range(len(x)):
                error = y[i]-self.prediction(x[i],xmin,xmax,d) 
                # print("prediction is {}".format(self.prediction(x[i],xmin,xmax)))
                # print("error is{}".format(error))
                # print("==========")
                # print(" ")
                if(abs(error)<= accuracy):
                    continue
                self.__updateWeights(x[i],learningRate,error, xmin,xmax,d)

            count = 0
            for k in range(len(x)):
                error = y[k] - self.prediction(x[k],xmin,xmax,d)
                if(error >= 0.01):
                    count += 1
            accuracyTable[i] = [i,count/len(x)*100]
        return accuracyTable

                
        
