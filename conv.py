import numpy as np

class Conv:

    def __init__(self,num_filters,filters,random):
        #Set the number of filters
        self.num_filters = num_filters

        #Set the numpy array of filters
        if random == False:
            self.filters = filters
        else:
            self.filters = np.random.randn(num_filters,3,3)/9

    #oversees convolution layer (convolution happens in conv1)
    def forward(self, input):
        self.last_input = input
        result = np.zeros((input.shape[0]-2,input.shape[1]-2,self.num_filters))
        
        for reg, i, j in self.iterate(input):
            result[i,j] = np.sum(reg*self.filters,axis=(1,2))

            '''MIGHT NOT NEED THIS/CONVOLUTION1 B/C OF ITERATE AND NP.SUM
            result[i,j] = self.convolution1(input,self.filters)'''
        
        #return feature map
        return result

    def iterate(self, img):
        y = img.shape[0]
        x = img.shape[1]

        for b in range(0,y-2):
            for a in range(0,x-2):
                reg = img[b:b+3,a:a+3]
                yield reg, b, a

    def backprop(self, dLoss_dOut,learn):
        dLoss_dFilters = np.zeros(self.filters.shape)

        for reg, y, x in self.iterate(self.last_input):
            for fil in range(0,self.num_filters):
                dLoss_dFilters[fil] += dLoss_dOut[y,x,fil]*reg

        self.filters-=learn*dLoss_dFilters

        return self.filters