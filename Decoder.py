import math
import numpy as np

class Decoder:
    
    def __init__ (self,H, channel_model, channel_parameters, num_iter):
        # code parameters
        self.n = H.shape[1]
        self.k = self.n - H.shape[0]
        self.H = H
        self.model = channel_model
        self.params = channel_parameters
        self.num_iter = num_iter
        
        # misc parameters
        self.ep = 1e-5
        
        # graph parameters
        self.num_VN = self.n
        self.num_CN = self.n-self.k
        
        # adjacency matrix for VN and CN
        self.CN = []
        self.VN = []
        self.construct_graph(H)
        
    def construct_graph(self,H):
        for i in range(self.num_CN):
            temp = []
            for j in range(self.num_VN):
                if H[i,j]==1:
                    temp.append(j)
            self.CN.append(temp)
            
        for i in range(self.num_VN):
            temp = []
            for j in range(self.num_CN):
                if H[j,i]==1:
                    temp.append(j)
            self.VN.append(temp)
    
    def print_graph(self,mode):
        if mode=="matrix":
            print(self.H)
        elif mode=="list":
            print('CN : ',self.CN)
            print('VN : ',self.VN)
        else:
            print('Invalid mode')
    
    def initialize_LLR(self,y):
        self.L = np.zeros(self.num_VN)
        self.L_tot = np.zeros(self.num_VN)
        
        for j in range(self.num_VN):
            if self.model=="bec":
                pass
            elif self.model=="bsc":
                p1 = 1-self.params[0] if y[j]==1 else self.params[0]
                p0 = 1-self.params[0] if y[j]==0 else self.params[0]
                llr = math.log(p0/p1)
                self.L[j] = llr
            elif self.model=="awgn":
                p0 = (1+math.exp(-2*y[j]/(self.params[0]**2)))**(-1)
                p1 = (1+math.exp(2*y[j]/(self.params[0]**2)))**(-1)
                llr = math.log(p0/p1)
                self.L[j] = llr
            else:
                raise Exception("Invalid model")
        
        # create message array of NaNs
        self.Lji = np.full((self.num_VN,self.num_CN), np.nan)
        self.Lij = np.full((self.num_CN,self.num_VN), np.nan)
        
        # initializing the 1st iteration of message passing j->i
        for i in range(self.num_CN):
            for j in range(self.num_VN):
                if self.H[i,j]==1:
                    self.Lji[j,i] = self.L[j]
                    
    def CN_update(self):
        for i in range(self.num_CN):
            prod = 1
            for j in self.CN[i]:
                prod = prod * math.tanh(0.5*self.Lji[j,i])
            for j in self.CN[i]:
                val = prod/(self.ep+math.tanh(0.5*self.Lji[j,i]))
                val = min(1-self.ep,val)
                val = max(-1+self.ep,val)
                self.Lij[i,j] = 2*math.atanh(val)
                
    def VN_update(self):
        for j in range(self.num_VN):
            total_llr = 0
            for i in self.VN[j]:
                total_llr+=self.Lij[i,j]
            for i in self.VN[j]:
                self.Lji[j,i] = self.L[j] + total_llr - self.Lij[i,j]
        
    def compute_LLR(self):
        for j in range(self.num_VN):
            tot = 0
            for i in self.VN[j]:
                tot+=self.Lij[i,j]
            self.L_tot[j] = self.L[j]+tot
        
    def stopping_criterion(self, iteration_number):
        v = np.array([1 if val<0 else 0 for val in self.L_tot])
        if iteration_number>=self.num_iter or (((np.dot(v,self.H.T)%2)==0).all()):
            return True
        return False
    
    def decode(self, y, iter_step=100,verbose="off"):
        self.initialize_LLR(y)
        
        for k in range(self.num_iter):
            if verbose=="on" and k%iter_step==0:
                print("Iteration ",k,end="\t")
                # print("LLRs : ",np.round(self.L,2),end=" ")
                print()
            self.CN_update()
            self.VN_update()
            self.compute_LLR()
            
            if self.stopping_criterion(k):
                break
        
        v = self.L<0
        return v
    
        