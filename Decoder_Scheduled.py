import math
import numpy as np

class Decoder_Scheduled:
    
    def __init__ (self,H, channel_model, channel_parameters, num_iter):
        # code parameters
        self.n = H.shape[1]
        self.k = self.n - H.shape[0]
        self.H = H
        self.model = channel_model
        self.params = channel_parameters
        self.num_iter = num_iter
        
        # misc parameters (epsilon : for numerical stability)
        self.ep = 1e-5
        
        # graph parameters
        self.num_VN = self.n
        self.num_CN = self.n-self.k
        
        # adjacency list for VN and CN
        self.CN = []
        self.VN = []
        self.construct_graph(H)
        
    # build the adjacency list for Tanner Graph
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
    
    # print the tanner graph
    def print_graph(self,mode):
        if mode=="matrix":
            print(self.H)
        elif mode=="list":
            print('CN : ',self.CN)
            print('VN : ',self.VN)
        else:
            print('Invalid mode')
    
    # initializing the apriori LLRs (based on the channel and noise)
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
        self.Lji = np.full((self.num_VN,self.num_CN), 0.0)
        self.Lij = np.full((self.num_CN,self.num_VN), np.nan)
        
        
        # initializing the 1st iteration of message passing j->i
        for i in range(self.num_CN):
            for j in range(self.num_VN):
                if self.H[i,j]==1:
#                     print(f"Assigning Lji[{j},{i}]={self.L[j]}")
                    self.Lji[j,i] = self.L[j]
#                     print(f"Lji[{j},{i}]={self.Lji[j,i]}")
    
    # initialize clusters of CNs
    def initialize_clusters(self,cluster_size):
        self.num_clusters = math.ceil(self.num_CN/cluster_size)
        self.clusters = []
        for i in np.arange(0,self.num_CN,cluster_size):
            temp = []
            for j in range(i,min(i+cluster_size,self.num_CN)):
                temp.append(j)
            self.clusters.append(temp)   
                    
    # CN update for cluster a                
    def CN_update(self,a):
        for i in self.clusters[a]:
            prod = 1
            for j in self.CN[i]:
                prod = prod * math.tanh(0.5*self.Lji[j,i])
#                 print(f"multiplying by {math.tanh(0.5*self.Lji[j,i])} for Lji[{j},{i}]={self.Lji[j,i]}")
#             print("prod : ",prod)
            for j in self.CN[i]:
                val = prod/(self.ep+math.tanh(0.5*self.Lji[j,i]))
                val = min(1-self.ep,val)
                val = max(-1+self.ep,val)
                self.Lij[i,j] = 2*math.atanh(val)
#                 print(f"assigning Lij[{i},{j}] = ",self.Lij[i,j])
                
    # VN update for cluster a
    def VN_update(self,a):
        vns = set()
        for cn in self.clusters[a]:
            for j in self.CN[cn]:
                vns.add(j)
        for j in vns:
#             print("doing ",j)
            total_llr = 0
            for i in self.VN[j]:
                if i in self.clusters[a]:
                    print(f"adding {self.Lij[i,j]} to total llr for {i},{j}")
                    total_llr=total_llr+self.Lij[i,j]
            for i in self.VN[j]:
                if i in self.clusters[a]:
#                     print("Updating Lji",j,",",i,"] to ",self.L[j] + total_llr - self.Lij[i,j])
                    self.Lji[j,i] = self.L[j] + total_llr - self.Lij[i,j]
            # updating Ltot here only (comment out calling compute_LLR)
            print(f"L_tot[{j}]={total_llr}+{self.L[j]}")
            self.L_tot[j] = total_llr + self.L[j]

    # compute LLRs of cluster a after an iteration of decoding
    def compute_LLR(self,a):
        for cn in self.clusters[a]:
            for j in self.CN[cn]:
                tot = 0
                for i in self.VN[j]:
                    tot+=self.Lij[i,j]
                self.L_tot[j] = self.L[j]+tot
#         print("L_tot : ",self.L_tot)
        
    # conditions for stopping the decoding (convergence or max_iterations)
    def stopping_criterion(self, iteration_number):
        v = np.array([1 if val<0 else 0 for val in self.L_tot])
        c1 = iteration_number>=self.num_iter
        c2 = np.sum(np.dot(self.H,v.T)%2)==0
        if c1 or c2:
            print("stopping! at condition ",c1,c2,v)
            return True
        return False
    
    # function to perform 1 iteration of message passing on the cluster a
    def decode_cluster(self,a):
        self.CN_update(a)
#         print(self.Lij)
        self.VN_update(a)
#         print(self.Lji)
    
    # function to decode the given recieved vector y
    def decode(self, y, iter_step=100,verbose="off"):
        self.initialize_clusters(6)
        self.initialize_LLR(y)
        
        for k in range(self.num_iter):
            a = self.get_next_cluster(k)
            if verbose=="on" and k%iter_step==0:
                print("Iteration ",k,"Cluster : ",a,end="\t")
                print("LLRs : ",np.round(self.L_tot,2),end=" ")
                print()
            self.decode_cluster(a)
#             self.compute_LLR(a)
            if self.stopping_criterion(k):
                break
        v = self.L_tot<0
        return v
    
    def get_next_cluster(self,iter_number):
        # round robin
        return (iter_number%self.num_clusters)
        
    
        