import numpy as np
import random
import math
from Dec import Dec
import gc


class Reldec:

    def __init__(self, dec, C, alpha, beta, epsilon, max_iter):
        """
        Parameters and variables : 
        - dec : Decoder instance passed 
        - H : parity check matrix for the decoder
        - num_iter : max number of iterations for decoder instances
        - cluster_size : size of the clusters (all equal) for decoders
        - clusters : list defining clusters for decoders
        - q : number of clusters
        - C : list of all codewords for the given H
        - alpha : learning rate for q-learning
        - beta : discount factor for q-learning 
        - epsilon : used in epsilon greedy policy choosing
        - max_iter : max number of iterations for an episode of q-learning
        """

        self.dec = dec
        self.H = self.dec.H
        self.num_iter = self.dec.num_iter
        self.cluster_size = self.dec.cluster_size 
        self.clusters = self.dec.clusters
        self.q = math.ceil(dec.num_CN/dec.cluster_size) # number of clusters
        self.C = C
        self.alpha = alpha
        self.beta = beta
        self.epsilon = epsilon
        self.max_iter = max_iter
        self.epsilon_list = []

        self.initialize()

    def initialize(self):
        """"
        Function that is called once for a given decoder setting. Initializes cluster 
        degrees, bit-lengths for state encoding/decoding, num_states, num_actions, 
        and q-table to stored the policy learnt.
        """
        # can be optimized later (please do!)
        # determine l_star = max(la)
        self.l = [0]*self.q
        l_star = -1
        ind = 0
        for cluster in self.clusters:
            st = set()
            for i in cluster:
                st.update(self.dec.CN[i])
            l_a = len(st)
            self.l[ind] = l_a
            ind = ind + 1
            l_star = max(l_star,l_a)
        self.l_star = l_star

        # determine number of states = log2(q)*2^(l_star)
        self.cluster_bits = math.ceil(math.log2(self.q))
        self.cluster_state_bits = self.l_star
        total_bits = self.cluster_bits + self.cluster_state_bits
        self.num_states = int(math.pow(2,total_bits))

        # determine the number of actions (q ranging from 0 to q-1)
        self.num_actions = int(self.q)

        # initialize the q-table and fill with 0s
        self.q_table = np.random.rand(self.num_states,self.num_actions)

    def encode_state(self, cluster, cluster_state):
        # takes in cluster number [0 to q-1] and cluster_state [0 to 2^l_a] and encodes it to a number
        # state = [cluster_state,cluster] encoded
        state = cluster + (cluster_state<<(self.cluster_bits))
        return state

    def decode_state(self, state):
        # takes the state and extracts cluster state and cluster from it
        mask = (1<<self.cluster_bits)-1
        cluster = state & mask
        # binary_rep = bin(state)[2:]
        # cluster_state = int(binary_rep[:self.cluster_state_bits], 2)
        cluster_state = state>>self.cluster_bits
        return cluster_state, cluster

    def get_reward(self, a, num):
        # return the reward obtained for cluster 'a' and cluster_State 'num'
        sum = 0
        st = set()
        for i in self.dec.clusters[a]:
            st.update(self.dec.CN[i])
        for j in st:
            actual_coordinate = self.c[j]
            predicted_coordinate = num & (1<<(self.l[a]-j-1))
            sum = sum + (1 if actual_coordinate==predicted_coordinate else 0)

        r = sum/self.l[a]
        return r



    def take_action(self, a):
        # take action by scheduling cluster a
        self.dec.row_update(a)
        self.dec.col_update(a)

        # obtaining new cluster state of cluster 'a'
        cluster_state = self.get_cluster_state(a)
        # print(f"cluster,cluster_state = {a},{bin(cluster_state)[2:]}")

        # obtanining new state and reward
        s_prime = self.encode_state(a,cluster_state)
        reward = self.get_reward(a, cluster_state)

        return s_prime, reward
    
    def get_cluster_state(self, a):
        st = set()
        for i in self.clusters[a]:
            st.update(self.dec.CN[i])
        num = 0
        for j in st:
            bit = 1 if self.dec.sum[j]<0 else 0
            num = (num<<1)+bit
        for k in range(self.l_star - self.l[a]):
            num = (num<<1)
        return num

    def choose_action(self,state):
        self.epsilon = max(0.01, self.epsilon * 0.99)  # Decays epsilon, with a minimum value of 0.01
        self.epsilon_list.append(self.epsilon)
        # Randomly choose action with probability epsilon (explore)
        if random.uniform(0, 1) < self.epsilon:
            return random.randint(0, self.num_actions - 1)
        # Choose the action with the highest Q-value (exploit)
        return np.argmax(self.q_table[state])



    def train(self,snrdB):
        """
        Main training loop. Implements q-learning for |snrdB| number of episode.
        For each snrdB in snrdB, it chooses a codeword and corrupts it with the snrdB and performs RL.
        """


        for snrdb in snrdB:
            print(f"SNRdB = {snrdb}")
            # initializing transmitted and recieved vector for given snr
            snr = math.pow(10,snrdb/20)
            sigma = 1/math.sqrt(snr)
            self.c = self.C[np.random.choice(len(self.C))]
            self.y = np.power(- 1,self.c) + np.random.normal(loc=0, scale=sigma, size=self.c.shape)

            # initializing the decoder
            self.dec = Dec(self.H,None,None,self.num_iter,self.cluster_size)
            self.dec.sum = self.y
            self.dec.L = np.full(self.dec.H.shape, np.nan, dtype=float)  # Initialize with NaNs
            for i in range(self.dec.num_CN):
                for j in range(self.dec.num_VN):
                    self.dec.L[i,j] = 0 if self.dec.H[i,j]==1 else np.nan

            # choosing a random starting state initially
#             a = random.randint(0, self.q-1)
            a = 0
            cluster_state = self.get_cluster_state(a)
            self.state = self.encode_state(a,cluster_state)
            iter = 0
            self.diff = self.q_table

            while iter<self.max_iter or np.max(self.diff)<0.2:
                a = self.choose_action(self.state)
                # print(f"action {a} choosen")
                s_prime, reward = self.take_action(a)
                # print(f"state {s_prime} reached and reward {reward} collected")
                temp = (1-self.alpha)*self.q_table[self.state,a] + self.alpha*(reward + self.beta*np.max(self.q_table[s_prime]))
                # added decaying learning rate (remove this after experimentation)
#                 self.alpha = self.alpha * 0.99
                self.q_table[self.state,a] = temp
                self.diff = np.abs(temp - self.q_table)
                # print(f"updating q[{self.state},{a}] to {self.q_table[self.state,a]}")
                self.state = s_prime
                iter = iter + 1
            print(f"diff average = {np.average(self.diff)}")
            
            # deleting the decoder memory (for new usage in next iter)
            # del self.dec
            # gc.collect()







