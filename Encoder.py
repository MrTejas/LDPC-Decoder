import numpy as np
import random

class Encoder:
    
    def __init__ (self, base_matrix_path, expansion_factor):
        self.z = expansion_factor
        self.B = np.loadtxt(base_matrix_path, dtype=int)
    
    def mul_sh(self, x, k):
        # x: input block
        # k: -1 or shift
        # y: output
        if k == -1:
            y = np.zeros(len(x), dtype=int)
        else:
            y = np.concatenate((x[k:], x[:k]))  # multiplication by shifted identity
        return y

    def check_cword(self, c):
        # B: base matrix
        # z: expansion factor
        # c: candidate codeword, length = #cols(B) * z
        # out = 1, if codeword is valid; 0, else
        
        B = self.B
        z = self.z
        m, n = B.shape
        syn = np.zeros(m * z, dtype=int)  # Hc^T
        for i in range(m):
            for j in range(n):
                syn[i * z:(i + 1) * z] = (syn[i * z:(i + 1) * z] + self.mul_sh(c[j * z:(j + 1) * z], B[i, j])) % 2
        return 0 if np.any(syn) else 1

    def nrldpc_encode(self, msg):
        # B: base matrix
        # z: expansion factor
        # msg: message vector, length = (#cols(B)-#rows(B))*z
        # cword: codeword vector, length = #cols(B)*z

        B = self.B
        z = self.z
        m, n = B.shape
        cword = np.zeros(n * z, dtype=int)
        cword[:(n - m) * z] = msg

        # double-diagonal encoding
        temp = np.zeros(z, dtype=int)
        for i in range(4):  # row 1 to 4
            for j in range(n - m):
                temp = (temp + self.mul_sh(msg[j * z:(j + 1) * z], B[i, j])) % 2
        p1_sh = B[2, n - m] if B[1, n - m] == -1 else B[1, n - m]
        cword[(n - m) * z:(n - m + 1) * z] = self.mul_sh(temp, z - p1_sh)  # p1

        # Find p2, p3, p4
        for i in range(3):
            temp = np.zeros(z, dtype=int)
            for j in range(n - m + i+1):
                temp = (temp + self.mul_sh(cword[j * z:(j + 1) * z], B[i, j])) % 2
            cword[(n - m + i +1) * z:(n - m + i + 1 + 1) * z] = temp

        # Remaining parities
        for i in range(5, m + 1):
            temp = np.zeros(z, dtype=int)
            for j in range(n - m + 4):
                temp = (temp + self.mul_sh(cword[j * z:(j + 1) * z], B[i - 1, j])) % 2
            cword[(n - m + i - 1) * z:(n - m + i - 0) * z] = temp

        return cword

    
    
#     def encode(self, msg):
#         """
#         LDPC encoding function based on base matrix (B), expansion factor (z), and message (msg).
#         B: base matrix
#         z: expansion factor
#         msg: message vector, length = (#cols(B)-#rows(B))*z
#         cword: codeword vector, length = #cols(B)*z
#         """
#         B = self.B
#         z = self.z
#         m, n = B.shape
#         cword = np.zeros(n * self.z, dtype=int)
#         cword[:(n - m) * self.z] = msg

#         # double-diagonal encoding
#         temp = np.zeros(z, dtype=int)
#         for i in range(4):  # row 1 to 4
#             for j in range(n - m):  # message columns
#                 if B[i, j] != -1:  # Handle -1 values in the base matrix (CHANGED)
#                     temp = (temp + self.mul_sh(msg[j * z:(j + 1) * z], B[i, j])) % 2

#         # Calculate p1
#         if B[1, n - m] == -1:
#             p1_sh = B[2, n - m]
#         else:
#             p1_sh = B[1, n - m]

#         cword[(n - m) * z:(n - m + 1) * z] = self.mul_sh(temp, z - p1_sh)

#         # Find p2, p3, p4
#         for i in range(1, 4):
#             temp = np.zeros(z, dtype=int)
#             for j in range(n - m + i):
#                 if B[i, j] != -1:  # Handle -1 values in the base matrix (CHANGED)
#                     temp = (temp + self.mul_sh(cword[j * z:(j + 1) * z], B[i, j])) % 2
#             cword[(n - m + i) * z:(n - m + i + 1) * z] = temp

#         # Remaining parities
#         for i in range(4, m):
#             temp = np.zeros(z, dtype=int)
#             for j in range(n - m + 4):
#                 if B[i, j] != -1:  # Handle -1 values in the base matrix (CHANGED)
#                     temp = (temp + self.mul_sh(cword[j * z:(j + 1) * z], B[i, j])) % 2
#             cword[(n - m + i - 1) * z:(n - m + i) * z] = temp

#         return cword
    
    def generate_random_message(self):
        """
        Function to generate a random message vector according to the code parameters
        input : None
        output : random 1d message numpy array of message length
        """
        k = (self.B.shape[1]-self.B.shape[0])*self.z
        msg = np.array(np.random.randint(2, size=k))
        return msg
    
    
    def circular_shift_identity(self, z, k):
        """
        Generate a z x z identity matrix with circularly shifted rows by k positions.
        If k == -1, return a z x z zero matrix.
        """
        if k == -1:
            return np.zeros((z, z), dtype=int)
        else:
            return np.roll(np.eye(z, dtype=int), -k, axis=1)

    def expand_base_matrix(self, B, z):
        """
        Expand the base matrix B into the full parity check matrix with expansion factor z.

        B: Base matrix of size m x n
        z: Expansion factor

        Returns the expanded parity-check matrix of size (m*z) x (n*z).
        """
        m, n = B.shape
        H = np.zeros((m * z, n * z), dtype=int)  # Full parity check matrix initialized to zeros

        for i in range(m):
            for j in range(n):
                # Expand each base matrix element into a z x z block
                block = self.circular_shift_identity(z, B[i, j])
                H[i * z:(i + 1) * z, j * z:(j + 1) * z] = block

        return H

    
    def get_parity_check_matrix(self):
        return self.expand_base_matrix(self.B, self.z)
        


