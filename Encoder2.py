import numpy as np
import random

class Encoder2:
    
    def __init__ (self, base_matrix_path, expansion_factor):
        self.z = expansion_factor
        self.B = np.loadtxt(base_matrix_path, dtype=int)
        self.find_generator_matrix()
    
    def encode(self,m):
        """
        Function to encode a message m into codeword c using the Generator matrix G
        """
        return np.dot(m,self.G)%2;
    
    def find_generator_matrix(self):
        """
        Function to find the generator matrix given non-systematic parity check matrix
        """
        
        # first convert H into its systematic form
        H = self.get_parity_check_matrix()
        print("H is : \n",H)
        rows, cols = H.shape
        r = 0
        for c in range(cols - rows, cols):
            # If H[r, c] == 0, find a row below it with a 1 and swap rows
            if H[r, c] == 0:
                for r2 in range(r + 1, rows):
                    if H[r2, c] != 0:
                        H[[r, r2]] = H[[r2, r]]  # Swap rows
                        break

            # If we still have a zero, H is singular
            if H[r, c] == 0:
                raise ValueError('H is singular')

            # Forward substitution: Make all entries below H[r, c] equal to 0
            for r2 in range(r + 1, rows):
                if H[r2, c] == 1:
                    H[r2] = np.bitwise_xor(H[r2], H[r])

            # Back substitution: Make all entries above H[r, c] equal to 0
            for r2 in range(r):
                if H[r2, c] == 1:
                    H[r2] = np.bitwise_xor(H[r2], H[r])

            # Move to the next row
            r += 1
        self.H = H
        self.G = self.HtoG(self.H)
        
    def HtoP(self, H):
        """Extract the submatrix P from a Parity Check Matrix in systematic form.
        Args:
            H: Parity Check Matrix in systematic form
        Returns:
            Submatrix P of G.
        """
        n = np.shape(H)[1]
        k = n - np.shape(H)[0]
        PK = H[:,0:n-k]
        P = np.transpose(PK)
        return P.astype(int)


    def HtoG(self, H):
        """Convert a Parity Check Matrix in systematic form to a Generator Matrix.
        Args:
            H: Parity Check Matrix in systematic form
        Returns:
            Generator Matrix G
        """
        n = np.shape(H)[1]
        k = n - np.shape(H)[0]
        print("n = ",n)
        print("k = ",k)
        P = HtoP(H)
        print("P extracted : ",P)
        Ik = np.eye(k)
        G = np.concatenate((Ik,P), axis=1)
        return G.astype(int)

        
    
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
        


