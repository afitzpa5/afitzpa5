# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import numpy.random as random
import scipy
from scipy import constants
import itertools
from numpy import linalg as la

class unique_element:
    def __init__(self,value,occurrences):
        self.value = value
        self.occurrences = occurrences
        
def perm_unique(elements):
    eset=set(elements)
    listunique = [unique_element(i,elements.count(i)) for i in eset]
    u=len(elements)
    return perm_unique_helper(listunique,[0]*u,u-1)

def perm_unique_helper(listunique,result_list,d):
    if d < 0:
        yield tuple(result_list)
    else:
        for i in listunique:
            if i.occurrences > 0:
                result_list[d]=i.value
                i.occurrences-=1
                for g in  perm_unique_helper(listunique,result_list,d-1):
                    yield g
                i.occurrences+=1


class hubbard:
    def __init__(self, sites=4, filling = 2, pbcs=True, t = 1, eps = np.ones(1), V=1/4, gridtype = '2D'):
    #The first step is to initialise/create various lists/value we us throughout the code                 
        self.sites=sites 
        self.filling = filling
        self.pbcs=pbcs                                                                                    
        self.neighbours=[]                      
        self.t = t
        self.eps = eps
        self.V = V
        self.states = self.create_basis(self.sites, self.filling)
        self.dimension = int(np.sqrt(self.sites))
        self.gridtype = gridtype
        
        if self. gridtype == '2D':
            self.create_square_grid()
        
        #print('Basis = ', self.states)
        #print(self.neighbours)
        
    def create_basis(self, sites, filling):
        sitebasis1 = np.concatenate((np.ones(filling), np.zeros(sites-filling)), axis = None) 
        ei = list(perm_unique(sitebasis1.tolist()))
        basis = np.array(ei)
        states=[]
        for i in range(len(basis)):
            p=0
            basis1 = np.flip(basis[i])
            for j in range(len(basis1)):
                p += basis1[j]*2**j
            states.append(p)


        #sort them for ease of reading
        states = np.sort(states)
        states = states.astype(np.int64)
        #dim = len(states)
        return  states
    
    
    def swapbits(self, x, p1, p2): 
  
        #isolating the first bit to swap  
        set1 =  (x >> p1) & 1 

        #isolating the second bit to swap  
        set2 =  (x >> p2) & 1

        # XOR the two sets  
        xor = (set1 ^ set2) 

        # Put the xor bits back 
        # to their original positions  
        xor = (xor << p1) | (xor << p2) 

        # XOR the 'xor' with the 
        # original number so that the  
        # two sets are swapped 
        result = x ^ xor 

        #the following if statement ensures states are set to 0 if a "1" bit on the left hand side is moved to the left
        #as we cannot move rightmost bits to the right with this function we need not worry
        if self.gridtype == '1D':
            if len(bin(result))>len(bin(self.states[len(self.states)-1])): 
                result = 0

        return result
    
    def countsetbits(self, n):
        count = 0
        while (n):
            n &= (n-1)
            count +=1
        return count
    
    def create_square_grid(self):
        
        #below a 2d grid is created, 
        x, y = np.linspace(0,self.dimension-1, self.dimension), np.linspace(0,self.dimension-1, self.dimension)
        [Y, X] = np.meshgrid(y,x)
        self.X, self.Y = X.flatten(), Y.flatten()
        alist1 = []
        list1 = []
        
        #here we generate the list of neighbours for each site, generated differently depending on whether periodic boundary
        #conditions are being used.
        
        for i, (x1, y1) in enumerate(zip(self.X, self.Y)):
            
            if self.pbcs==True:
                templist = [[ (x1-1) % self.dimension, y1], [(x1+1) % self.dimension, y1], [x1, (y1-1) % self.dimension], [x1, (y1+1) % self.dimension]]
                reallist = [int(a[0]*self.dimension+a[1]) for a in templist]
                self.neighbours.append(reallist)
                
        """n1 = np.array(self.neighbours)
        n1 = np.flip(n1)
        n1 = n1.tolist()"""
     
 
    def isKthBitSet(self, n, k):  
        if ((n >> (k - 1)) & 1): 
            return True 
        else: 
            return False   
            
        
        
    def hamiltonian(self):
        dim = len(self.states)
        bindim = len(bin(self.states[dim-1]))
        H = np.zeros([len(self.states),len(self.states)])

#       setbits = np.zeros(len(self.states))
        
        if self.gridtype == '1D':
            for i in range(len(self.states)):
                count = 0
                for j in range(1,bindim-1):
                    if (self.isKthBitSet(self.states[i],j)):
                        count += self.eps[j-1] 
                        
                H[i][i]+=count + self.countsetbits(self.states[i]&(self.states[i]<<1))*self.V
                
                
                if self.pbcs == True:
                    if len(bin(self.states[i])) == bindim and bin(self.states[i])[2]=='1' and  bin(self.states[i])[bindim-1]=='1':
                        H[i][i]+=self.V
                    
                    y=self.swapbits(self.states[i],0,self.sites-1)
                    if y in self.states and y!= self.states[i]:
                        H[i][np.where(self.states==y)]+=(-self.t)
                        
                    
                for p in range(len(bin(self.states[i]))-2):
                    x=self.swapbits(self.states[i],p,p+1)
                    if x in self.states and x != self.states[i]:
                        H[i][np.where(self.states==x)]+=(-self.t)
        
        if self.gridtype == '2D':
            for i in range(len(self.states)):
                count = 0
                #here we check if a given bit is 1, if so, on site energy is added to count
                for j in range(1,bindim-1):
                    if (self.isKthBitSet(self.states[i],j)):
                        count += self.eps[j-1] 
                
                #we add the sum of the on site energies plus any interaction terms to diagonal elements
                H[i][i]+=count + self.countsetbits(self.states[i]&(self.states[i]<<1))*self.V                
                
      
                for j in range(1,self.sites+1):
                    if (self.isKthBitSet(self.states[i],j)):  
                        #we only do when bit is set to save time and avoid duplicates
                        for k in range(4):
                            x = self.swapbits(self.states[i],j-1,self.neighbours[j-1][k])
                            if x in self.states and x == self.states[i]:
                                H[i][i]+=self.V/2
                            if x in self.states and x != self.states[i]:
                                H[i][np.where(self.states==x)]+=(-self.t)
                
        return H