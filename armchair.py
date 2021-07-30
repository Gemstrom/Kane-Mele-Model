import numpy as np
import matplotlib.pyplot as plt
import math
import cmath 
import functools

def sz(spin):
#turn spin number into z projection of spin
    if spin == 0 : 
    #spin up
        return 1
    if spin == 1 :
    #spin down
        return -1
    else :
        print("Wrong spin.")

def KMHam(N,t,t2,k):
#N 4-atom inferior struture in a cell, with nearest hopping term t and second nearest t2, k
    Ham = np.zeros((4*2*N,4*2*N),dtype = complex)
    #m is the order of inferior structure, num is the number of atom, spin=0 shows spin up,1 shows spin down
    #nearest hopping
    for spin in range(2):
        #intra cell
        #intra unit
        for m in range(N):
            Ham[8*m+2*0+spin, 8*m+2*1+spin] = t
            Ham[8*m+2*0+spin, 8*m+2*2+spin] = t
            Ham[8*m+2*1+spin, 8*m+2*3+spin] = t
        #inter unit
        for m in range(N-1):
            Ham[8*m+2*2+spin, 8*(m+1)+2*0+spin] = t
            Ham[8*m+2*3+spin, 8*(m+1)+2*1+spin] = t 
        #inter cell
        for m in range(N):
            Ham[8*m+2*2+spin, 8*m+2*3+spin] = t* cmath.exp(-1j*3*k)
    #second nearest hopping
    for spin in range(2):
        #intra cell
        #intra unit
        for m in range(N):
            Ham[8*m+2*0+spin, 8*m+2*3+spin] = -1j * t2 * sz(spin)
            Ham[8*m+2*1+spin, 8*m+2*2+spin] = 1j * t2 * sz(spin)
        #inter unit
        for m in range(N-1):
            Ham[8*m+2*0+spin, 8*(m+1)+2*0+spin] = 1j * t2 * sz(spin)
            Ham[8*m+2*1+spin, 8*(m+1)+2*1+spin] = -1j * t2 * sz(spin)
            Ham[8*m+2*0+spin, 8*(m+1)+2*0+spin] = -1j * t2 * sz(spin)
            Ham[8*m+2*0+spin, 8*(m+1)+2*0+spin] = 1j * t2 * sz(spin)
            
            Ham[8*m+2*2+spin, 8*(m+1)+2*1+spin] = 1j * t2 * sz(spin)
            Ham[8*m+2*3+spin, 8*(m+1)+2*0+spin] = -1j * t2 * sz(spin)
            
        #inter cell 
        for m in range(N):
            Ham[8*m+2*0+spin, 8*m+2*3+spin] = Ham[8*m+2*0+spin, 8*m+2*3+spin] - 1j * t2 * sz(spin) * cmath.exp(-3*k*1j)
            Ham[8*m+2*1+spin, 8*m+2*2+spin] = Ham[8*m+2*1+spin, 8*m+2*2+spin] + 1j * t2 * sz(spin) * cmath.exp(3*k*1j)
        for m in range(N-1):
            Ham[8*m+2*3+spin, 8*(m+1)+2*0+spin] = Ham[8*m+2*3+spin, 8*(m+1)+2*0+spin] -1j * t2 * sz(spin)* cmath.exp(3*k*1j)
            Ham[8*m+2*2+spin, 8*(m+1)+2*1+spin] = Ham[8*m+2*2+spin, 8*(m+1)+2*1+spin] +1j * t2 * sz(spin)* cmath.exp(-3*k*1j)
        
    Ham = Ham + Ham.conj().T      
            
    E,v = np.linalg.eig(Ham)
    return E

def main():
    # N : the number of units per cell
    # n : the number of kx 
    N = 20
    n = 500
    
    kk = np.linspace(-cmath.pi/3, cmath.pi/3, n)
    E = np.zeros((n,(4*2*N)),dtype = complex)
    
    for i in range(n):
        E[i] = KMHam(N,1,0.03,kk[i])
    
    #plotting
    plt.text(-1.1,1.05,'E/t')
    plt.text(1.5,-1.2,'kx')
    plt.xlim(-cmath.pi/3, cmath.pi/3)
    plt.ylim(-1,1)
    plt.plot(kk,E.real,'k.')
    plt.savefig("armchair.png")
    plt.show()
    
if __name__ == '__main__':
    main()
