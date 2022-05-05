# -*- coding: utf-8 -*-
"""
Created on Sat Apr 30 04:07:53 2022

@author: harsh
"""

import numpy as np
from numpy import vectorize as vec

def get_tridiag(l,d,u):
    N = len(d)
    if not (N-1 == len(l) and N-1 ==len(u)):
        raise ValueError(f"length of l and d must be {N-1}.")
    return np.diag(l,-1) + np.diag(d,0) + np.diag(u,1)

#y′′(x) + p(x)y′(x) + q(x)y(x) + r(x) = 0
class ordinary_bvp:
    def __init__(self,p,q,r,dom=(0,1),exact=lambda x:1 ) -> None:
        self.p = p 
        self.q = q 
        self.r = r
        self.dom = dom
        self.arb,self.brb = (1,0,0),(1,0,0) #default boundary conditions
        self.exact = exact
        pass

    def discretize(self,N:int):
        self.N = N 
        self.ddom = np.linspace(self.dom[0],self.dom[1],N+1)
        self.w = np.zeros(self.ddom.shape) 
        #ddom,w = [0,1,2....N-1,N]
        self.h = abs(np.diff(self.ddom)[0])
        h,x = self.h,self.ddom
        self.d = vec(lambda i : 2 + h**2*self.p(x[i]))
        self.u = vec(lambda i : -1 + h/2*self.q(x[i]))
        self.l = vec(lambda i : -1 - h/2*self.q(x[i]))
        self.b = vec(lambda i : -h**2*self.r(x[i]))  
        return self.ddom
    
    def set_dirichlet(self,a,b):
        self.set_robin((1,0,a),(1,0,b))

    def set_neumann(self,a,b):
        self.set_robin((0,1,a),(0,1,b))
    
    def set_robin(self,a=None,b=None): # a1 y(0) + a2 y'(0) = a3 ; b1 y(N) + b2y'(N) = b3 
        if a == (0,0,0) or b==(0,0,0):
            raise ValueError("Give appropriate Boundary conditions, (0,0,0) is nonsense.")
        if a is not None:
            self.arb = a
        if b is not None:
            self.brb = b
        (a1,a2,a3),(b1,b2,b3) = self.arb,self.brb
        h,x = self.h,self.ddom
        b_,d,l,u,N = self.b,self.d,self.l,self.u,self.N
        if a2 == 0 :
            self.a11,self.a12 = 1,0
            self.b1 = a3/a1
        else:
            self.a11,self.a12 = d(0) + 2*h*l(0)*a1/a2,-2
            self.b1 = b_(0)+2*h*l(0)*a3/a2
        if b2 == 0 :
            self.ann,self.an_1n = 1,0
            self.bn = b3/b1
        else:
            self.ann,self.an_1n = d(N) - 2*h*u(N)*b1/b2 ,-2
            self.bn = b_(N)-2*h*l(N)*b3/b2
        
    def get_A_b(self):
        ii = np.arange(1,self.N)
        l_,d_ = np.zeros(self.N),np.zeros(self.N+1)
        u_,self.b_ = l_.copy(),d_.copy()
        l_[:-1],l_[-1] = self.l(ii), self.an_1n 
        u_[1:],u_[0] = self.u(ii), self.a12 
        d_[1:-1],d_[0],d_[-1] = self.d(ii),self.a11,self.ann
        self.b_[1:-1],self.b_[0],self.b_[-1] = self.b(ii),self.b1,self.bn 
        self.A = get_tridiag(l_,d_,u_)
        return self.A,self.b_

    def solve(self):
        A,b =self.get_A_b()
        soln = np.linalg.solve(A,b)
        anasoln = self.exact(self.ddom)
        rmse = np.sqrt((soln-anasoln)**2/self.N) 
        return soln,rmse

    def plot_exact(self,ax):
        x_space = np.linspace(*self.dom)
        ax.plot(x_space,self.exact(x_space),label="Exact")
    def plot_num(self,ax):
        ax.plot(self.ddom,np.linalg.solve(self.A,self.b_),"1",label=f"$N={self.N}$")
        

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from matplotlib import use
    plt.style.use("bmh")
    bvp1 = ordinary_bvp(lambda x: np.pi**2,lambda x:0,lambda x:-2*np.pi**2*np.sin(np.pi*x),(0,1),lambda x : np.sin(np.pi*x))
    bvp1.discretize(5)
    bvp1.set_robin((1,0,0),(1,0,0))
    A,b = bvp1.get_A_b()
    
    fig1,ax1 = plt.subplots(1,1)
    bvp1.plot_exact(ax1)
    bvp1.plot_num(ax1)
    plt.legend()
    plt.show()

    bvp2 = ordinary_bvp(lambda x: -1,lambda x:0,lambda x:np.sin(3*x),(0,np.pi/2))
    bvp2.discretize(5)
    bvp2.set_robin((1,1,-1),(0,1,1))
    y2_exact = lambda x : 3/8*np.sin(x) - np.cos(x) - 1/8*np.sin(3*x) 
    A,b = bvp2.get_A_b()
    print(A)
    print(b)