import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


def Make_matrix(condition,dom,N,P,Q,R,al_3,bet_3,al_1=None,al_2=None,bet_1=None,bet_2=None):
    x = np.linspace(dom[0],dom[-1],N+2)
    h = x[1]-x[0]
           
    A = np.zeros((len(x),len(x)))       
    B = np.zeros(len(x))
    for i in range(len(x)):
        diag = lambda t : 2 + (h**2)*P(x[t])
        upper = lambda t :  ((h/2)*Q(x[t]) - 1)
        lower = lambda t : - ((h/2)*Q(x[t]) + 1)
        cons = lambda t : -(h**2)*R(x[t])
        
        if i ==0:
           if condition == 'Dirichlet':
              A[i,i] = 1 ; A[i,i+1] = 0
              B[i] = al_3
           else :
              A[i,i] =  diag(i) + 2*h*(al_1/al_2)*lower(i)
              A[i,i+1] = -2
              B[i] = cons(i) + 2*h*(al_3/al_2)*lower(i)
           continue
        if i == (len(x)-1):
           if condition == 'Dirichlet':
              A[i,i] = 1 ; A[i,i-1] = 0
              B[i] = bet_3 
           else :
              A[i,i] =  diag(i) - 2*h*(bet_1/bet_2)*upper(i)
              A[i,i-1] = -2
              B[i] = cons(i) - 2*h*(bet_3/bet_2)*upper(i)
           break 
        A[i,i] = diag(i)
        A[i,i+1] = upper(i)  
        A[i,i-1] = lower(i)
        B[i] = cons(i)
        
    return A,B   

def Crout(A):
    m,n = A.shape
    L = np.zeros((n,n))
    U = L.copy()
    
    for i in range(n):
        U[i,i] = 1
        
    for j in range(n):
        dot = np.dot(L[:,:j],U[:j,j])
        L[:,j] = A[:,j] - dot
        
        dot2 = np.dot(L[j,:j],U[:j,:])
        U[j,:] = (A[j,:] - dot2)/L[j,j]
        
    return L,U 

def Solution_fin(condition,dom,N,P,Q,R,al_3,bet_3,al_1=None,al_2=None,bet_1=None,bet_2=None):
    
    A,B = Make_matrix(condition, dom, N, P, Q, R, al_3, bet_3,al_1,al_2,bet_1,bet_2)
    L,U = Crout(A)
    
    x = np.linspace(dom[0], dom[-1],N+2)
    Z = np.zeros(len(x))
    # finding LZ = B using substitution
    for i in range(len(Z)):
        
        Z[i] = (B[i] - np.dot(L[i,:i],Z[:i]))/L[i,i]
    
    Y = np.linalg.solve(A, B)
    
    return x,Y 
        
def plotting(f,arr,N,title):
    sig_l = ['o','1','v','*','2','x','d','o','x','v','*','2','v']
    fig,ax = plt.subplots()
    
    for i in range(len(N)):
        ax.plot(arr[i][0],arr[i][1],f'--{sig_l[i]}',label = f'for N = {N[i]}')
    ax.plot(arr[4][0],f(arr[4][0]),'b')
    ax.legend() 
    ax.set_title(title)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')

def solve_cnt(f,condition,dom,N,P,Q,R,al_3,bet_3,al_1=None,al_2=None,bet_1=None,bet_2=None):

    df = []
    for i in range(len(N)):
        df1 = Solution_fin(condition,dom,int(N[i]),P,Q,R,al_3,bet_3,al_1,al_2,bet_1,bet_2)
        df.append(df1)
    
    plotting(f,df,N,condition)

def Error(f,condition,dom,N,P,Q,R,al_3,bet_3,al_1=None,al_2=None,bet_1=None,bet_2=None):
    mat = Solution_fin(condition,dom,N,P,Q,R,al_3,bet_3,al_1,al_2,bet_1,bet_2)
    x = mat[0]
    h = x[1] - x[0]
    ynum = mat[1]
    E_l = abs(ynum - f(x)) 
    E_n = max(E_l)
    E_r = np.sqrt(np.sum(np.power(E_l,2))/len(E_l))
    data = {'x':x,'y_num':ynum,'y_analytic':f(x),'Error':E_l}
    df  = pd.DataFrame(data)
    return np.log(N),np.log(h),np.log(E_n),np.log(E_r),df



#first equation

P = lambda x : np.pi**2
r = lambda x : -2*(np.pi**2)*np.sin(np.pi*x)
Q = lambda x : 0

dom1 = [0,1]

def analy(x):
    return np.sin(np.pi*x)

analy = np.vectorize(analy)

N1 = np.logspace(1,6,base=2,num = 6,dtype=(float))

#solve_cnt(analy,"Dirichlet", dom1, N1,P, Q, r,0,0)

Err1 = []

for i in range(len(N1)):
    
    Err1.append(Error(analy,'Dirichlet',dom1,int(N1[i]),P,Q,r,0,0))

for i in [1,2]:
    print(Err1[i][4])
h1 = []
e_m1 = []
for i in range(len(N1)):
    h1.append(Err1[i][1])
    e_m1.append(Err1[i][2])
    
plt.plot(h1,e_m1)  
slope_max, intercept_max,r_value, p_value, std_err = stats.linregress(h1,e_m1)  

print(slope_max)

#second equation

P1 = lambda x: -1
Q1 = lambda x:0
r1 =  lambda x: np.sin(3*x)

dom2 = [0,np.pi/2]


def analy_2(x):
    return (3/8)*np.sin(x) - np.cos(x) - (1/8)*np.sin(3*x)

#solve_cnt(analy_2,"Mixed BC for DE 2", dom2, N1,P1, Q1, r1,-1,1,1,1,0,1)


             