import numpy as np

def gradient(x,offs,learning_rate):
    A_exp =np.exp((100-np.sqrt(x[1]**2+x[0]**2))/3.14)
    A =  ( 1+np.abs(np.sin(x[0]) * np.sin(x[1]) *  A_exp))
    f = - A**(-0.1)

    E_tot= f

    y=-1

    h = np.log(E_tot + offs)
    y_corre = np.log(y + offs)
    y_00=(h-y_corre)


    y_=y_00**2


    cost_2 = np.square(y_00).sum()

    y_pred = -2*y_

    grad_=y_pred/(E_tot+offs)*y_00
    grad_h9 = grad_#*2*y_00 #

    d_x1 = -0.1*A**(-1.1) * (-0.5*A_exp/np.sqrt(x[0]**2+x[1]**2)) * x[0]**2
    d_x2 = -0.1*A**(-1.1) * (-0.5*A_exp/np.sqrt(x[0]**2+x[1]**2)) * x[1]**2

    d_x1 += 0.1*A**(-1.1) *  (np.sin(x[0])*np.cos(x[0]))*x[0]
    d_x2 += 0.1*A**(-1.1) *  (np.sin(x[1])*np.cos(x[1]))*x[1]



  #  A_sign = 1
    x[0] += grad_h9* d_x1 * learning_rate
   # x[1] += grad_h9* d_x2 * learning_rate #*A_sign


    return (x,E_tot,d_x1*grad_h9)

x=np.arange(-1.0,1.,0.1)*5.12

d=2

f_para=open('para.dat','w')
f_cost=open('cost.dat','w')
f_Etot=open('Etot.dat','w')
f_des=open('des.dat','w')


learning_rate = 5e0
offs=1e1

N=200

X = []
Y = []


n_tot=100000000

x=[3,5]

for t in range(n_tot):

    [x,E_tot,dx_1]=gradient(x,offs,learning_rate)

    if (t % 1000 == 0):

        print(t,x) #, d_x1,d_x2)
        f_Etot.write(str(t)+' '+str(E_tot)+'\n')
        for ii in range(d):
            f_para.write(str(t)+' '+str(x[ii])+'\n')

