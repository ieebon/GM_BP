# -*- coding: utf-8 -*-


import numpy as np
import random as rd



x=np.arange(-1.0,1.,0.1)*5.12

x_rand=[]
x_rand_b=[]

for i in range(5):
    x_rand.append(np.float(rd.randrange(59)+1))
#    x_rand_b.append((rd.randrange(5)-10)*2.048/5)


print(x_rand)

x_rand = np.asarray(x_rand)

d = len(x_rand)
y = np.zeros(d)

f_para=open('para.dat','w')
f_cost=open('cost.dat','w')
f_Etot=open('Etot.dat','w')

learning_rate0 = 5e-5
learning_rate_=1*learning_rate0
learning_rate = learning_rate0

offs=1e1

x_rand = [9.173704255512332, 5.382779212429532, 45.30520080228269, 35.3843227490205, 0.04675149033479609]
x_rand = [21.805471728535185, 2.4606749327370685, 43.670606818795655, 36.57045330292465, 0.04675149033479609]
x_rand = [49.428593002264705, 1.3294233275751366, 9.859963209654568, 36.57045330292465, 0.5114514546613164]
x_rand = [49.70684979241903, 1.3294233275751366, 5.812118885641318, 1.512944703643793, 0.5108403513836247]
x = x_rand #[10.57, 1.22, 46.00, 34.99, 1]
#2 x = [24.13, 2.07, 45.012, 35.07, 0.63]
#3 x = [77.83, 1.00, 3.125, 62.812, 0.63]
#4 x = [77.83, 1.00, 2.809, 62.812, 0.49]
#x = [82.83, 1.2, 2.79, 62.812, 0.49] # 이 상태에서 learning_rate = 5e-5 등의 값을 사용시 피팅없이 정체


for t in range(10000000):


    f_x_sq = 0

    for ii in range(24):
        #print(i)
        t_i = 0.1*(ii-1)
        y_i = 53.81*(1.27**t_i)*np.tanh(3.012*t_i+np.sin(2.13*t_i))*np.cos((2.718**0.507)*t_i)
       # print(np.tanh(x[2]*t_i+np.sin(x[3]*t_i)))
        f_x_i = x[0]*(x[1]**t_i)*np.tanh(x[2]*t_i+np.sin(x[3]*t_i))*np.cos(t_i*2.718**x[4])-y_i
        f_x_sq += f_x_i**2


    E_tot=f_x_sq
    y=0

    h = np.log(E_tot + offs)
    y_corre = np.log(y + offs)
    y_00=(h-y_corre)


    #for ii in range(d):
    y_=y_00**2


    cost_2 = np.square(y_00).sum()

    y_pred = -2*y_

    grad_=y_pred/(E_tot+offs)*y_00 #/y_corre
    grad_h9 = grad_#*2*y_00 #

    d_f_x1=0
    d_f_x2=0
    d_f_x3=0
    d_f_x4=0
    d_f_x5=0

    for ii in range(24):
        t_i = 0.1*(ii-1)
        y_i = 53.81*(1.27**t_i)*np.tanh(3.012*t_i+np.sin(2.13*t_i))*np.cos((2.718**0.507)*t_i)
        f_x_i = x[0]*x[1]**t_i*np.tanh(x[2]*t_i+np.sin(x[3]*t_i))*np.cos(t_i*2.718**x[4])-y_i

        y_i_t_i = 53.81*(1.27**t_i)*np.log(1.27)*np.tanh(3.012*t_i+np.sin(2.13*t_i))*np.cos((2.718**0.507)*t_i)
        y_i_t_i += 53.81*(1.27**t_i)/np.cosh(3.012*t_i+np.sin(2.13*t_i))**2*np.cos((2.718**0.507)*t_i)*(3.012+np.cos(2.13*t_i))
        y_i_t_i += 53.81*(1.27**t_i)*np.tanh(3.012*t_i+np.sin(2.13*t_i))*(-np.sin((2.718**0.507)*t_i))*(2.718**0.507)


        d_f_x_common = f_x_i*grad_h9

        d_f_x1 += d_f_x_common * (x[1]**t_i*np.tanh(x[2]*t_i+np.sin(x[3]*t_i))*np.cos(t_i*2.718**x[4]) )* x[0]
        #d_f_x2 += d_f_x_common * np.exp(np.log(x[1])*(t_i)) *np.log(x[1])*x[1]  *x[0] * np.tanh(x[2]*t_i+np.sin(x[3]*t_i))*np.cos(t_i*2.718**x[4])
        d_f_x2 += d_f_x_common * x[0]*np.exp(np.log(x[1])*t_i)*t_i*np.tanh(x[2]*t_i+np.sin(x[3]*t_i))*np.cos(t_i*2.718**x[4])
        d_f_x3 += d_f_x_common *x[0]*x[1]**t_i/ np.cosh(x[2]*t_i+np.sin(x[3]*t_i))**2 * x[2]  *np.cos(t_i*2.718**x[4])
        d_f_x4 += d_f_x_common *x[0]*x[1]**t_i/ np.cosh(x[2]*t_i+np.sin(x[3]*t_i))**2 *np.cos(x[3]*t_i)*x[3] *np.cos(t_i*2.718**x[4])
        d_f_x5 += d_f_x_common * (-np.sin(t_i*np.exp(x[4])))*t_i*np.exp(x[4])*x[4]*x[0] *np.tanh(x[2]*t_i+np.sin(x[3]*t_i))*(x[1]**t_i)

    x[0] += d_f_x1 * learning_rate
    x[1] += d_f_x2 * learning_rate
    x[2] += d_f_x3 * learning_rate
    #x[3] += d_f_x4 * learning_rate
    x[4] += d_f_x5 * learning_rate


    if (t % 1000 == 0):

        print(t,x,E_tot)

        f_cost.write(str(t)+' '+str(cost_2)+'\n')

       # for ii in range(d):
        f_Etot.write(str(t)+' '+str(E_tot)+'\n')
        for ii in range(d):
            f_para.write(str(t)+' '+str(x[ii])+'\n')

        print('x_rand',x_rand)

