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

learning_rate0 = -1e-2
learning_rate_=1*learning_rate0
learning_rate = learning_rate0

offs=1e3

#ç # x[2], lr=5e-5
#x_rand =[1.5860058360150593, 12.68349277392557, 0.30520080228269, -0.556087758622959, 1.225358800141429] #x[3], lr=5e-5
#x_rand =[51.96808186166077, 1.2942182535612115, 5.975194127329363, -0.556087758622959, 0.5086760360106641] # x[4]
#x_rand =[2.0621941407096465, 12.68349277392557, 0.28981764787905334, -0.7205543504294568, 1.241769571736572] #x[0]

#x_rand = [4.9961038136469593e-29, 9.15713424546432, 0.2879988892636841, 0.3680294220760386, -19.2288286664365] / 5e5

x_rand =[0.2,0.2,0.2,0.2,0.2]
#x_rand [53.81025872376107, 1.269996937921439, 3.011592875368144, 2.1303954383576227, 0.5069998109336532]
x=x_rand
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

    y_pred = 1/(1+np.exp(-y_00))

    grad_=y_pred/(E_tot+offs)#*y_00 #/y_corre
    grad_h9 = grad_#*2*y_00 #
    grad_h90=y_00/(E_tot+offs)

    d_f_x1=0
    d_f_x2=0
    d_f_x3=0
    d_f_x4=0
    d_f_x5=0

    d_f_x1_=0
    d_f_x2_=0
    d_f_x3_=0
    d_f_x4_=0
    d_f_x5_=0

    d_f_x1__=0
    d_f_x2__=0
    d_f_x3__=0
    d_f_x4__=0
    d_f_x5__=0

    d_f_x_common_i = 0
    d_f_x_common0_i = 0

    for ii in range(24):
        t_i = 0.1*(ii-1)
        y_i = 53.81*(1.27**t_i)*np.tanh(3.012*t_i+np.sin(2.13*t_i))*np.cos((2.718**0.507)*t_i)
        f_x_i = x[0]*x[1]**t_i*np.tanh(x[2]*t_i+np.sin(x[3]*t_i))*np.cos(t_i*2.718**x[4])-y_i



        d_f_x1_ +=  f_x_i*((x[1] ** t_i * np.tanh(x[2] * t_i + np.sin(x[3] * t_i)) * np.cos(t_i * 2.718 ** x[4])))
        # d_f_x2 += d_f_x_common * np.exp(np.log(x[1])*(t_i)) *np.log(x[1])*x[1]  *x[0] * np.tanh(x[2]*t_i+np.sin(x[3]*t_i))*np.cos(t_i*2.718**x[4])
        d_f_x2_ +=  f_x_i*(x[1]) ** (t_i-1) * t_i * (
                    1 ) * np.cos(t_i * 2.718 ** x[4])
        d_f_x3_ += f_x_i / np.cosh(x[2] * t_i + np.sin(x[3] * t_i)) ** 2 * (x[2] * t_i) * (
                    1 )
        d_f_x4_ += f_x_i / np.cosh(x[2] * t_i + np.sin(x[3] * t_i)) ** 2 * np.cos(x[3] * t_i) * (x[3] * t_i) * (
                    1 )
        d_f_x5_ += f_x_i* (-np.sin(t_i * np.exp(x[4]))) * t_i * np.exp(x[4]) * (x[4]) * (
                    1 )

      #  y_i_t_i = 53.81*(1.27**t_i)*np.log(1.27)*np.tanh(3.012*t_i+np.sin(2.13*t_i))*np.cos((2.718**0.507)*t_i)
      #  y_i_t_i += 53.81*(1.27**t_i)/np.cosh(3.012*t_i+np.sin(2.13*t_i))**2*np.cos((2.718**0.507)*t_i)*(3.012+np.cos(2.13*t_i))
      #  y_i_t_i += 53.81*(1.27**t_i)*np.tanh(3.012*t_i+np.sin(2.13*t_i))*(-np.sin((2.718**0.507)*t_i))*(2.718**0.507)


        d_f_x_common_i += f_x_i
        d_f_x_common0_i += f_x_i

    d_f_x_common = grad_h9
    d_f_x_common0 = grad_h90

    for ii in range(24):
        t_i = 0.1 * (ii - 1)
        y_i = 53.81 * (1.27 ** t_i) * np.tanh(3.012 * t_i + np.sin(2.13 * t_i)) * np.cos((2.718 ** 0.507) * t_i)
        f_x_i = x[0] * x[1] ** t_i * np.tanh(x[2] * t_i + np.sin(x[3] * t_i)) * np.cos(t_i * 2.718 ** x[4]) - y_i

       # print(f_x_i)

        d_f_x1 += f_x_i * (
                    (x[1] ** t_i * np.tanh(x[2] * t_i + np.sin(x[3] * t_i)) * np.cos(t_i * 2.718 ** x[4])))
        # d_f_x2 += d_f_x_common * np.exp(np.log(x[1])*(t_i)) *np.log(x[1])*x[1]  *x[0] * np.tanh(x[2]*t_i+np.sin(x[3]*t_i))*np.cos(t_i*2.718**x[4])
        d_f_x2 += f_x_i * (x[1]) ** (t_i-1) * t_i * (
                    x[0] * np.tanh(x[2] * t_i + np.sin(x[3] * t_i)) * np.cos(t_i * 2.718 ** x[4]))
        d_f_x3 += f_x_i / np.cosh(x[2] * t_i + np.sin(x[3] * t_i)) ** 2 * ( t_i) * (
                     x[0] * x[1] ** t_i * np.cos(t_i * 2.718 ** x[4]))
        d_f_x4 += f_x_i / np.cosh(x[2] * t_i + np.sin(x[3] * t_i)) ** 2 * np.cos(x[3] * t_i) * ( t_i) * (
                     x[0] * x[1] ** t_i * np.cos(t_i * 2.718 ** x[4]))
        d_f_x5 += f_x_i * (-np.sin(t_i * np.exp(x[4]))) * t_i * np.exp(x[4])  * (
                     x[0] * np.tanh(x[2] * t_i + np.sin(x[3] * t_i)) * (x[1] ** t_i))


        d_f_x1__ += d_f_x1_*x[0]#*f_x_i
        d_f_x2__ += d_f_x2_*x[1]**t_i*t_i#*f_x_i
        d_f_x3__ += d_f_x3_* (x[2] * t_i)#*f_x_i
        d_f_x4__ += d_f_x4_* (x[3] * t_i)#*f_x_i
        d_f_x5__ += d_f_x5_* t_i * np.exp(x[4]) * (x[4])#*f_x_i



       # print('checking',d_f_x1_)
    x[0] += d_f_x1__ * learning_rate * d_f_x_common +  d_f_x1 * learning_rate*grad_h9 #* d_f_x_common0
    x[1] += d_f_x2__ * learning_rate * d_f_x_common +  d_f_x2 * learning_rate*grad_h9 #* d_f_x_common0
    x[2] += d_f_x3__ * learning_rate * d_f_x_common +  d_f_x3 * learning_rate*grad_h9 #* d_f_x_common0
    x[3] += d_f_x4__ * learning_rate * d_f_x_common +  d_f_x4 * learning_rate*grad_h9 #* d_f_x_common0
    x[4] += d_f_x5__ * learning_rate * d_f_x_common +  d_f_x5 * learning_rate*grad_h9 #* d_f_x_common0

    if(x[1]<0):
        x[1]=0.1

    if (x[1]>60):
        x[1] =60

    if (t % 1000== 0):

        print(t, x, E_tot)
     #   print(d_f_x1__)

        f_cost.write(str(t) + ' ' + str(cost_2) + '\n')

        # for ii in range(d):
        f_Etot.write(str(t) + ' ' + str(E_tot) + '\n')
        for ii in range(d):
            f_para.write(str(t) + ' ' + str(x[ii]) + '\n')

        print('x_rand', x_rand)
'''
    if x[4] < -5:
        x[4] = -4

    if x[1] < 0.001:
        x[1] = 0.001

    if x[1] > 50:
        x[1] = 50

    if x[0] < 0.1:
        x[0] = 0.1
'''



