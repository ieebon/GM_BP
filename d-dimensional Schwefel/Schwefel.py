# -*- coding: utf-8 -*-
#CODE originally from https://pytorch.org/tutorials/beginner/pytorch_with_examples.html#warm-up-numpy
# will be modified to fit the parameter in LJ potential energy function 
#import numpy as np
#import Class_Ebond as Eb
#import Class_Evdw as vdw

#import Class_Evdw as Evdw, Class_parameter_initial as Cp, Class_Ebond as Eb, \
#    Class_parameter_fitting as Cpf, Class_Eover_under as EU
import numpy as np
import random as rd



x=np.arange(-1.0,1.,0.1)*5.12

x_rand=[]
x_rand_b=[]

for i in range(5):
    x_rand.append((rd.randrange(5)-10)*2.048/5)
    x_rand_b.append((rd.randrange(5)-10)*2.048/5)


print(x_rand)

x_rand = np.asarray(x_rand)

d = len(x_rand)
y = np.zeros(d)

f_para=open('para.dat','w')
f_cost=open('cost.dat','w')
f_Etot=open('Etot.dat','w')

learning_rate0 =5e-6
learning_rate_=1*learning_rate0
learning_rate = learning_rate0

offs=1e3

print(np.cos(5.12))
x=x_rand
for t in range(5000000):

    x_2 = np.sqrt(np.abs(x))
    x_i_sin =  x*np.sin(x_2)
    F_x_rand = 418.9829*d - x_i_sin.sum() #x_2[0:d-1].sum() + x_i_sq.sum() #10*d + x_2.sum() - 10*np.cos(2*3.14*x_rand)

    E_tot=F_x_rand

    h = np.log(E_tot + offs)
    y_corre = np.log(y + offs)
    y_00=(h/y_corre)


    #for ii in range(d):
    y_=y_00-1


    cost_2 = np.square(y_00).sum()

    y_pred = -2*y_**2

    grad_=y_pred/(E_tot+offs)/y_corre*y_#
    grad_h9 = grad_#*2*y_00 #
    d_f_x_rand = -1 - x*np.cos(x_2)/x_2*0.5 #2*(x_rand[0:d-1]**2 - x_rand[1:d])*x_rand[0:d-1] #2*x_rand - 10#*2*3.14/#*np.sin(2*3.14*x_rand)
 #   print('sin',np.sin(x_2))
    d_max = d_f_x_rand.max()
    d_min = d_f_x_rand.min()

    d_grad = d_max-d_min

    a_max= np.argmax(d_f_x_rand)


    x_new = x_rand[0:d-1] + d_f_x_rand[0:d-1] * learning_rate * x_rand[0:d-1]
    x_rand[0:d-1] = x_new


    if (t % 1000 == 0):

        f_cost.write(str(t)+' '+str(cost_2)+'\n')

       # for ii in range(d):
        f_Etot.write(str(t)+' '+str(E_tot)+'\n')
        for ii in range(d):
            f_para.write(str(t)+' '+str(x_rand[ii])+'\n')

        print('x_rand',x_rand)

