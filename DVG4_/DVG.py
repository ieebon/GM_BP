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

learning_rate0 = 5e-4
learning_rate_=1*learning_rate0
learning_rate = learning_rate0

offs=1e2

x_rand = [9.173704255512332, 5.382779212429532, 45.30520080228269, 35.3843227490205, 0.04675149033479609]

#x_rand= [105.6545279695097, 22.183096796441383, 45.3052316721752, 35.38428240266805, 0.37612974581728614]
#x_rand = [39.300200588770856, 1.507711973491663, 44.391848861127485, 36.0453563083087, 0.5237652823340704]
#x_rand = [39.28949668885517, 1.5075122206424372, 44.391892215173215, 36.0453563083087, 0.5237726392467215]
#x_rand = [53.132587259539434, 1.2041948256689129, 3.01132004837687, 36.0451037578504, 0.507237726392467215]
#x_rand = [54.42184188753683, 1.2620262505902886, 4.921840125268107, 31.419452871631982, 0.5065474827358899]
#x_rand = [53.503895628199515, 1.2737888628713943, 5.124003897592866, 16.41547963055705, 0.507342547575498]
#x_rand = [53.803094814436335, 1.2711258911226314, 3.023239426396416, 36.0451037578504, 0.5024535342339431]
#x_rand =  [4.647931380105112, 5.382779212429532, 45.30816519186606, 35.38218757040482, 0.6776014649257615]
#x_rand =[23.71823042801518, 2.1497807289853776, 44.78441923335439, 35.76062199435905, 0.6776014649257615]
#x_rand =  [23.71823042801518, 1.9953365496962008, 44.35674176347167, 36.07098874417749, 0.5605384463915973]
#x_rand = [39.24306951793171, 1.508918098216628, 45.30816519186606, 35.45125956941425, 0.5238529554126462]
#x_rand = [39.298117570346314, 1.50775581343215, 44.74196258350892, 35.45125956941425, 0.5237685324302346]
#x_rand =  [50.837617020023956, 1.310494986551664, 9.365438057873861, 58.80552168712108, 0.5237685324302346]
#x_rand = [50.837617020023956, 1.3096745015836577, 9.17937138041349, 59.01024272748172, 0.5098925271342408]
#x_rand =  [52.05520325878219, 1.2927352998200456, 9.17937138041349, 58.79617666856368, 0.5088956183169685]
#x_rand = [52.90409428585353, 1.2815594504826262, 7.600102708733627, 60.40466437713232, 0.5088956183169685]
#x_rand = [53.18340991199258, 1.2779559896526702, 6.5917627551008575, 61.41976156335467, 0.5088956183169685]

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
    y_=-y_00


    cost_2 = np.square(E_tot-y).sum()

    #if (y_ > 0): y_pred = -2*np.abs(y_)
    y_pred = y_

    grad_=y_pred/(E_tot+offs)#/y_corre
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

       # d_f_x1 += d_f_x_common
        d_f_x1_ = d_f_x_common
        d_f_x1 += d_f_x_common * (x[1]**t_i*np.tanh(x[2]*t_i+np.sin(x[3]*t_i))*np.cos(t_i*2.718**x[4]) )*(1-d_f_x1_)

        d_f_x2_ = d_f_x_common* x[0]  * np.tanh(x[2] * t_i + np.sin(x[3] * t_i)) * np.cos(t_i * 2.718 ** x[4])
        d_f_x2 += d_f_x_common * x[0] *x[1]**(t_i-1)*t_i* np.tanh(x[2] * t_i + np.sin(x[3] * t_i)) * np.cos(t_i * 2.718 ** x[4])*(1-d_f_x2_)

       # d_f_x3 -= d_f_x_common *x[0]*x[1]**t_i/ np.cosh(x[2]*t_i+np.sin(x[3]*t_i))**2 *np.cos(t_i*2.718**x[4]) *t_i
        d_f_x3_ = d_f_x_common *x[0]*x[1]**t_i *np.cos(t_i*2.718**x[4])
        d_f_x3 += d_f_x_common *x[0]*x[1]**t_i/ np.cosh(x[2]*t_i+np.sin(x[3]*t_i))**2 *np.cos(t_i*2.718**x[4]) *(1-d_f_x3_)#* (x[2]+1)

        d_f_x4_ = d_f_x_common *x[0]*x[1]**t_i  *np.cos(t_i*2.718**x[4])
        d_f_x4 += d_f_x_common *x[0]*x[1]**t_i/ np.cosh(x[2]*t_i+np.sin(x[3]*t_i))**2 *np.cos(x[3]*t_i)*(1-d_f_x4_) *np.cos(t_i*2.718**x[4])


        d_f_x5_ = d_f_x_common * x[0] * np.tanh(x[2] * t_i + np.sin(x[3] * t_i)) * (x[1] ** t_i) *np.cos(t_i*2.718**x[4])
        d_f_x5_ = d_f_x_common * x[0] * np.tanh(x[2] * t_i + np.sin(x[3] * t_i)) * (x[1] ** t_i) *np.cos(t_i*2.718**x[4])
#        d_f_x5 += d_f_x_common * (-np.sin(t_i*np.exp(x[4])))*t_i*(np.exp(x[4])*x[4])*x[0] *np.tanh(x[2]*t_i+np.sin(x[3]*t_i))*(x[1]**t_i)*(1-d_f_x5_)
        d_f_x5 += d_f_x_common * (-np.sin(t_i * np.exp(x[4]))) * t_i * np.exp(x[4])  * x[0] * np.tanh( x[2] * t_i + np.sin(x[3] * t_i)) * (x[1] ** t_i)*(1-d_f_x5_)

    x[0] += d_f_x1 * learning_rate
    x[1] += d_f_x2 * learning_rate
    x[2] += d_f_x3 * learning_rate
   # x[3] += d_f_x4 * learning_rate
    x[4] += d_f_x5 * learning_rate


 #   if(d_f_x1<0):
 #       x[0] += d_f_x1 * learning_rate
 #   else: x[0] -= d_f_x1 * learning_rate


    #x[4] += d_f_x5 * learning_rate


    if (t % 1000 == 0):

        print(t,x,E_tot,cost_2)

        f_cost.write(str(t)+' '+str(cost_2)+'\n')

       # for ii in range(d):
        f_Etot.write(str(t)+' '+str(E_tot)+'\n')
        for ii in range(d):
            f_para.write(str(t)+' '+str(x[ii])+'\n')

        print('x_rand',x_rand, d_f_x3, d_f_x4)

