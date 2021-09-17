import numpy as np
import numpy.random as rd
import matplotlib.pyplot as plt




def d_pos_calcul(sys,pos):

    n_atoms = sys[0]
    eps = sys[1]
    sig = sys[2]

    pos1 = pos[0]
    pos2 = pos[1]
    pos3 = pos[2]

    d_pos1 = []
    d_pos2 = []
    d_pos3 = []

    E_i = []

    for i in range(n_atoms):

        E_i_ii =0

        for ii in range(i+1,n_atoms):
            dx = pos1[ii]-pos1[i]
            dy = pos2[ii]-pos2[i]
            dz = pos3[ii]-pos3[i]

            dr = np.sqrt(dx**2+ dy**2+ dz**2)
        #    print(ii,i,dr)
            r6 = dr**6
            r12 = dr**12
            E_i_ii += 4*eps*(-sig**6/r6+sig**12/r12)



        E_i.append(E_i_ii)
        #print(E_i)

        dE_i_ii = 0

        dE_x = 0
        dE_y = 0
        dE_z = 0




        for ii in range(i+1,n_atoms):
            dx = pos1[ii] - pos1[i]
            dy = pos2[ii] - pos2[i]
            dz = pos3[ii] - pos3[i]

            dr = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)

            r6 = dr ** 6
            r12 = dr ** 12

            dE_i_ii = 4 * eps * (-6*sig**6 / r6 + 12*sig**12 / r12) * (-1 / dr)

            dE_x += 0.5 * dE_i_ii / dr * 2 *  (1 + dx) *dx
            dE_y += 0.5 * dE_i_ii / dr * 2 *  (1 + dy) *dy
            dE_z += 0.5 * dE_i_ii / dr * 2 * (1 + dz) *dz


        d_pos1.append(dE_x) # * eta
        d_pos2.append(dE_y) # * eta
        d_pos3.append(dE_z) # * eta

    return d_pos1, d_pos2, d_pos3, E_i



n_it = int(1e4)

n_atoms = 55
d_atom = 0.7

file = open('input.dat','r')
input_lines = file.readlines()
b = [i.split() for i in input_lines]

pos1=[]
pos2=[]
pos3=[]
for i in range(len(input_lines)):
    bb=b[i]
    pos1.append(float(bb[1]))
    pos2.append(float(bb[2]))
    pos3.append(float(bb[3]))

eps = 1e-4
sig = 2.6

eta = -1e13
E0 = -10000
off = 1e9

sys=[]
sys.append(n_atoms)
sys.append(eps)
sys.append(sig)

pos=[0,0,0]

out_file = open('coord.xyz','w')
for i in range(1000000):

    pos[0] = pos1
    pos[1] = pos2
    pos[2] = pos3

    d_pos1,d_pos2,d_pos3, E_i = d_pos_calcul(sys, pos)

    E = np.asarray(E_i)
    d_Etot = E.sum()/2 - E0



    grad_0 = np.log(d_Etot+off) - np.log(E0+off)
    grad = grad_0 /(d_Etot+off)#*np.ones(n_atoms)

   # print(type(pos1))
    pos1 -= np.asarray(d_pos1) * eta * grad#[1:n_atoms]
    pos2 -= np.asarray(d_pos2) * eta * grad#[1:n_atoms]
    pos3 -= np.asarray(d_pos3) * eta * grad#[1:n_atoms]


    if (i%100 ==0):
        print(i,  d_Etot)
        out_file.write(str(n_atoms)+'\n')
        out_file.write('atoms'+'\n')
        for ii in range(n_atoms):
            out_file.write(str(ii)+' '+str(pos1[ii])+' '+str(pos2[ii])+ ' ' + str(pos3[ii])+'\n')
        print(pos1)