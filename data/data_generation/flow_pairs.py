
import matplotlib.pyplot as plt
from PIL import Image
import os
import matplotlib.image as mpimg
import random
import math
import numpy as np
count=0

if not os.path.exists('./new_data/flow/'): 
    os.makedirs('./new_data/flow/train/')
    os.makedirs('./new_data/flow/test/')


def sample_excluding_current(values, current):
    options = [v for v in values if v != current]
    return random.choice(options)


def everything(r, h_raw, hole, h_int=None, flow=None, var="orig", origin=None):
    ball_r = r/30.0

    if h_int:
        h = h_int
    else:
        h = pow(ball_r,3)+h_raw/10.0 

    deep = hole/3.0
    plt.rcParams['figure.figsize'] = (1.0, 1.0)
    ax = plt.gca()
    

    # water in cup 
    rect = plt.Rectangle(([3, 0]),5,5+h,color='lightskyblue')
    ax.add_artist(rect)
    ball = plt.Circle((5.5,+ball_r+0.5), ball_r, color = 'firebrick')
    ## cup
    left = plt.Polygon(([3, 0],[3, 19]), color = 'black', linewidth = 2)
    right_1 = plt.Polygon(([8, 0],[8, deep]), color = 'black', linewidth = 2)
    right_2 = plt.Polygon(([8, deep+0.4],[8, 19]), color = 'black', linewidth = 2)
    ax.add_artist(left)
    ax.add_artist(right_1)
    ax.add_artist(right_2)
    ax.add_artist(ball)

    #water line
    y = np.linspace(deep,0.5)

    epsilon = 0.01 * max(abs(np.random.randn()), 1)
    x = np.sqrt(2*(0.98+epsilon)*h*(deep-y))+8

    x_max = x[-1]-8
    x_true = np.sqrt(2*(0.98)*h*(deep-0.5))

    plt.plot(x,y,color='lightskyblue',linewidth = 2)

    ##ground
    x = np.linspace(0,20,num=50)
    y = np.zeros(50)+0.2
    plt.plot(x,y,color='black',linewidth = 2)
    
    ax.set_xlim((0, 20))
    ax.set_ylim((0, 20))

    plt.axis('off')

    if origin:
        origin = origin
    else:
        origin = 'a_' + str(int(r)) + '_' + str(int(hole)) + '_' + str(int(h)) + '_' + str(
                int(x_true*10)) + '_' + str(var)

    if var == 'orig':
        if not os.path.exists('./new_data/flow/train/a_'+ str(int(r)) + '_' + str(int(hole)) + '_' + str(int(h)) + '_' + str(
                int(x_true*10)) + '_' + str(var)):
            os.makedirs('./new_data/flow/train/a_' + str(int(r)) + '_' + str(int(hole)) + '_' + str(int(h)) + '_' + str(
                int(x_true*10)) + '_' + str(var))

    plt.savefig(
        './new_data/flow/train/' + origin + '/a_' + str(int(r)) + '_' + str(int(hole)) + '_' + str(int(h)) + '_' + str(
                int(x_true*10)) + f'_{str(var)}.png', dpi=96)

    plt.clf()

    return origin, h, x_true*10


for r in range(5, 35):
    for h_raw in range(10, 40):
        for hole in range(6, 15):
            orig, h_orig, f_orig = everything(r, h_raw, hole)

            for k in range(4): 
                if k == 0:
                    if r < 20:   
                        r_new = np.random.randint(20, 35)
                    else:
                        r_new = np.random.randint(5, 20)
                    _, h_d, f = everything(r_new, h_raw, hole, var=str(k), origin=orig)
                if k == 1:
                    if hole < 11:
                        hole_new = np.random.randint(11, 15)
                    else:
                        hole_new = np.random.randint(6, 11)
                    _, h_d, f = everything(r, h_raw, hole_new, var=str(k), origin=orig)
                if k == 2:
                    # print(h_orig)
                    # exit(0)
                    if h_orig > 1 and h_orig < 2:
                        h_new = np.random.uniform(2, 4)
                    elif h_orig > 2 and h_orig < 3:
                        h_new = np.array(sample_excluding_current([1, 3], 2))
                    elif h_orig > 3:
                        h_new = np.random.uniform(1, 3)
                    _, h_d, f = everything(r, h_raw, hole, h_int=h_new, var=str(k), origin=orig)







    
