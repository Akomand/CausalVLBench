
import matplotlib.pyplot as plt
import os

import matplotlib.image as mpimg
import random
import math
import numpy as np
import pandas as pd


def projection(theta, phi, x, y, base=-0.5):
    b = y - x * math.tan(phi)
    shade = (base - b) / math.tan(phi)
    return shade


scale = np.array([[0, 44], [100, 40], [7, 7.5], [10, 10]])
count = 0



def everything(i, j, mid=None, shade=None, var='orig', origin=None):
    plt.rcParams['figure.figsize'] = (1.0, 1.0)
    theta = i * math.pi / 200.0
    phi = j * math.pi / 200.0
    x = 10 + 8 * math.sin(theta)
    y = 10.5 - 8 * math.cos(theta)

    # calculate the mid index of
    ball_x = 10 + 9.5 * math.sin(theta)
    ball_y = 10.5 - 9.5 * math.cos(theta)

    if mid:
        mid = mid
    else:
        mid = (projection(theta, phi, 10.0, 10.5) + projection(theta, phi, ball_x, ball_y)) / 2

    if shade:
        shade = shade
    else:
        shade = max(3, abs(projection(theta, phi, 10.0, 10.5) - projection(theta, phi, ball_x, ball_y)))



    ball = plt.Circle((x, y), 1.5, color='firebrick')
    gun = plt.Polygon(([10, 10.5], [x, y]), color='black', linewidth=3)

    light = projection(theta, phi, 10, 10.5, 20.5)
    sun = plt.Circle((light, 20.5), 3, color='orange')

    shadow = plt.Polygon(([mid - shade / 2.0, -0.5], [mid + shade / 2.0, -0.5]), color='black', linewidth=3)

    ax = plt.gca()
    ax.add_artist(gun)
    ax.add_artist(ball)
    ax.add_artist(sun)
    ax.add_artist(shadow)
    ax.set_xlim((0, 20))
    ax.set_ylim((-1, 21))
    
    plt.axis('off')
    
    if origin:
        origin = origin
    else:
        origin = 'a_' + str(int(i)) + '_' + str(int(j)) + '_' + str(int(shade)) + '_' + str(
                int(mid)) + '_' + str(var)

    if var == 'orig':
        if not os.path.exists('./new_data/pendulum/train/a_'+ str(int(i)) + '_' + str(int(j)) + '_' + str(int(shade)) + '_' + str(
                int(mid)) + '_' + str(var)):
            os.makedirs('./new_data/pendulum/train/a_' + str(int(i)) + '_' + str(int(j)) + '_' + str(int(shade)) + '_' + str(
                int(mid)) + '_' + str(var))

    plt.savefig(
        './new_data/pendulum/train/' + origin + '/a_' + str(int(i)) + '_' + str(int(j)) + '_' + str(int(shade)) + '_' + str(
            int(mid)) + f'_{str(var)}.png', dpi=96)

    plt.clf()

    return origin, shade, mid



for i in range(-40, 44):  # pendulum
    for j in range(60, 148):  # light
        # if j == 100:
        #     continue

        orig, shade, mid  = everything(i, j)

        for k in range(4):
            if k == 0:
                if i < -6:
                    i_new = np.random.randint(6, 44)
                elif i > 6:
                    i_new = np.random.randint(-44, -6)

                _ = everything(i_new, j, var=str(k), origin=orig)
            if k == 1:
                if j < 100:
                    j_new = np.random.randint(105, 148)
                elif j > 100:
                    j_new = np.random.randint(60, 95)

                _ = everything(i, j_new, var=str(k), origin=orig)
            if k == 2:
                if shade < 7:
                    shade = np.random.randint(8, 12)
                elif shade > 7:
                    shade = np.random.randint(3, 6)

                _ = everything(i, j, shade=shade, var=str(k), origin=orig)
            if k == 3:
                if mid < 8:
                    mid = np.random.randint(10, 15)
                elif shade > 8:
                    mid = np.random.randint(3, 7)

                _ = everything(i, j, mid=mid, var=str(k), origin=orig)


