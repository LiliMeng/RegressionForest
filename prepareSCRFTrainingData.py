# -*- coding: utf-8 -*-
"""
Created on Wed Mar 30 15:33:52 2016

@author: jimmy&Lili

prepare SCRF training data
"""
import random
import numpy as np

color_image_names = []
depth_image_names = []
pose_names = []

path = '/Users/jimmy/Desktop/images/7_scenes/pumpkin/seq-01/'
for i in range(0, 1000):
    buf1 = 'frame-%06d.color.png' % i
    buf2 = 'frame-%06d.depth.png' % i
    buf3 = 'frame-%06d.pose.txt' % i
    color_image_names.append(path + buf1)
    depth_image_names.append(path + buf2)
    pose_names.append(path + buf3)
    
"""
path = '/Users/jimmy/Desktop/images/7_scenes/pumpkin/seq-03/'
for i in range(0, 1000):
    buf1 = 'frame-%06d.color.png' % i
    buf2 = 'frame-%06d.depth.png' % i
    buf3 = 'frame-%06d.pose.txt' % i
    color_image_names.append(path + buf1)
    depth_image_names.append(path + buf2)
    pose_names.append(path + buf3)

path = '/Users/jimmy/Desktop/images/7_scenes/pumpkin/seq-04/'
for i in range(0, 1000):
    buf1 = 'frame-%06d.color.png' % i
    buf2 = 'frame-%06d.depth.png' % i
    buf3 = 'frame-%06d.pose.txt' % i
    color_image_names.append(path + buf1)
    depth_image_names.append(path + buf2)
    pose_names.append(path + buf3)

path = '/Users/jimmy/Desktop/images/7_scenes/pumpkin/seq-08/'
for i in range(0, 1000):
    buf1 = 'frame-%06d.color.png' % i
    buf2 = 'frame-%06d.depth.png' % i
    buf3 = 'frame-%06d.pose.txt' % i
    color_image_names.append(path + buf1)
    depth_image_names.append(path + buf2)
    pose_names.append(path + buf3)

assert(len(color_image_names) == len(depth_image_names))
assert(len(color_image_names) == len(pose_names))
"""

N = len(color_image_names)

index = range(0,N)
random.shuffle(index)

index = index[0:200]
print('sampled frame number is %d' % len(index))

sampled_color_image_names = []
sampled_depth_image_names = []
sampled_pose_names = []

for item in index:
    sampled_color_image_names.append(color_image_names[item])
    sampled_depth_image_names.append(depth_image_names[item])
    sampled_pose_names.append(pose_names[item])

f = file('rgb_image_list.txt', 'w')
for item in sampled_color_image_names:
    f.write('%s\n' % item)
f.close()

f = file('depth_image_list.txt', 'w')
for item in sampled_depth_image_names:
    f.write('%s\n' % item)
f.close()

f = file('camera_pose_list.txt', 'w');
for item in sampled_pose_names:
    f.write('%s\n' % item)
f.close()


    

