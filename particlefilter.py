# -*- coding: utf-8 -*-
"""
Created on Thu Apr 27 06:39:55 2017

@author: monica
"""
import cv2
import random


img1_name = 'o1.png'

img1 = cv2.imread(img1_name)          
disp_img = cv2.imread(img1_name)

height, width, channels =  img1.shape 
print height, width, channels

MAX_PAR = 5000
mask_m = 5
pad_size =10
velocity = [0,0]
p_list = []
w_list = []#[0] * MAX_PAR
inp_c = 1
out_c = 1

img1= cv2.copyMakeBorder(img1,pad_size,pad_size,pad_size,pad_size,cv2.BORDER_CONSTANT,value=(0,0,0))
disp_img= cv2.copyMakeBorder(disp_img,pad_size,pad_size,pad_size,pad_size,cv2.BORDER_CONSTANT,value=(0,0,0))

class Particle(object):
    def __init__(self,x,y):
        self.point = (x,y)
        self.weight = 0
        self.color = (0,0,0)
        self.deltaT = 1
        
    def move(self,v):
        newX = (self.point[0] + self.deltaT * v[0]) % (width-1)
        newY = (self.point[1] + self.deltaT * v[1]) % (height-1)
        self.point = (newX,newY)
        
    def update_weight(self,I_diff):
        weight = 0
        for i in range(-int(mask_m/2),int(mask_m/2+1)):
            for j in range(-int(mask_m/2),int(mask_m/2+1)):
                #print self.point[0]-i,self.point[1]-j
                weight += I_diff[self.point[1]-j,self.point[0]-i]/255
                
        weight = weight / float(mask_m**2)
        self.weight = weight[0]


for i in range(MAX_PAR):
    x = random.randint(0,width)
    y = random.randint(0,height)
    p = Particle(x,y)
    p.color = (0,0,255)
    p_list.append(p)
    cv2.circle(disp_img,p.point, 1, p.color, -1)
    
def resample():
    resampled_p = []
    index = int(random.random() * MAX_PAR)
    beta = 0.0
    mw = max(w_list)
    
    for i in range(MAX_PAR):
        beta += random.random() * 2.0 * mw
        while beta > w_list[index]:
            #print w_list[index]
            beta -= w_list[index]
            index = (index + 1)% MAX_PAR
        resampled_p.append(p_list[index])
    return resampled_p

def particleFilter(I_diff):
    global p_list, w_list
    temp_w = []
    
    for e in p_list:
        e.move(velocity)
        e.update_weight(I_diff)
        temp_w.append(e.weight)
    
    w_list = [x/sum(temp_w) for x in temp_w]
    print len(w_list)
    #w_list = list(w_list / sum(w_list))
    p_list = resample()
    print len(p_list)
    
        
while inp_c <= 6:
    print img1_name
    img1 = cv2.imread(img1_name)
    disp_img = cv2.imread(img1_name)
    
    height, width, channels =  img1.shape 
    print height, width, channels

    img1= cv2.copyMakeBorder(img1,pad_size,pad_size,pad_size,pad_size,cv2.BORDER_CONSTANT,value=(0,0,0))
    disp_img= cv2.copyMakeBorder(disp_img,pad_size,pad_size,pad_size,pad_size,cv2.BORDER_CONSTANT,value=(0,0,0))
    
    old_p = p_list
    
    particleFilter(img1)
    
    for p in range(len(p_list)):
        old = (0,0,0)
        new = (0,0,255)

        #cv2.circle(disp_img,old_p[p].point, 2, old, -1)
        cv2.circle(disp_img,p_list[p].point, 2, new, -1)
        
        
    inp_c += 1
    img1_name = 'o' + str(inp_c) + '.png'
        

    output_name = 'po' + str(out_c) + '.png'
    cv2.imwrite(output_name,disp_img)
    out_c += 1
    

print 'broke out of while \n'    
cv2.destroyAllWindows()