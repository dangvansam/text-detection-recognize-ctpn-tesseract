import math
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import math
from PIL import Image
from collections import Counter
#%matplotlib inline
import cv2
import numpy as np
from matplotlib import pyplot as plt
import json
import sys
import subprocess
import os

class cutImage(object):
    def __init__(self,img, bin_threshold, kernel, iterations, areaRange, filename, border=10, show=True, write=True,):
        '''
        :param img:
        :param /:
        :param kernel:
        :param iterations:
        :param areaRange:
        :param filename:
        :param border:
        :param show:
        :param write:
        '''
        self.img = img
        self.bin_threshold = bin_threshold
        self.kernel = kernel
        self.iterations = iterations
        self.areaRange = areaRange
        self.border = border
        self.show = show
        self.write = write
        self.filename = filename

    def getRes(self):
        fl = open(self.filename,'w')
        if self.img.shape[2] == 1:
            img_gray = self.img
        elif self.img.shape[2] ==3:
            img_gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(img_gray,self.bin_threshold,255,cv2.THRESH_BINARY_INV)
        img_erode = cv2.dilate(thresh, self.kernel, iterations=self.iterations)

        cv2.imshow('thresh',thresh)
        cv2.imshow('erode',img_erode)
        image, contours, hierarchy = cv2.findContours(img_erode,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        roiList = []
        res =[]
        result = {}
        area_coord_roi = []
        for i in range(len(contours)):
            cnt = contours[i]
            area = cv2.contourArea(cnt)
            if area >self.areaRange[0] and area <self.areaRange[1]:
                x, y, w, h = cv2.boundingRect(cnt)
                roi = self.img[y+self.border:(y+h)-self.border,x+self.border:(x+w)-self.border]
                area_coord_roi.append((area,(x,y,w,h),roi))
        max_area = max([info[0] for info in area_coord_roi])

        for info in area_coord_roi:
            if info[0]==max_area:
                max_rect = info[1]
        for each in area_coord_roi:
            x,y,w,h = each[1]
            if x>max_rect[0] and y>max_rect[1] and (x+w)<(max_rect[0]+max_rect[2]) and (y+h) <(max_rect[1]+max_rect[3]):
                pass
            else:
                tmp_= each[1]
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

                tmp = []
                name = "tmp.jpg"
                cv2.imwrite(name,each[2])
                # text = image_to_string(name,False,'-l chi_sim')
                # tmp.append(text)
                tmp.append(" ")
                tmp.extend(list(tmp_))
                tmp.append("0 0 0")
                res.append(tmp)
                os.remove(name)
        #cv2.imshow("cuted img",img)

        result['1']=[res]
        fl.write(json.dumps(result))
        return roiList

class detectTable(object):
    def __init__(self, src_img):
        self.src_img = src_img

    def run(self):
        if len(self.src_img.shape) == 2:
            gray_img = self.src_img
        elif len(self.src_img.shape) ==3:
            gray_img = cv2.cvtColor(self.src_img, cv2.COLOR_BGR2GRAY)

        thresh_img = cv2.adaptiveThreshold(~gray_img,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,15,-2)
        h_img = thresh_img.copy()
        v_img = thresh_img.copy()
        scale = 15
        h_size = int(h_img.shape[1]/scale)

        h_structure = cv2.getStructuringElement(cv2.MORPH_RECT,(h_size,1))
        h_erode_img = cv2.erode(h_img,h_structure,1)

        h_dilate_img = cv2.dilate(h_erode_img,h_structure,1)
        # cv2.imshow("h_erode",h_dilate_img)
        v_size = int(v_img.shape[0] / scale)

        v_structure = cv2.getStructuringElement(cv2.MORPH_RECT, (1, v_size))
        v_erode_img = cv2.erode(v_img, v_structure, 1)
        v_dilate_img = cv2.dilate(v_erode_img, v_structure, 1)

        mask_img = h_dilate_img+v_dilate_img
        joints_img = cv2.bitwise_and(h_dilate_img,v_dilate_img)
        #cv2.imshow("joints",joints_img)
        #cv2.imshow("mask",mask_img)

        return mask_img, joints_img

def drawLine(all_lines,height,width):
    blank_image = np.zeros((height, width, 3), np.int8)
    color = tuple(reversed((0,0,0)))
    blank_image[:] = color
    for _line in all_lines:
        for line in _line:
            if line[1]<0:
                line[1]=0
            if line[2]<0:
                line[2]=0
            if line[3]<0:
                line[3]=0
            if line[4]<0:
                line[4]=0

            p1=  [int(np.round(line[1])),int(np.round(line[2]))]
            p2 = [int(np.round(line[1])+np.round(line[3])),int(np.round(line[2]))]
            p3 = [int(np.round(line[1])),int(np.round(line[2])+np.round(line[4]))]
            p4 = [int(np.round(line[1])+np.round(line[3])),int(np.round(line[2])+np.round(line[4]))]
            cv2.line(blank_image, (p1[0],p1[1]), (p2[0],p2[1]),(255, 0, 0),1)
            cv2.line(blank_image, (p1[0],p1[1]),(p3[0],p3[1]),(255, 0, 0),1)
            cv2.line(blank_image, (p2[0],p2[1]),(p4[0],p4[1]),(255, 0, 0),1)
            cv2.line(blank_image, (p3[0],p4[1]),(p4[0],p4[1]),(255, 0, 0),1)

    cv2.imshow("img", blank_image)
    cv2.waitKey()
def rotate_img(filename):
    # Read image
    img_path = filename
    src = cv.imread(img_path, cv.IMREAD_COLOR)

    # Convert to grayscale image
    gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)

    # Use Canny to find edges
    edges = cv.Canny(gray, 50, 200, None, 3)

    f, axarr = plt.subplots(1,3,figsize=(20, 20))
    [axi.set_axis_off() for axi in axarr.ravel()]
    # axarr[0].imshow(cv.cvtColor(src, cv.COLOR_BGR2RGB))
    # axarr[0].set_title("Original")
    # axarr[1].imshow(gray, cmap=plt.get_cmap('gray'))
    # axarr[1].set_title("Gray")
    # axarr[2].imshow(edges, cmap=plt.get_cmap('gray'))
    # axarr[2].set_title("Edges")
    #f.show()
    #plt.show()

    cdst = src.copy()
    cdstP = src.copy()

    lines = cv.HoughLines(edges, 1, np.pi / 180, 150, None, 0, 0)
    '''
    if lines is not None:
        for i in range(0, len(lines)):
            rho = lines[i][0][0]
            theta = lines[i][0][1]
            a = math.cos(theta)
            b = math.sin(theta)
            x0 = a * rho
            y0 = b * rho
            pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
            pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
            cv.line(cdst, pt1, pt2, (0,0,255), 3, cv.LINE_AA)
    '''
    linesP = cv.HoughLinesP(edges, 1, np.pi / 1800, 100, None, 90, 60) # xoay 1/10 do
    angle =[]
    if linesP is not None:
        for i in range(0, len(linesP)):
            l = linesP[i][0]
            #print (linesP[i])
            cv.line(cdstP, (l[0], l[1]), (l[2], l[3]), (0,0,255), 1, cv.LINE_AA)
            tga = 999999999
            if np.abs(l[2] - l[0]) >0.00001:
                dx = l[2] - l[0]
                dy = l[3]-l[1]
                sina = dy / math.sqrt( dx * dx + dy*dy)
                tga = np.round( np.arcsin(sina)*180/np.pi,1)
                angle.append(tga)
            #print (tga) #print ('(',l[0],',', l[1],') , (',l[2],',', l[3],') tga=', tga) #print (tga) #
    Angles = Counter(angle)
    most_agl = [ma[0]*ma[1] for ma in Angles.most_common(2)]
    most_agl_rp = [ma[1] for ma in Angles.most_common(2)]
    print(most_agl)
    (h, w) = src.shape[:2]
    print(h,w)
    center = (w / 2, h / 2)
    #degrees_to_rotate = Angles.most_common(1)[0][0]
    #degrees_to_rotate = np.sum(most_agl)/len(most_agl)
    degrees_to_rotate = np.sum(most_agl)/np.sum(most_agl_rp)
    print(degrees_to_rotate)
    M = cv.getRotationMatrix2D(center, degrees_to_rotate, 1)
    rotated90 = cv.warpAffine(src, M, (h, w), borderValue=(255,255,255))
    #cv.imshow('rorated img',rotated90)
    #cv.waitKey(0)
    #print(rotated90.shape)
    #print("rotate done")
    return rotated90

#rotate_img("hoadontiendien-3.png")

#exit()
# f, axarr = plt.subplots(1,2,figsize=(20, 20))
# [axi.set_axis_off() for axi in axarr.ravel()]
# axarr[0].imshow(cv.cvtColor(cdst, cv.COLOR_BGR2RGB))
# axarr[0].set_title("Standard Hough Line Transform")
# axarr[1].imshow(cv.cvtColor(cdstP, cv.COLOR_BGR2RGB))
# axarr[1].set_title("Probabilistic Line Transform")
# f.show()

'''
f, axarr = plt.subplots(1,2,figsize=(40, 40))
[axi.set_axis_off() for axi in axarr.ravel()]
axarr[0].imshow(cv.cvtColor(cdstP, cv.COLOR_BGR2RGB))
axarr[0].set_title("Probabilistic Line Transform")
#plt.show()
'''

#mask, joint = detectTable(rotated90).run()
#cv2.waitKey()