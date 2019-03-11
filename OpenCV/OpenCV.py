
from PIL import Image
from pylab import *
from numpy import *
from scipy.ndimage import filters

from CVtools import *

#im=Image.open('f.jpg')
pil_im=Image.open('f.jpg').convert('L')
im=array(pil_im)

im2,cdf=CVtools.histeq(im)

im2=255-im#反相处理

#figure()#创建一个图像
#gray()#不使用颜色信息
#contour(im,origin='image')#在原点的左上角显示
#im2=filters.gaussian_filter(im,10) #高斯模糊
imshow(im2)#绘制图像
figure()
hist(im.flatten(),128)#直方图



show()
