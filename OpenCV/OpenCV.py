
from PIL import Image
from pylab import *
from numpy import *
from scipy.ndimage import filters

from CVtools import *
from Harris import *

#im=Image.open('f.jpg')
pil_im1=Image.open('3.jpg').convert('L')
im1=array(pil_im1)
pil_im2=Image.open('s2.jpg').convert('L')
im2=array(pil_im2)
#im2,cdf=CVtools.histeq(im)

#im2=255-im#反相处理

#figure()#创建一个图像
#gray()#不使用颜色信息
#contour(im,origin='image')#在原点的左上角显示
#im2=filters.gaussian_filter(im,10) #高斯模糊
#imshow(im2)#绘制图像
#figure()
#hist(im.flatten(),128)#直方图



#show()
wid=5
harrisim=Harris.compute_harris_response(im1,5)
filtered_coords=Harris.get_harris_points(harrisim,wid+1)
Harris.plot_harris_points(im1,filtered_coords)

#harrisim = Harris.compute_harris_response(im1,5)
#filtered_coords1 = Harris.get_harris_points(harrisim,wid + 1)
#d1 = Harris.get_descriptors(im1,filtered_coords1,wid)


#harrisim = Harris.compute_harris_response(im2,5)
#filtered_coords2 = Harris.get_harris_points(harrisim,wid + 1)
#d2 = Harris.get_descriptors(im1,filtered_coords2,wid)

#print('starting matching')
#matches = Harris.match_towsided(d1,d2)

#figure()
#gray()
#Harris.plot_matches(im1,im2,filtered_coords1,filtered_coords2,matches)
#show()

