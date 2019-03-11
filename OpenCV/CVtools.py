
from PIL import Image
from pylab import *
from numpy import *
from scipy.ndimage import filters

class CVtools(object):
    """图像基础操作"""
    def imresize(im,sz):
        """重新定义图像数组大小"""
        pil_im = Image.fromarray(uint8(im))
        return array(pil_im.resize(sz))

    def histeq(im,nbr_bins=256):
        """灰图图像进行直方图均衡化"""
        imhist,bins = histogram(im.flatten(),nbr_bins,normed=True)
        cdf = imhist.cumsum()#累积分布函数
        cdf = 255 * cdf / cdf[-1]#归一化
        im2 = interp(im.flatten(),bins[:-1],cdf)#使用累积分布函数的线性插值，计算新的像素值
        return im2.reshape(im.shape),cdf

    def compute_average(imlist):
        """计算图像列表的平均图像"""

        averageim = array(Image.open(imlist[0]),'f')

        for imname in imlist[1:]:
            try:
                averageim+=array(Image.open(imname))
            except :
                pass
        averageim/=len(imlist)

        return array(averageim,'uint8')

    def pca(X):
        """
        主成分分析
        输入：矩阵X其中该矩阵中存储训练数据，每一行为一条训练数据
        输出：投影矩阵（按照维度的重要性排序）、方差、均值
        """
    
        num_data,dim = X.shape#获取维数

        #数据中心化
        mean_X = X.mean(axis=0)
        X = X - mean_X
    
        if dim < num_data:
            #PCA-使用紧致技巧
            M = dot(X,X.T)#协方差矩阵
            e,EV = linalg.eigh(M)#特征值和特征向量
            tmp = dot(X.T,EV)#紧致技巧
            V = tmp[::-1]#逆转特征向量
            S = sqrt(e)[::-1]#逆转特征值
            for i in range(V.shape[1]):
                V[:,i]/=S
        else:
            #PCA-使用SVD方法
            U,S,V = linalg.svd(x)
            V = V[:num_data]#返回前num_data维的数据
    
        #返回投影矩阵、方差和均值
        return V,S,mean_X

    def imsobel(im):
        '''sobel导数滤波器'''
        imx = zeros(im.shape)
        filters.sobel(im,1,imx)

        imy = zeros(im.shape)
        filters.sobel(im,0,imy)

        magnitude = sqrt(imx ** 2 + imy ** 2)

        return magnitude

class Harris(object):
    def compute_harris_response(im,sigma=3):
        """在一副灰度图像中，对每个像素计算Harris角点检测器响应函数"""

        #计算导数
        imx = zeros(im.shape)
        filters.gaussian_filter(im,(sigma,sigma),(0,1),imx)
        imy = zeros(im.shape)
        filters.gaussian_filter(im,(sigma,sigma),(1,0),imy)

        #计算Harris矩阵的分量
        Wxx = filters.gaussian_filter(imx * imx,sigma)
        Wxy = filters.gaussian_filter(imx * imy,sigma)
        Wyy = filters.gaussian_filter(imy * imy,sigma)

        #计算特征值和迹
        Wdet = Wxx * Wyy - Wxy ** 2
        Wtr = Wxx + Wyy

        return Wdet / Wtr

    def get_harris_points(harrisim,min_dist=10,threshoid=0.1):
        """ 从一副Haris响应图像中返回角点。 min_dist为分割角点和图像边界的最少像素数目"""

        #寻找高于阈值的候角点
        corner_threshold = hharrisim.max() * thershold
        harrisim_t = (harrisim > corner_threshold) * 1

        #得到候选点的坐标
        coords = array(harrisim_t.nonzero()).T

        #以及它们的Harris响应值
        index = aargsort(candidate_values)

        #将可行点的位置保存到数组中
        allowed_locations = zeros(hharrisim.shape)
        allowed_locations[min_dist:-min_dist,min_dist:-min_dist] = 1

        #按照min_distance 原则 ，选择最佳Harris点
        filtered_coords = []
        for i in index:
            if allowed_locations[coords[i,0],coords[i,0]] == 1:
                filtered_coords.append(coords[i])
                allowed_locations[(coords[i,0] - min_dist):(coords[i,0] + min_dist),(coords[i,1] - min_dist):(coords[i,1] + min_dist)] = 0

        return filtered_coords
