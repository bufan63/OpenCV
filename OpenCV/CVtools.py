
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
