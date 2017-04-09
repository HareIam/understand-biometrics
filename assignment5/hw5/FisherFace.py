import os
import numpy as np
from PIL import Image
from numpy import linalg
from matplotlib import pyplot
from numpy import *
import operator

def ComputeNorm(x):
    # function r=ComputeNorm(x)
    # computes vector norms of x
    # x: d x m matrix, each column a vector
    # r: 1 x m matrix, each the corresponding norm (L2)

    [row, col] = x.shape
    r = np.zeros((1,col))

    for i in range(col):
        r[0,i] = linalg.norm(x[:,i])
    return r

def myLDA(A,Labels):
    # function [W,m]=myLDA(A,Label)
    # computes LDA of matrix A
    # A: D by N data matrix. Each column is a random vector
    # W: D by K matrix whose columns are the principal components in decreasing order
    # m: mean of each projection


    classLabels = np.unique(Labels)
    classNum = len(classLabels)
    dim,datanum = A.shape
    totalMean = np.mean(A,1)
    partition = [np.where(Labels==label)[0] for label in classLabels]
    classMean = [(np.mean(A[:,idx],1),len(idx)) for idx in partition]

    #compute the within-class scatter matrix
    W = np.zeros((dim,dim))
    for idx in partition:
        W += np.cov(A[:,idx],rowvar=1)*len(idx)

    #compute the between-class scatter matrix
    B = np.zeros((dim,dim))
    for mu,class_size in classMean:
        offset = mu - totalMean
        B += np.outer(offset,offset)*class_size

    #solve the generalized eigenvalue problem for discriminant directions
    import scipy.linalg as linalg
    import operator
    ew, ev = linalg.eig(B,W+B)
    sorted_pairs = sorted(enumerate(ew), key=operator.itemgetter(1), reverse=True)
    selected_ind = [ind for ind,val in sorted_pairs[:classNum-1]]
    LDAW = ev[:,selected_ind]
    Centers = [np.dot(mu,LDAW) for mu,class_size in classMean]
    Centers = np.transpose(np.array(Centers))
    return LDAW,Centers, classLabels

def myPCA(A):
    # function [W,LL,m]=mypca(A)
    # computes PCA of matrix A
    # A: D by N data matrix. Each column is a random vector
    # W: D by K matrix whose columns are the principal components in decreasing order
    # LL: eigenvalues
    # m: mean of columns of A

    # Note: "lambda" is a Python reserved word


    # compute mean, and subtract mean from every column
    [r,c] = A.shape
    m = np.mean(A,1)

    A = A - np.transpose(np.tile(m, (c,1)))
    B = np.dot(np.transpose(A), A)
    [d,v] = linalg.eig(B)
    # v is in descending sorted order

    # compute eigenvectors of scatter matrix
    W = np.dot(A,v)
    Wnorm = ComputeNorm(W)

    W1 = np.tile(Wnorm, (r, 1))

    W2 = W / W1
    
    LL = d[0:-1]

    W = W2[:,0:-1]      #omit last column, which is the nullspace
    
    return W, LL, m


def read_faces(directory):
    # function faces = read_faces(directory)
    # Browse the directory, read image files and store faces in a matrix
    # faces: face matrix in which each colummn is a colummn vector for 1 face image
    # idLabels: corresponding ids for face matrix

    A = []  # A will store list of image vectors
    Label = [] # Label will store list of identity label
 
    # browsing the directory
    for f in os.listdir(directory):
        infile = os.path.join(directory, f)
        im = Image.open(infile)
        im_arr = np.asarray(im)
        im_arr = im_arr.astype(np.float32)

        # turn an array into vector
        im_vec = np.reshape(im_arr, -1)
        A.append(im_vec)
        name = f.split('_')[0][-1]
        Label.append(int(name))

    faces = np.array(A)
    faces = np.transpose(faces)
    idLabel = np.array(Label)

    return faces,idLabel

def GetPCA(im,W):
    A=[]

    im_arr = np.asarray(im)
    im_arr = im_arr.astype(np.float32)
    m = np.mean(im)
    # turn an array into vector
    im_vec = np.reshape(im_arr, -1)
    A.append(im_vec)
    faces = np.array(A)
    faces = np.transpose(faces)
    [r, c] = faces.shape
    B=faces - np.transpose(np.tile(m, c))
    y = np.dot(np.transpose(W),B)
    return y

def GetLDA(im,LDAW,W1):
    A=[]

    m=np.mean(im)
    im_arr = np.asarray(im)
    im_arr = im_arr.astype(np.float32)
    # turn an array into vector
    im_vec = np.reshape(im_arr, -1)
    A.append(im_vec)
    faces = np.array(A)
    faces = np.transpose(faces)
    [r, c] = faces.shape
    y = np.dot(np.transpose(LDAW), np.dot(np.transpose(W1), faces -np.transpose(np.tile(m, c))))
    return y

def classify(inputPoint,dataSet,labels,k):
    dataSetSize = dataSet.shape[0]

    diffMat = tile(inputPoint,(dataSetSize,1))-dataSet  #样本与训练集的差值矩阵

    sqDiffMat = diffMat ** 2                    #差值矩阵平方

    sqDistances = sqDiffMat.sum(axis=1)         #计算每一行上元素的和
    distances = sqDistances ** 0.5              #开方得到欧拉距离矩阵
    sortedDistIndicies = distances.argsort()    #按distances中元素进行升序排序后得到的对应下标的列表

    classCount = {}
    for i in range(k):
        voteIlabel = labels[ sortedDistIndicies[i] ]
        classCount[voteIlabel] = classCount.get(voteIlabel,0)+1

    sortedClassCount = sorted(classCount.items(), key = operator.itemgetter(1), reverse = True)
    return sortedClassCount[0][0]

def ConfusionMatrix(Zmix,classLabels,rfa,We,m,LDAW,W1):
    faces,labels=read_faces('C:/Users/xushi/Desktop/papare for SUMclab/Understanding biometrics/assignment5/hw5/test')
    Snum=0
    CMatrix = np.zeros((10, 10))
    [r, c] = faces.shape

    y = np.dot(np.transpose(We), (faces - np.transpose(np.tile(m, (c, 1)))))
    y2 = np.dot(np.transpose(LDAW), np.dot(np.transpose(W1), faces - np.transpose(np.tile(m, (c, 1)))))

    Znew = np.vstack((y * rfa, y2 * (1 - rfa)))

    for num in range(120):
        WhoIam = classify(np.transpose(Znew[:,num]), np.transpose(Zmix), classLabels, 1)
        CMatrix[labels[num],WhoIam]+=1
        if labels[num] == WhoIam:
            Snum+=1
    alfa=Snum/120
    return CMatrix, alfa

# PCA
PathTrain='C:/Users/xushi/Desktop/papare for SUMclab/Understanding biometrics/assignment5/hw5/train'
faces,idLabel=read_faces(PathTrain)
[r,c] = faces.shape
W, LL, m=myPCA(faces)
K=30
We = W[:,: K]
y=np.dot(np.transpose(We),(faces -np.transpose(np.tile(m, (c,1)))))

z=[]

for num in range(0,10):
    yperson=y[:,(num*12):(num*12+12)]
    z.append(np.mean(yperson,1))
z=np.transpose(z)


# LDA
K1=90
W1 = W[:,: K1]
x2=np.dot(np.transpose(W1),(faces -np.transpose(np.tile(m, (c,1)))))
z2=[]
LDAW,Centers, classLabels=myLDA(x2,idLabel)
y2=np.dot(np.transpose(LDAW),np.dot(np.transpose(W1),faces -np.transpose(np.tile(m, (c,1)))))

for num in range(0,10):
    yperson=y2[:,(num*12):(num*12+12)]
    z2.append(np.mean(yperson,1))
z2=np.transpose(z2)
ylabel=[]

# FF
for rfa in range(0,11):
    Zmix = np.vstack((z*rfa/10,Centers*(1-rfa/10)))
    p,alfa=ConfusionMatrix(Zmix,classLabels,rfa/10,We,m,LDAW,W1)
    print(p)


for rfa in range(0,11):
    Zmix = np.vstack((z*rfa/10,Centers*(1-rfa/10)))
    p,alfa=ConfusionMatrix(Zmix,classLabels,rfa/10,We,m,LDAW,W1)
    ylabel.append(alfa)
    print('when rfa is：',rfa/10,' The ratio is:',alfa)

pyplot.figure(1)
xlabe=[0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
pyplot.plot(xlabe, ylabel)
pyplot.savefig('plot.jpg')
pyplot.show()
