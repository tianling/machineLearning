#__author__ = 'tianling'

from numpy import *
import operator

def createDataSet():
    group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels


def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]

    #计算待分类点和训练集中任一点的欧式距离
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances**0.5

    #排序和统计
    sortedDistIndicies = distances.argsort()
    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]



def file2matrix(filename):
    fr = open(filename)
    arrayOlines = fr.readlines()
    lineNum = len(arrayOlines) #得到文件行数
    returnMat = zeros((lineNum,3))
    classLabelVector = []
    index = 0
    #解析文件数据到列表
    for line in arrayOlines:
        line = line.strip()
        listFromLine = line.split('/t')
        returnMat[index,:] = listFromLine[0:3]
        classLabelVector.append(int(listFromLine(-1)))
        index += 1

    return returnMat,classLabelVector




