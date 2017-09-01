import numpy
import csv
import os
import pandas as pd
import matplotlib.pyplot as plt

def matrix_factorization(R, P, Q, K, steps=5000, alpha=0.0002, beta=0.02):
    Q = Q.T
    for step in range(steps):
        print('step:', step)
        for i in range(len(R)):
            for j in range(len(R[i])):
                if R[i][j] > 0:
                    eij = R[i][j] - numpy.dot(P[i,:],Q[:,j])
                    for k in range(K):
                        P[i][k] = P[i][k] + alpha * (2 * eij * Q[k][j] - beta * P[i][k])
                        Q[k][j] = Q[k][j] + alpha * (2 * eij * P[i][k] - beta * Q[k][j])
        eR = numpy.dot(P,Q)
        e = 0
        for i in range(len(R)):
            for j in range(len(R[i])):
                if R[i][j] > 0:
                    e = e + pow(R[i][j] - numpy.dot(P[i,:],Q[:,j]), 2)
                    for k in range(K):
                        e = e + (beta/2) * (pow(P[i][k],2) + pow(Q[k][j],2))
        if e < 0.001:
            break
    return P, Q.T

def getData(filePath):
    # df = pd.read_csv(filePath, sep='::', names=['user_id', 'post_id', 'rating'], engine='python')
    df = pd.read_csv(filePath, engine='python')
    maxUserId = df.loc[df['user_id'].idxmax()]['user_id']
    maxPostId = df.loc[df['post_id'].idxmax()]['post_id']
    data = numpy.zeros((maxUserId + 1, maxPostId + 1))
    print('previous: ', data)
    print('max user Id:', maxUserId)
    print('max post Id:', maxPostId)
    refactoredData = refactorData(df, data)
    print('refactorData: ', refactoredData)

    return refactoredData

def refactorData(df, data):
    for row in df.itertuples():
        data[:getattr(row, 'user_id'), getattr(row, 'post_id')] = getattr(row, 'rating')

    return data

R = getData('~/Documents/machine_learning_data_2017_05_25/votes.csv')
# getData('~/Downloads/ml-20m/rating.csv')
# R = getData('~/Downloads/ml-1m/ratings.dat')

# R = [
#     [5,3,0,1],
#     [4,0,0,1],
#     [1,1,0,5],
#     [1,0,0,4],
#     [0,1,5,4],
# ]

R = numpy.array(R)

N = len(R)
M = len(R[0])
K = 2

P = numpy.random.rand(N,K)
Q = numpy.random.rand(M,K)

nP, nQ = matrix_factorization(R, P, Q, K)
nR = numpy.dot(nP, nQ.T)

print(R)
print(numpy.around(nR))



def getMatrixColumnSize(df):
    matrixColumnSize = 0
    lastUserId = df.iloc[1]['user_id']

    for row in df.itertuples():
        userId = getattr(row, 'user_id')
        if (userId == lastUserId):
            matrixColumnSize += 1
        else:
            break

    return matrixColumnSize

def getMatrixRowSize(df, matrixColumnSize):
    print('getMatrixRowSize')
    print(len(df.index))
    print(len(df.index) / matrixColumnSize)
    matrixRowSize = 0

    # lastUserId = df.iloc[1]['user_id']

    # for row in df.itertuples():
        # userId = getattr(row, 'user_id')
        # if (userId != lastUserId):
            # matrixRowSize += 1
            # lastUserId = userId

    return matrixRowSize
