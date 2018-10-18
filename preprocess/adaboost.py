"""
Adaboost algorithm

Python implementation of adaboost algorithm.

Learned from Hanbo-Sun:Multi-class-AdaBoost
https://github.com/Hanbo-Sun/Multi-class-AdaBoost/blob/master
"""
import numpy as np
import math


class AdaBoost:
    """
    """

    def __init__(self):
        self.Tag = 'AdaBoost'


    class initAdaBoost:
        def __init__(self, N):
            self.nWC = 0
            self.WeakClass = []
            self.Weight = np.zeros((N, 1))
            self.trainError = np.zeros((N, 1))
            self.testError = np.zeros((N, 1))
            self.hasTestData = False


    def predAdaBoost(self, classifier, Data, gT=None):
        N = Data.shape[0]

        if gT is None:
            gT = []
        p = len(set(gT))
        M = classifier.nWC
        Label = np.zeros((N, M))

        for i in range(M):
            tid = self.predict(classifier.WeakClass[i], Data)
            for j in range(N):
                Label[j, tid[j]] = Label[j, tid[j]] + classifier.Weight[i, 0]

        Label = np.argmax(Label, axis=1)
        Err = np.sum(np.not_equal(Label, gT)) / float(N)

        return (Label, Err)


    def buildAdaBoost(self, trainData, trainLabel, iter, testData=None, testLabel=None):
        classifier = self.initAdaBoost(iter)
        if testLabel is not None:
            allNum = trainLabel.size + testLabel.size
        else:
            allNum = trainLabel.size
        trNum = trainLabel.size
        sW = np.asarray([1. / trNum])
        sampleWeight = np.matlib.repmat(sW, trNum, 1)

        for i in range(iter):
            weakClassifier = self.fitctree(trainData, trainLabel, 'weights', sampleWeight)
            classifier.WeakClass.append(weakClassifier)
            classifier.nWC = i

            # compute the weight of the classifier
            t, temp_error = self.preAdaBoost(classifier, trainData, trainLabel)

            ide = np.equal(t, trainLabel)
            idne = ~ide
            if temp_error == 0:
                temp_error += 0.01

            classifier.Weight[i] = math.log((1. - temp_error) / temp_error) + math.log(allNum - 1)

            # update sample weight
            temeq = math.exp((-(allNum - 1) / allNum) * (classifier.Weight[i]))
            temneq = math.exp((1 / allNum) * classifier.Weight[i])
            sampleWeight[ide, 0] *= temeq
            sampleWeight[idne, 0] *= temneq
            sampleWeight = sampleWeight / np.sum(sampleWeight, 0)
