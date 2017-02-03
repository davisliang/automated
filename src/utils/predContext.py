
import numpy

def createHtDict(dic, allHashtags):
    htDic = {}
    for ht  in allHashtags:
        if ht not in htDic.keys():
            htDic[ht] = dic[ht]
    return htDic

def predContext(htDictionary, modelOutput, topN, label):
    correct = False
    keyResult = []
    sortedKeyResult = []
    dotResult = numpy.zeros([len(htDictionary)])

    counter = 0
    for k in htDictionary.keys():
        dotResult[counter] = -numpy.dot(modelOutput,htDictionary[k])
        keyResult.append(k)
        counter = counter + 1
    
    sortIndex = numpy.argsort(dotResult)
    topNdots = dotResult[sortIndex[0:topN]]
    
    for i in range(topN):
        sortedKeyResult.append(keyResult[sortIndex[i]])
   	if label == keyResult[sortIndex[i]]:
	    correct = True 
    return sortedKeyResult, correct, topNdots
