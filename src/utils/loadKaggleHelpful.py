#import variables
import cPickle as pickle
import numpy
import gzip
from os.path import expanduser
# function implementations
def readGz(f):
  for l in gzip.open(f):
    yield eval(l)
    
def loadTrain():
    text=[];helpful=[];outOf=[];userID=[];itemID=[]
    # collecting the data
    for metablock in readGz(expanduser('~/tweetnet/data/train.json.gz')):
        text.append(metablock['reviewText'])
        helpful.append(metablock['helpful']['nHelpful'])
        outOf.append(metablock['helpful']['outOf'])
        userID.append(metablock['reviewerID'])
        itemID.append(metablock['itemID'])
    return text, helpful,outOf, userID, itemID

def loadTest():
    text=[];outOf=[];userID=[];itemID=[]
    #collecting the data
    for metablock in readGz(expanduser('~/tweetnet/data/test_Helpful.json.gz')):
        text.append(metablock['reviewText'])
        outOf.append(metablock['helpful']['outOf'])
        userID.append(metablock['reviewerID'])
        itemID.append(metablock['itemID'])
    return text, outOf, userID, itemID    
