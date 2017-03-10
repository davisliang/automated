import numpy
from os.path import expanduser
def logger(listVals, name):
    logwriter = open(expanduser('~/tweetnet/logs/'+ name), 'a')
    logwriter.write("\n ########################################## \n")
    logwriter.write("EPOCH: " + str(listVals[0][0]) + "\n")
    for i in xrange(1,len(listVals)-1):
        logwriter.write("input: " + str(listVals[i][0]) + "\n")
        logwriter.write("target: " + str(listVals[i][1]) + "\n")
        logwriter.write("isCorrect: " + str(listVals[i][2]) + "\n")
        logwriter.write("topN: "+ str(listVals[i][3]) + "\n\n")
    logwriter.write("numCorrect: " + str(listVals[len(listVals)-1][0]))
    logwriter.write(" percCorrect: " + str(listVals[len(listVals)-1][1]))
    
   
    
