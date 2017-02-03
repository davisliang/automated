# python dumpDedup.py > ~/tweetnet/data/dump.txt
import numpy
from os.path import expanduser

dumpFile = open(expanduser("~/tweetnet/data/bigDupeDump.txt"))
dumpLines = dumpFile.readlines()

dumpSet = set()

for i in range(len(dumpLines)):
    if dumpLines[i][0:4] == 'text':
        if dumpLines[i] in dumpSet:
            continue;
        else:
            dumpSet.add(dumpLines[i])
            print "\n",
            print dumpLines[i],
            print dumpLines[i+1],
