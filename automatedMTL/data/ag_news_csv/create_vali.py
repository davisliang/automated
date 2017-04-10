import os
from os.path import expanduser
import random
import shutil
from random import shuffle
#Step 1: list all folder under training data set:
data_path = "~/tweetnet/automatedMTL/data/ag_news_csv"
all_classes = os.listdir(expanduser(data_path + "/Train")) 

for c in all_classes:
    if c[0] != ".":
	print c
	files = os.listdir(expanduser(data_path+"/Train/"+c))
	shuffle(files)
	if "Sports" in c:
	    for f in files[0:4568]:
	        shutil.move(expanduser(data_path+"/Train/"+c+"/"+f), expanduser(data_path+"/Validation/"+c+"/"+f))
	else:
	    for f in files[0:4569]:
	        shutil.move(expanduser(data_path+"/Train/"+c+"/"+f), expanduser(data_path+"/Validation/"+c+"/"+f))

	
