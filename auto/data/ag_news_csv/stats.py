import os
import cPickle as pickle
from os.path import expanduser

def dedup(data_path):
    classes = os.listdir(expanduser(data_path+"/Test_raw/")) 
    n_data_per_class = {}
    n_train_data = 0
    n_test_data = 0
    file_cnt = 0
    for c in classes:
        print c
	if c[0] != ".":
	    test_files = os.listdir(expanduser(data_path+"/Test_raw/"+c))
            train_files = os.listdir(expanduser(data_path+"/Train_raw/"+c))
	    for t in test_files:
		if t[0] != ".":
                    print file_cnt
		    f = open(expanduser(data_path+"/Test_raw/"+c+"/"+t), "r")
		    txt = f.read().lower()
		    words = txt.split(" ")
		    chrs = list(" ".join(words))
		    for i in range(len(chrs)):
			if ((ord(chrs[i]) < ord('a') or ord(chrs[i]) > ord('z'))) and chrs[i] != "'":
			    chrs[i] = " "
		    remove_long_txt = "".join(chrs)
                    words = remove_long_txt.split()
                    words.append("EOS")
                    remove_long_txt = " ".join(words)
		    with open(expanduser(data_path+"/Test/"+c+"/"+str(file_cnt)+".txt"), "w") as f:
			f.write(remove_long_txt)
			f.close()
                        file_cnt += 1
	    for t in train_files:
		if t[0] != ".":
                    print file_cnt
		    f = open(expanduser(data_path+"/Train_raw/"+c+"/"+t), "r")
		    txt = f.read().lower()
		    words = txt.split(" ")
		    chrs = list(" ".join(words))
		    for i in range(len(chrs)):
			if ((ord(chrs[i]) < ord('a') or ord(chrs[i]) > ord('z'))) and chrs[i] != "'":
			    chrs[i] = " "
		    remove_long_txt = "".join(chrs)
                    words = remove_long_txt.split()
                    words.append("EOS")
                    remove_long_txt = " ".join(words)
		    with open(expanduser(data_path+"/Train/"+c+"/"+str(file_cnt)+".txt"), "w") as f:
			f.write(remove_long_txt)
                        f.close()
                        file_cnt += 1
def stats(data_path):
    classes = os.listdir(expanduser(data_path+"/Test/")) 
    n_data_per_class = {}
    n_train_data = 0
    n_test_data = 0
    length = []
    for c in classes:
        if c[0] != ".":
            test_files = os.listdir(expanduser(data_path+"/Test/"+c))
	    train_files = os.listdir(expanduser(data_path+"/Train/"+c))
	    all_files = test_files + train_files
	    for t in test_files:
	        if t[0] != ".":
	    	    if n_data_per_class.get(c) == None:
		        n_data_per_class[c] = 1
		    else:
			n_data_per_class[c] += 1
		    n_test_data += 1
                    with open(expanduser(data_path+"/Test/"+c+"/"+t), "r") as f:
                        txt = f.read()
                        words = txt.split()
                        length.append(len(words))
                        f.close()
	    for t in train_files:
	        if t[0] != ".":
	    	    if n_data_per_class.get(c) == None:
		        n_data_per_class[c] = 1
		    else:
			n_data_per_class[c] += 1
		    n_train_data += 1
                    with open(expanduser(data_path+"/Train/"+c+"/"+t), "r") as f:
                        txt = f.read()
                        words = txt.split()
                        length.append(len(words))
                        f.close()
    length = sorted(length)
    print "Number of classes: ", len(n_data_per_class)
    print "Numbe of data per class: ", n_data_per_class
    print "Number of train data: ", n_train_data
    print "Number of test data: ", n_test_data
    print "Longest sequence: ", length[-1]
    print "Shortest sequence: ", length[0]
    print "Average sequence: ", sum(length) * 1.0 / len(length)
    print length[len(length)-200:len(length)]
    all_data = 0
    for i in n_data_per_class.keys():
        all_data += n_data_per_class[i]
    print all_data
    data_stats={}
    data_stats['n_classes'] = len(n_data_per_class)
    data_stats['n_data'] = n_train_data + n_test_data
    data_stats['n_data_per_class'] = n_data_per_class
    data_stats['n_train_data'] = n_train_data
    data_stats['n_test_data'] = n_test_data
    data_stats['max_length'] = length[-1]
    print data_stats
    pickle.dump(data_stats, open(expanduser(data_path+"/stats.pkl"),"w")) 
if __name__ == "__main__":
    stats("~/automatedMTL/data/ag_news_csv")
