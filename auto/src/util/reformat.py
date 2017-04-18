import cPickle as pickle
import numpy as np
import os
from os.path import expanduser
from path import Path
import random
from stop_words import get_stop_words

# The dataset has 5331 positive and 5331 negative reviews
# According to prev work, split inot 90% training (4998) and 10% testing (533)

word2vec_dic = pickle.load(open(expanduser("~/tweetnet/data/word2vec_dict.pkl")))
stop_words = get_stop_words('english')
length = []
missing_word_dic = {}

def get_dataset(dataset_path):
    data_stats = pickle.load(open(expanduser(dataset_path + "/stats.pkl")))
    all_example = {}
    all_class_folders = os.listdir(expanduser(dataset_path+"/all_data/"))
    for class_folder in all_class_folders:
        if class_folder[0] != ".":
            all_example[class_folder] = open(expanduser(dataset_path+'/all_data/' + class_folder))
    return all_example, data_stats


def replace_missing_word(data_by_word):

    new_data = []
    for i in range(len(data_by_word)):
        word = data_by_word[i]
        if word in stop_words and word2vec_dic.get(word) == None:
	    continue
        else:
            new_data.append(word)
             
    idx = range(0, len(new_data))
    random.shuffle(idx) 
    removed = ""
    
    if len(new_data) == 1 and word2vec_dic.get(new_data[0]) != None:
        return new_data + ["-"], new_data[-1]
    elif len(new_data) == 1 and word2vec_dic.get(new_data[0]) == None:
        return [], ""

    valid = False
    for i in idx:
        word = new_data[i]
        if word not in stop_words and word2vec_dic.get(word)!= None:
            removed = new_data[i]
	    data_by_word[i] = "REMOVE"
	    valid = True
            break
    if not valid:  
        print data_by_word
        return [], ""
    return data_by_word, removed

 
def process_data(data, is_missing_word):
    
    d = list(data)
    for i in range(len(d)):
        if ord(d[i]) > ord('z') or ord(d[i]) < ord('a') and d[i] != "'":
            d[i] = " "
    string = "".join(d)
   
    if not is_missing_word:
        string = " ".join(string.split())
        string = string + " " + "EOS"
        length.append(len(string.split()))
        return string, "_"
    
    string, removed = replace_missing_word(string.split())
    if string == []: return [], ""
    string = " ".join(string)
    string = string + " " + "EOS"
    length.append(len(string.split()))
    return string, removed

def reformat_data(dataset_path, is_missing_word):
    
    # Clean up the directory in train and test folder
    d_train, d_test = Path(expanduser(dataset_path+"/Train")), Path(expanduser(dataset_path+"/Test"))
    train_files, test_files = d_train.walk("*.txt"), d_test.walk("*.txt")
    for f in train_files:   
        f.remove()
    for f in test_files:
        f.remove()

    all_example, data_stats = get_dataset(dataset_path)
    n_classes, n_data, n_data_per_class, trainPercent, testPercent = data_stats['n_classes'], data_stats['n_data'], data_stats['n_data_per_class'], data_stats['trainPercent'], data_stats['testPercent']
    all_idx = range(0,n_data)
    random.shuffle(all_idx)
    test_idx = all_idx[0:int(testPercent*n_data)]
    identifier = 0

    for one_class in all_example.keys(): 
        for p in all_example[one_class].readlines():
            if p == "\n":
                continue
            else:
                if identifier in test_idx:
                    file = open(expanduser(dataset_path + "/Test/" + one_class[0:len(one_class)-4] + "/" + str(identifier) + ".txt"), "w")
                else:
                    file = open(expanduser(dataset_path + "/Train/" + one_class[0:len(one_class)-4] + "/" + str(identifier) + ".txt"), "w")

                string, removed = process_data(p, is_missing_word)
                if string != []:
                    file.write(string)
                    missing_word_dic[identifier] = removed
                    identifier += 1
                    file.close()
                else: print p 
    pickle.dump(missing_word_dic, open(expanduser(dataset_path + "/missing_word_dic.pkl"),"w"))
    return sorted(length)[-1]

if __name__ == "__main__":
    print reformat_data("~/automatedMTL/data/rotten_tomato", False)
