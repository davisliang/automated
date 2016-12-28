'''
Creates word2vec embeddings for each tweet using co-occurence 
counts and weighted vector aggregations.
'''
import pickle
import re
import numpy as np

#generalized datapath
data_path = "newdump.txt"

def get_hashtags(): 

    """
    return all hashtags as a list in the dataset (with duplicates) 
    """

    hashtags = []
    with open(data_path, "r") as lines:
        for l in lines:
            if l[0] == 'h':
                hashtag = l.split(" ")
                hashtags.append(hashtag[1:len(hashtag) - 1])
    return hashtags

def get_unique_hashtags():

    """
    return all unique hashtags as a dictionary {hashtag: hastag_id}
    """

    hashtags = get_hashtags()
    hash_dict = {}
    hash_id = 0
    hash_num = 0
    for hashtag in hashtags:
        for i in hashtag:
            hash_num += 1
            if hash_dict.get(i) == None:
                hash_dict[i] = hash_id
                hash_id += 1
    print "Number of unique hashtags: ", hash_id
    print "Total number of hashtags: ", hash_num
    return hash_dict.keys()


def get_coocc_dict():

    """
    compute the cooccurrence between each pair of unique hashtags
    return dictionary {hashtag1: {hashtag2: coocur, hashtag5: coocur}, hashtag2: {hashtag3: coocur, hashtag10: coocur), ...}
    All the coocurrence are none zero
    """

    hashtags = get_hashtags()
    unique_hashtags = get_unique_hashtags()
    coocc_dict = {}
    for h in unique_hashtags:
        coocc_dict[h] = {}
    for hashtag in hashtags:
        if len(hashtag) > 1:
            for p1 in range(len(hashtag)):
                for p2 in range(p1+1, len(hashtag)):
                    dic = coocc_dict[hashtag[p1]]
                    if dic.get(hashtag[p2]) == None:
                        dic[hashtag[p2]] = 1
                    else:
                        dic[hashtag[p2]] += 1
                    dic = coocc_dict[hashtag[p2]]
                    if dic.get(hashtag[p1]) == None:
                        dic[hashtag[p1]] = 1
                    else:
                        dic[hashtag[p1]] += 1
    return coocc_dict

def get_vocabs(file_path, outfile):

    """
    Get the vocabularies in word2vec (in file_path) and store in outfile
    """

    counter = 0
    outfile = open(outfile, "w")
    with open(file_path, "r") as lines:
        for line in lines:
            if counter > 0:
                line = line.split(" ")
                outfile.write(line[0])
                outfile.write("\n")
            counter += 1
        
def get_hashmap_prep(vocab_path):

    """
    generate two hashmaps:
    (1) hashmap_prep removes the keys that are English words in the dictionary returned by get_coocc_dict()
    (2) hashmap1 further removes all the values that are not English words
    """

    all_vocab = []
    with open(vocab_path,"r") as lines:
        for line in lines:
            all_vocab.append(line[0:len(line)-1])
    hash_dict = get_coocc_dict()
    count1 = 0
    count2 = 0
    english_hashtags = 0
    for hashtag in hash_dict.keys():
        print count1
        count1 += 1
        if hashtag in all_vocab:
            hash_dict.pop(hashtag)
            english_hashtags += 1
    print "Number of English hashtags: ", english_hashtags
    with open("hashmap_prep.pkl", "wb") as ff:
            pickle.dump(hash_dict, ff, pickle.HIGHEST_PROTOCOL)
    for hashtag in hash_dict.keys():
        print count2
        count2 += 1
        dic = hash_dict[hashtag]
        for h in dic.keys():
            if h not in all_vocab:
                dic.pop(h)
    with open("hashmap1.pkl", "wb") as f:
            pickle.dump(hash_dict, f, pickle.HIGHEST_PROTOCOL)


def word2vec_dict(file_path):

    """
    convert the word2vec.txt to a dictionary with format {word: vector}
    """
    
    word2vec = {}
    count = 0
    with open(file_path, "rb") as lines:
        for line in lines:
            print count
            if count > 0:
                line = line.split(" ")
                word2vec[line[0]] = [float(num) for num in line[1:len(line)-1]]
                word2vec[line[0]] = np.asarray(word2vec[line[0]])
            count += 1
    with open("word2vec_dict.pkl", "wb") as f:
        pickle.dump(word2vec, f, pickle.HIGHEST_PROTOCOL)


def generate_vector(hash_map, word2vec):

    """
    remove all the hashtags that do not have any coocurrence with an English word
    given the dictionary of format {nonEnglish1: {English2: coocur2, English3, coocur3}},
    compute the weight of each English word embedding by coocur_i/sum_i (coocur_i)
    get the embedding of each English word from word2vec_dict and compute the weighted sum
    then average over number of embedding used to generate this embedding
    store the final dictionary as a pickle file
    """

    with open(hash_map, "rb") as f:
        hashmap = pickle.load(f)
    for k in hashmap.keys():
        if hashmap[k] == {}:
            hashmap.pop(k)
        else:
            for kk in hashmap[k].keys():
                if hashmap[k][kk] == {}:
                    hashmap[k].pop(kk)
            if hashmap[k] == {}:
                hashmap.pop(k)
    with open("hashmapfinal.pkl", "wb") as ff:
        pickle.dump(hashmap, ff, pickle.HIGHEST_PROTOCOL)
    print "Number of valid none English word hashtags: ", len(hashmap.keys())

    print "Start loading word2vec"
    v = open(word2vec, "rb")
    word2vec_dict = pickle.load(v)
    
    vector_dict = {}

    print "Start converting word to vector"
    for tag1 in hashmap.keys():
        summ = sum(hashmap[tag1].values())
        vec = np.zeros(300)
        for tag2 in hashmap[tag1].keys():
            weight = hashmap[tag1][tag2]*1.0/summ
            vec += weight * word2vec_dict[tag2]
        vector_dict[tag1] = vec

    with open("word2vec_additional.pkl", "wb") as fff:
        pickle.dump(vector_dict, fff, pickle.HIGHEST_PROTOCOL)

def tweet2vector(dict1_path, dict2_path):

    """
    given the word2vec dictionary and the addtional non english word dictionary, generate one vector for each tweet
    by adding up and average over the number of hashtags 
    """

    with open(dict1_path, "rb") as f:
        words_dict = pickle.load(f)
    print "word to vec dictionary loaded!"
    with open(dict2_path, "rb") as ff:
        additional_dict = pickle.load(ff)
    print "additional words dictionary loaded!"
    words = words_dict.keys()
    additional = additional_dict.keys()
    hashtag_embed = []
    print "Start to convert hashtags to vectors!"
    count = 0
    with open(data_path, "r") as lines:
        for l in lines:
            num_valid = 0
            if l[0] == 'h':
                print count
                count += 1
                hashtags = l.split(" ")
                embed = np.zeros(300)
                for hashtag in hashtags[1:len(hashtags) - 1]:
                    if words_dict.get(hashtag) != None:
                        embed += words_dict[hashtag]
                        num_valid += 1
                    elif additional_dict.get(hashtag) != None:
                        embed += additional_dict[hashtag]
                        num_valid += 1
                if num_valid != 0:
                    hashtag_embed.append(embed*1.0/(num_valid*1.0))
                else:
                    hashtag_embed.append(embed)
    with open("hashtag_embedding.pkl","wb") as fff:
        pickle.dump(hashtag_embed, fff, pickle.HIGHEST_PROTOCOL)
    
def remove_invalid_tweets(embed):
    tweets = []
    embeddings = []
    count = 0
    
    f = open(embed, "r")
    old_embed = pickle.load(f)
    print "Old embedding loaded!"

    with open(data_path, "r") as lines:
        for l in lines:
            if l[0] == 't':
                if (not (old_embed[count] == np.zeros(300)).all()):
                    print count
                    tweets.append(l)
                    embeddings.append(old_embed[count])
                count += 1
    #tweets = np.asarray(tweets)
    embeddings = np.asarray(embeddings)
    print "Removing finished!"

    with open("new_tweets.pkl", "wb") as ff:
        pickle.dump(tweets, ff, pickle.HIGHEST_PROTOCOL)
    with open("new_embeddings.pkl", "wb") as fff:
        pickle.dump(embeddings, fff, pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":
    get_unique_hashtags()
    get_hashmap_prep("vocabs.txt")
    generate_vector("hashmap1.pkl", "word2vec_dict.pkl")
    tweet2vector("word2vec_dict.pkl", "word2vec_additional.pkl")
    #remove_invalid_tweets("hashtag_embedding.pkl")
