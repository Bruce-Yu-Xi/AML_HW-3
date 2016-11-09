
# coding: utf-8

# # Question_1

# ### import lib

# In[2]:

from nltk.stem import WordNetLemmatizer
import numpy as np
import random
from sklearn.linear_model import LogisticRegression as LGR
from sklearn.cluster import KMeans


# ### PreProcess data

# In[3]:

"""
train_data includes 2400 samples, where each sample is a list including the
elements which are the words in reviews.

train_label includes 2400 samples which belongs to {0,1}, which is the label 
of train_data.

test_data has the same form as the train_data, while it has 600 sample.

test_label is the same as train_label.
"""
def Split(filenames):

    train_data = []
    train_label = []
    test_data = []
    test_label = []
    root = "sentiment labelled sentences/"
    for filename in filenames:
        path = root + filename
        count = [1,1]
        punctuation = ["!","%","&","(",")","+",".",":",";","<","=",">","?","*",",","\t",""]
        for line in open(path):
            if line[-1] == "\n":
                line = line[:-1]
            a = int(line[-1])
            b=[]
            for word in line[:-1].split(' '):
                ##while word and word[-1] in punctuation:
                    ##word = word[:-1]
                ##b.append(wordnet_lemmatizer.lemmatize(word.lower()))
                i = 0
                while i < len(word):
                    if word[i] in punctuation:
                        word = word[:i]+word[i+1:]
                    else:
                        i+=1
                c = word.lower()
                if c == "and" or c == "or" or c=="":
                    continue
                b.append(c)
            if count[a] > 400:
                test_label.append(a)
                test_data.append(b)
            else:
                train_label.append(a)
                train_data.append(b)
            count[a]+=1
    return [train_data, train_label, test_data, test_label]


# In[4]:

[train_data, train_label, test_data, test_label] = Split(["yelp_labelled.txt","amazon_cells_labelled.txt","imdb_labelled.txt"])


# ## Bag of Words 

# In[5]:

"""
dic is a dictionary where key is the word shows in train_data and the items
of is a list with two elements, first one is the frequency of the key and 
second element is the index of the key in feature vector, which we will use
after.
"""
def bagOfWord(data):
    dic = {}
    t = 0
    n = 0
    for dataset in data:
        for line in dataset:
            for word in line:
                if word in dic:
                    dic[word][0] += 1
                elif t == 0:
                    dic[word] = [1,n]
                    n+=1
        t = 1
    return dic


# In[6]:

Dic = bagOfWord([train_data, test_data])
len(Dic)


# In[127]:

"""Build feature vector."""
def buildB(data, dic):
    data_b = []
    size_dic = len(dic)
    for line in data:
        temp = [0]*size_dic
        for word in line:
            if word in dic:
                temp[dic[word][1]]+=1.0
        data_b.append(np.array(temp))
    return data_b
    


# In[128]:

get_ipython().magic('time [train_data_b, test_data_b] = [buildB(train_data,Dic), buildB(test_data,Dic)]')


# ### postprocess feature vectors

# In[129]:

"""
l^2 normalization
"""
def l2normalize(data):
    for vector in data:
        L = np.linalg.norm(vector)
        vector /= L
                
def standardize(data_b, size_dic):
    s = np.array([0.0]*size_dic)
    for bite in data_b:
        s += bite
    s_ = s/len(data_b)
    vec = []
    for bit in data_b:
            vec.append(bit - s_)
    return vec


# In[130]:

"""
train_vec and test_vec will be the feature vector to be used for future.
"""
l2normalize(train_data_b), l2normalize(test_data_b)
[train_vec, test_vec] = [standardize(train_data_b,len(Dic)), standardize(test_data_b,len(Dic))]


# ### K-means

# In[119]:

"""
randomly pick two points in sample set to be initial points
label is a list indicate which cluster the vector is signed to.
p is the list including two mean point that the model converget to.
During the function, it first prints which two points function pick as
initial points and then how many time it iterates.
"""
def KMeans_2(data,size_dic):
    ##p = kmeans.cluster_centers_
    ##label = kmeans.labels_
    a = random.randint(0,len(data)-1)
    b = random.randint(0,len(data)-1)
    while a==b:
        b = random.randint(0,len(data)-1)
    p = np.array([data[a], data[b]])
    print("point_init1 is ",a)
    print("point_init2 is ",b)
    label = [0]*len(data)
    conver = False
    count = 0
    while not conver:
        count += 1
        conver = True
        for i in range(len(data)):
            d = [0]*2
            d[0] = np.linalg.norm(p[0]-data[i])
            d[1] = np.linalg.norm(p[1]-data[i])
            if d[label[i]] > d[1-label[i]]:
                conver = False
                label[i] = 1-label[i]
        if not conver:
            ##print("a")
            for j in [0,1]:
                n_p = 0
                s_p = np.array([0.0]*size_dic)
                for point in range(len(label)):
                    if label[point] == j:
                        s_p += data[point]
                        n_p += 1
                p[j] = s_p/n_p
    print("iterate time is ",count)
    return(label, p)


# In[120]:

def n_kmeans(vec, k_train_label,kmeans_lib,size):
    [k_label, k_p]=KMeans_2(vec, size)
    n_bruce = 0
    n_python = 0
    for i in range(len(k_label)):
        if k_train_label[i] == kmeans_lib.labels_[i]:
            n_python+=1
        if k_train_label[i] == k_label[i]:
            n_bruce+=1
    print("self-designed accuracy is",n_bruce/len(k_label))
    print("          lib accuracy is", n_python/len(k_label))
    print("higher than lib?: ",n_python/len(k_label)<n_bruce/len(k_label) )
    print("************************")


# In[111]:

kmeans = KMeans(n_clusters=2, random_state=0).fit(train_vec)


# In[121]:


for i in range(3):
    n_kmeans(train_vec, train_label, kmeans,len(Dic))


# ### Logistic Regression

# In[104]:

lgr = LGR()
lgr.fit(train_vec ,train_label)
lgr.score(test_data_b,test_label)


# ## N-gram model

# In[20]:

def Ngram(data):
    data_ng = []
    for line in data:
        line_new = []
        for i in range(len(line)-1):
            line_new.append(line[i]+' '+line[i+1])
        data_ng.append(line_new)
    return data_ng


# In[23]:

train_data_ng = Ngram(train_data)
test_data_ng = Ngram(test_data)


# In[24]:

Dic_ng = bagOfWord([train_data_ng, test_data_ng])
len(Dic_ng)


# In[92]:

get_ipython().magic('time [train_data_ng_b, test_data_ng_b] = [buildB(train_data_ng,Dic_ng), buildB(test_data_ng,Dic_ng)]')


# In[95]:

##drop empty element
train_label_ng = train_label[:]
i = 0
while i < len(train_data_ng_b):
    if not np.linalg.norm(train_data_ng_b[i]):
        train_label_ng.pop(i)
        train_data_ng_b.pop(i)
        train_data_ng.pop(i)
    else:
        i+=1
        
test_label_ng = test_label[:]
i = 0
while i < len(test_data_ng_b):
    if not np.linalg.norm(test_data_ng_b[i]):
        test_label_ng.pop(i)
        test_data_ng_b.pop(i)
        test_data_ng.pop(i)
    else:
        i+=1


# In[103]:

l2normalize(train_data_ng_b)
l2normalize(test_data_ng_b)
[train_vec_ng, test_vec_ng] = [standardize(train_data_ng_b, Dic_ng), standardize(test_data_ng_b, Dic_ng)]


# In[123]:

kmeans_ng = KMeans(n_clusters=2, random_state=0).fit(train_vec_ng)


# In[122]:

for i in range(3):
    n_kmeans(train_vec_ng, train_label_ng, kmeans_ng, len(Dic_ng))


# In[ ]:



