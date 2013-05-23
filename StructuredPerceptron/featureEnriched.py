import sys;
import math;
import copy
import numpy  
from sets import Set
from numpy import *
import random;
import Queue;
import operator;
from os import listdir
from os.path import isfile, join
from collections import OrderedDict;
import MPLP;


class Word(object):
    def __init__(self, token, tag, features):
        self.token = token;
        self.tag = tag;
        self.features = features;
    
    def __str__(self):
        return str(self.token) + "   "  +  str(self.tag) + "   "+ str(self.features);
        
        
class IO(object):    
    def __init__(self, path, train_size, test_size):
        self.train = [];
        self.test = [];
        suffixes, prefixes = self.getSuffixDistributions(path);
        self.suffix_len = int(len(suffixes) + 1);
        self.prefix_len = int(len(prefixes) + 1); 
        #Read the training sentences
        for i in range(1, train_size):
            filename = path + "train-" + str(i) + ".txt";
            #Read the file
            words = (open(filename).read()).rstrip().split('\n');
            sentence = [];
            for word_ in words:
                w = word_.split(',');
                temp = [int(k) for k in w[2:]];
                #Adding a feature based on prefix 
                if len(w[0]) <= 3 or w[0][:4] not in prefixes:
                    temp.append(1);
                else: 
                    temp.append(prefixes.index(w[0][:4]) + 2);
                #Adding a feature based on suffix 
                if len(w[0]) <= 3 or w[0][-4:] not in suffixes:
                    temp.append(1);
                else: 
                    temp.append(suffixes.index(w[0][-4:]) + 2);    
                w = Word(w[0], w[1], temp); 
                sentence.append(w);
            self.train.append(sentence);
        #Read the testing sentences
        for i in range(1, test_size):
            filename = path + "test-" + str(i) + ".txt";
            #Read the file
            words = (open(filename).read()).rstrip().split('\n');
            sentence = [];
            for word_ in words:
                w = word_.split(',');
                temp = [int(k) for k in w[2:]];
                #Adding a feature based on prefix 
                if len(w[0]) <= 3 or w[0][:4] not in prefixes:
                    temp.append(0);
                else: 
                    temp.append(prefixes.index(w[0][:4]) + 1);
                #Adding a feature based on suffix 
                if len(w[0]) <= 3 or w[0][-4:] not in suffixes:
                    temp.append(0);
                else: 
                    temp.append(suffixes.index(w[0][-4:]) + 1);    
                w = Word(w[0], w[1], temp);
                sentence.append(w);
            self.test.append(sentence);
    
    def getSuffixDistributions(self,path):
        suffixes = dict();
        prefixes = dict();
        for i in range(1, 5000):
            filename = path + "train-" + str(i) + ".txt";
            #Read the file
            words = (open(filename).read()).rstrip().split('\n');
            for word_ in words:
                w = word_.split(',');
                if len(w[0]) <= 3:
                    continue;
                if w[0][:4] in prefixes:
                    prefixes[w[0][:4]] = prefixes[w[0][:4]] + 1;
                else:
                    prefixes[w[0][:4]] = 1; 
                if w[0][-4:] in suffixes:
                    suffixes[w[0][-4:]] = suffixes[w[0][-4:]] + 1;
                else:
                    suffixes[w[0][-4:]] = 1;
        for s in suffixes.keys():
            if suffixes[s] <= 100:
                del suffixes[s];
        for p in prefixes.keys():
            if prefixes[p] <= 90:
                del prefixes[p];
        return (suffixes.keys(), prefixes.keys())
        
        
        

class Structured_Perceptron(object):
    def __init__(self, train, test, prefix_len, suffix_len):
        self.train = train;
        self.test = test;
        #Initialize the weight vector 
        self.no_of_tags = 10;
        self.f_len = [1,2,2,201,201, prefix_len, suffix_len]
        self.total_feature_sp = sum(self.f_len);
        self.f_loc = [0,1,3,5,206,407, (407 + prefix_len)];
        self.dimension = (self.total_feature_sp) * self.no_of_tags + math.pow(self.no_of_tags, 2);
        self.weights = [0 for _ in range(0,int(self.dimension)) ];
        self.weights_avg = [0 for _ in range(0,int(self.dimension)) ];
        self.no_of_epochs = 50;
        
    
    def predict_POS(self, sentence, weights):
        #Build the node and edge potentials for inference
        nodes = OrderedDict();
        wc = 0;
        for w in sentence:
            #Compute the node potential for this word
            tags = OrderedDict();
            for t in range(1, self.no_of_tags+1):
                #Compute the potential at this state
                sum = 0;
                for f_no in range(1, 8):
                    #For this tag
                    index = (t-1) * self.total_feature_sp;
                    #For this Feature
                    index = index - 1;
                    index = index + self.f_loc[f_no-1];
                    #For this feature value
                    if f_no == 2 or f_no == 3:
                        #Starts from 0
                        index = index + w.features[f_no-1] + 1;
                    else:
                        #Starts from 1
                        index = index + w.features[f_no-1];
                    sum = sum + weights[index];
                tags[t-1] = sum;
            #Store the node potential
            nodes[wc] = tags;
            wc = wc + 1;
        #Compute the edge potentials
        edges = OrderedDict();
        for i in range(0, wc-1):
            source = i;
            dest = i + 1;
            states = OrderedDict();
            index = (self.total_feature_sp) * self.no_of_tags - 1;
            count = 1;
            for source_state in range(0, self.no_of_tags):
                for dest_state in range(0, self.no_of_tags):
                    states[(source_state,dest_state)] = weights[index + count];
                    count = count + 1;
            edges[(source,dest)] = states;
        #Return the Map Assignment
        return MPLP.MPLPAlgo().findMax(nodes, edges);   
    
    def computeFeatureVec(self, sentence, tags):
        #Compute f(xi,yi)
        features = [0 for _ in range(0,int(self.dimension)) ];
        #For Individual Tags
        tag_count = 0;
        for w in sentence:
            tag = tags[tag_count];
            index = int(tag * self.total_feature_sp);
            index = index - 1;
            for f_no in range(1, 8):
                start_loc = index + self.f_loc[f_no-1];
                #For this feature value
                if f_no == 2 or f_no == 3:
                    #Starts from 0
                    start_loc = start_loc + w.features[f_no-1] + 1;
                else:
                    #Starts from 1
                    start_loc = start_loc + w.features[f_no-1];
                #print start_loc
                features[start_loc] = features[start_loc] + 1;     
            tag_count = tag_count + 1;    
        #For Tag to Tag 
        for i in range(0, len(tags.keys()) -1):
            source = tags[i]
            dest = tags[i + 1];
            index = ((self.total_feature_sp) * self.no_of_tags) - 1;
            index = index + (source*self.no_of_tags) + (dest+1);
            features[index] = features[index] + 1;
        return features;
    
    def getTags(self, sentence):
        true_tags = dict();
        wc = 0;
        for w in sentence:
            true_tags[wc] = int(w.tag) - 1;
            wc = wc + 1;
        return true_tags;
        
    
    def learn(self):
        for t in range(0, self.no_of_epochs):
            print "Epoch " + str(t);
            for sentence in self.train:
                #Predict the Tag
                y_hat = self.predict_POS(sentence, self.weights);
                features_yhat = self.computeFeatureVec(sentence, y_hat);
                features_yi = self.computeFeatureVec(sentence, self.getTags(sentence));
                diff = [e for e in numpy.subtract(features_yi,features_yhat)];
                self.weights = [e for e in numpy.add(self.weights,diff)];
                temp = [ w / float(self.no_of_epochs * len(self.train)) for w in self.weights]
                self.weights_avg = [e for e in numpy.add(self.weights_avg,temp)];
        return self.weights_avg
       
                
    def computeAvgError(self, sentences, weights):
        total_no_words = 0;
        total_correct = 0;
        for sentence in sentences:
            #Predict
            predicted_tags = self.predict_POS(sentence, weights);
            true_tags = self.getTags(sentence);
            for word in true_tags.keys():
                if predicted_tags[word] == true_tags[word]:
                    total_correct = total_correct  + 1;
                total_no_words = total_no_words + 1;
        if total_correct == 0:
            return 1;
        else:
            return (1 - (total_correct / float(total_no_words)));
                
                    
                
            
                    
        
if __name__ == '__main__':
  args = list(sys.argv[1:])  
  train_size = 100;
  io = IO(args[0], train_size, 1000)
  #print str(train_size)
  perceptron = Structured_Perceptron(io.train, io.test, io.prefix_len, io.suffix_len);
  learnt_weights = perceptron.learn();
  print str(perceptron.computeAvgError(io.train, learnt_weights));
  print str(perceptron.computeAvgError(io.test, learnt_weights));
