import sys;
import numpy as np;
import math;
import random;
import copy;
import scipy;
from scipy import special;


class IO(object):
    def __init__(self,s):
        self.s = s; 

    def parse(self):
        lines = self.s.rstrip().split('\n');
        #Store the No of Topics
        no_of_topics = int(lines[0]);
        #Topic Distributions
        line = lines[1].split(' ');
        alphas = [];
        for alpha in line:
            if alpha:
                alphas.append(float(alpha));
        #Words distributions stored as word - all topics
        word_topic_map = dict();
        for i in range(2, len(lines)):
            line = lines[i].split(' ');
            word_topic_map[line[0]] = []; 
            for tp in range(1, len(line)):
                if line[tp]:
                    word_topic_map[line[0]].append(float(line[tp]));
        #Return 
        return (no_of_topics, alphas, word_topic_map);


class Gibbs(object):
    def __init__(self, no_of_topics, alphas, word_topic_map):
        self.no_of_topics = no_of_topics;
        self.alphas = alphas;
        self.word_topic_map = word_topic_map;
         #Initialize
        self.sample = dict();
        for word in self.word_topic_map.keys():
            #Randomly choose a topic for this word
            self.sample[word] = random.randint(0,self.no_of_topics);    

    def sampleTopic(self, pdf):
        #Generate a Random No
        accumulate_ = 0.0
        rand_no = random.random()
        for i in range(0,len(pdf)):
            accumulate_ = accumulate_ + pdf[i];
            if accumulate_ > rand_no:
                return i
 
    def gen_Gibbs_Sample(self):
        new_sample = dict();
        new_sample = copy.deepcopy(self.sample);
        while True:
            #Prepare a Topic to Count Map
            topic_to_wordcount = dict();
            for word in new_sample.keys():
                if new_sample[word] in topic_to_wordcount:
                    topic_to_wordcount[new_sample[word]] = topic_to_wordcount[new_sample[word]] + 1; 
                else:
                    topic_to_wordcount[new_sample[word]] = 1;
            for word in new_sample.keys():
                #Prepare New Alpha Counts
                new_alpha = copy.deepcopy(alphas);
                for topic in range(0, self.no_of_topics):
                    if topic in topic_to_wordcount:
                        new_alpha[topic] = new_alpha[topic] + topic_to_wordcount[topic];
                #Compute Theta's
                theta = np.random.dirichlet(new_alpha, 1)[0];
                #Compute the Pdf
                topic_pdf = [];
                for topic in range(0, self.no_of_topics):
                    topic_pdf.append(self.word_topic_map[word][topic] * theta[topic]);
                #Normalize this Probability Distribution
                sum_ = sum(topic_pdf);
                for i in range(0, len(topic_pdf)):
                    topic_pdf[i] = topic_pdf[i] / float(sum_);
                topic_sampled = self.sampleTopic(topic_pdf);
                #Old Topic Count decrememnts
                topic_to_wordcount[new_sample[word]] = topic_to_wordcount[new_sample[word]] - 1;
                if topic_sampled in topic_to_wordcount:
                    topic_to_wordcount[topic_sampled] = topic_to_wordcount[topic_sampled] + 1;
                else:
                    topic_to_wordcount[topic_sampled] = 1;
                #Update the topic for this word of the Sample
                new_sample[word] = topic_sampled;
            #Spit out the Sample
            yield new_sample;
        yield None;
    
    def gen_collapsedGibbs_Sample(self):
        #Produce a Sample
        new_sample = dict();
        new_sample = copy.deepcopy(self.sample);
        while True:
            #Prepare a Topic to Count Map
            topic_to_wordcount = dict();
            for word in new_sample.keys():
                if new_sample[word] in topic_to_wordcount:
                    topic_to_wordcount[new_sample[word]] = topic_to_wordcount[new_sample[word]] + 1; 
                else:
                    topic_to_wordcount[new_sample[word]] = 1;
            for word in new_sample.keys():
                #Sample for this word
                #Generating the Probability Distribution for this word over all the topics
                topic_pdf = [];
                for topic in range(0, self.no_of_topics):
                    #No of other words which have been assigned to this topic. -1 used to exclude this word.
                    if topic in topic_to_wordcount:
                        c_otherw_this_topic = topic_to_wordcount[topic] - 1;
                    else:
                        c_otherw_this_topic = 0;
                    #Add the topic prior
                    left_numerator = c_otherw_this_topic + alphas[topic];
                    #Multiply with the word prior for this topic
                    left_numerator = left_numerator * self.word_topic_map[word][topic];
                    #Save it
                    topic_pdf.append(left_numerator);
                #Normalize this Probability Distribution
                sum_ = sum(topic_pdf);
                for i in range(0, len(topic_pdf)):
                    topic_pdf[i] = topic_pdf[i] / float(sum_);
                topic_sampled = self.sampleTopic(topic_pdf);
                #Old Topic Count decrememnts
                topic_to_wordcount[new_sample[word]] = topic_to_wordcount[new_sample[word]] - 1;
                if topic_sampled in topic_to_wordcount:
                    topic_to_wordcount[topic_sampled] = topic_to_wordcount[topic_sampled] + 1;
                else:
                    topic_to_wordcount[topic_sampled] = 1;
                #Update the topic for this word of the Sample
                new_sample[word] = topic_sampled;
            #Spit out the Sample
            yield new_sample;
        yield None;
    
    
    
    def plotCollapsed(self, expected_thetas,no_of_iterations):
        ground_truth = np.array(expected_thetas[no_of_iterations-51]);
        f = open("./data/Collapsed.dat", "w");
        f.write("#Iteration   #Error\n");
        for itr in range(0, 1500):
            b = np.array(expected_thetas[itr]);
            f.write(str(itr) + "      " + str(np.linalg.norm(ground_truth-b)) + "\n");
        f.close();

    def plotGibbs(self,expected_thetas,no_of_iterations):
        f = open("./data/Gibbs.dat", "w");
        f.write("#Iteration   #Error\n");
        #Compute the Ground Truth
        ground_truth = np.zeros(self.no_of_topics);
        for itr in range(0, no_of_iterations-51):
            ground_truth = ground_truth + expected_thetas[itr];
        ground_truth = np.divide(ground_truth,float(no_of_iterations-51));
        total = np.zeros(self.no_of_topics);
        for itr in range(0, 2000):
            #Compute the Expected value
            b = np.array(expected_thetas[itr]) + total;
            b = np.divide(b,float(itr+1));
            f.write(str(itr) + "      " + str(np.linalg.norm(ground_truth-b)) + "\n");
            total = total + b;
        f.close();
        


    def doInference(self,type_of_sampler = "Gibbs"):
        if type_of_sampler == "Gibbs":
            sample_generator = self.gen_Gibbs_Sample();
        else:
            sample_generator = self.gen_collapsedGibbs_Sample();
        #Storing the samples
        no_of_iterations = 10050;
        expected_thetas = [];
        sum_topic_priors = 0;
        for i in range(0, self.no_of_topics):
            sum_topic_priors = sum_topic_priors + self.alphas[i];
        N = len(self.word_topic_map.keys());    

        topic_pos_ = dict();
        for i in range(0, self.no_of_topics):
            topic_pos_[i] = 0;
            
        for itr in range(0, no_of_iterations):
            sample = sample_generator.next();  
            if itr < 50:
                continue;
            for s in sample.keys():
                topic_pos_[sample[s]] = topic_pos_[sample[s]] + 1;
            
            #Compute the Ground Truth
            exp_theta = [];
            for topic in range(0, self.no_of_topics):
                exp_theta.append(( ( (itr-49) * self.alphas[topic] ) +  topic_pos_[topic] ) /  float(((itr-49) * (sum_topic_priors * N ) )) );
            expected_thetas.append(exp_theta);
        #Compute the L2 Error 
        if type_of_sampler == "Gibbs":
            self.plotGibbs(expected_thetas, no_of_iterations);
        else:
            self.plotCollapsed(expected_thetas, no_of_iterations);
       
            
        
        
class MeanField(object):
    def __init__(self, no_of_topics, alphas, word_topic_map):
        self.no_of_topics = no_of_topics;
        self.alphas = alphas;
        self.word_topic_map = word_topic_map;

    def normalize(self):
        total = 0;
        expected_theta = [];
        for topic in range(0,self.no_of_topics):
            total = total + self.gamma[topic];
        for topic in range(0,self.no_of_topics):
            expected_theta.append( self.gamma[topic] / float(total) );
        return expected_theta;
        
    def plot(self, expected_thetas):
        ground_truth = np.array(len(expected_thetas)-1);
        f = open("./data/MeanField.dat", "w");
        f.write("#Iteration   #Error\n");
        for itr in range(0,len(expected_thetas)-1):
            b = np.array(expected_thetas[itr]);
            f.write(str(itr) + "      " + str(np.linalg.norm(ground_truth-b)) + "\n");
        f.close();


    def doInference(self):
        #Initialize
        self.phi = dict();
        for word in self.word_topic_map.keys():
            self.topics = dict();
            for i in range(0,self.no_of_topics):
                self.topics[i] = (1.0 / (float) (self.no_of_topics));
            self.phi[word] = self.topics;
        self.gamma = dict();
        for topic in range(0,self.no_of_topics):
            self.gamma[topic] = self.alphas[topic] + ( len(self.word_topic_map.keys()) / (float)(self.no_of_topics) );
        #Iterate until convergence
        convergence = False;
        count = 0;
        it = 0;
        #Store the Gammas
        expected_thetas = [];
        while not convergence:
            newPhi = dict();
            newGamma = dict();
            #Compute the new Phi's
            it = it + 1;
            print "Iteration " + str(it);
            #Store the Digamma of sum of gamma's for all topics
            sum_diagma = 0;
            for topic in range(0,self.no_of_topics):
                sum_diagma = sum_diagma + self.gamma[topic];
            sum_diagma = scipy.special.psi(sum_diagma);    
            for word in self.word_topic_map.keys():
                topics = dict();
                sum = 0.0;
                for topic in range(0,self.no_of_topics):
                    topics[topic] = self.word_topic_map[word][topic] * math.exp(scipy.special.psi(self.gamma[topic]) - sum_diagma);
                    sum = sum + topics[topic];
                #Normalize
                for topic in range(0,self.no_of_topics):
                    topics[topic] = topics[topic] / float(sum);
                newPhi[word] = topics;
            #Compute the new Gamma's
            for topic in range(0,self.no_of_topics):
                newGamma[topic] = self.alphas[topic];
                #For all the words
                for word in self.word_topic_map.keys():
                    newGamma[topic] = newGamma[topic] + newPhi[word][topic];
            #Check for Convergence
            check = True;
            for word in self.word_topic_map.keys():
                for topic in range(0,self.no_of_topics):
                    if abs(newPhi[word][topic] - self.phi[word][topic]) > 1e-5:
                        check = False;
                        break;
                if not check:
                    break
            if check:
                for topic in range(0,self.no_of_topics):
                    if abs(newGamma[topic] - self.gamma[topic]) > 1e-5:
                        check = False;
                        break;
            #Normalizing
            ex_theta = self.normalize();
            expected_thetas.append(ex_theta);
            self.gamma = copy.deepcopy(newGamma);
            self.phi = copy.deepcopy(newPhi);
            #If converged increment the counter
            if check:
                count = count + 1;
                if count == 3:
                    convergence = True;

        #Plot
        self.plot(expected_thetas);                

                   
                    
                    
    
    
                

if __name__ == '__main__':
  args = list(sys.argv[1:])
  abstract = args[0];
  #Read the names
  with open(abstract) as f:
      no_of_topics, alphas, word_topic_map = IO(f.read()).parse();


#===============================================================================
# gs = Gibbs(no_of_topics, alphas, word_topic_map);
# gs.doInference("Gibbs");
#===============================================================================
#gs.doInference("Collapsed");


mf = MeanField(no_of_topics, alphas, word_topic_map);
mf.doInference()
