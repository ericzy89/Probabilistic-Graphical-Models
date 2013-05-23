import sys;
import math;
import matplotlib.pyplot as plt
import networkx as nx  


class ChowTiuIO(object):
    def __init__(self, s):
        self.s = s; 
        
    def parseNames(self):
        names = self.s.rstrip().split('\n');
        return names; 
    
    def parseMatrix(self):
        image_object_map = [];
        #Store a list of lines;
        lines = self.s.rstrip().split('\n');
        for l in lines:
            line = [int(x) for x in l.split(' ')];
            image_object_map.append(line);
        return image_object_map;
    
    def writeFile(self, nodes, edges):
        f = open(self.s, "w");
        f.write("MARKOV" + "\n");
        f.write(str(len(nodes.keys())) +"\n");
        for _ in range(len(nodes.keys())):
            f.write(str(2) + " ");
        f.write("\n" + str ( len(nodes.keys()) + len(edges.keys())  ));    
        for n in nodes.keys():
            f.write("\n" + str(1) + " " + str(n) );
        for e in edges.keys():
            f.write("\n" + str(2) + " " + str(e[0]) + " " + str(e[1]));
        f.write("\n")
        for n in nodes.keys():
            f.write("\n" + str(2) + "\n" + " " + str(nodes[n][0]) + " " + str(nodes[n][1]) +"\n");
        for e in edges.keys():
            f.write("\n" + str(4) + "\n" + " " + str(edges[e][(0,0)]) + " " + str(edges[e][(0,1)]) + "\n" + " " + str(edges[e][(1,0)]) +" " + str(edges[e][(1,1)]) + "\n");
        f.write("\n");
        f.close();

class ChowTiuAlgo(object):
    def __init__(self, names, image_object_map):
        self.names = names;
        self.image_object_map = image_object_map;
        #Map from name to its index. Faster loopup
        self.name_Index = dict();
        #Storing Marginal Probabilities 
        self.marginal_prob = dict();
        #Storing the Joint Probabilities
        self.joint_prob = dict();
        #Storing pairs of objects
        self.pairs = set();
        #Storing the edge weight based on empirical mutual information
        self.edge_weight = dict();
        #Generate the Pairs
        self.generatePairs();
        #Initialize the marginal Probabilities
        index = 0;
        for name in self.names:
            self.marginal_prob[name] = [0,0];
            self.name_Index[name] = index;
            index = index + 1;
        #Initialize the Joint Probabilities
        for pair in self.pairs:
            states = dict();
            #Possible Combinations
            states[(0,0)] = 0; states[(0,1)] = 0; states[(1,0)] = 0; states[(1,1)] = 0;
            self.joint_prob[pair] = states;
    
    def generatePairs(self):
        for name1 in self.names:
            for name2 in self.names:
                if name1 != name2 and (name1, name2) not in self.pairs and (name2, name1) not in self.pairs:
                    self.pairs.add((name1,name2));
    
    def learn(self):
        #Learn the Marginal Distributions 
        for image in self.image_object_map:
            i = 0;
            for object in image:
                if object == 0:
                    self.marginal_prob[names[i]][0] = self.marginal_prob[names[i]][0] + 1;
                else:
                    self.marginal_prob[names[i]][1] = self.marginal_prob[names[i]][1] + 1;
                i = i + 1;
        #Normalize
        for name in self.names:
            self.marginal_prob[name][0] = self.marginal_prob[name][0] / float((len(self.image_object_map)));
            self.marginal_prob[name][1] = self.marginal_prob[name][1] / float((len(self.image_object_map)));
        #Learn the Joint Distribution for a pair of variables 
        for image in self.image_object_map:
            for pair in self.pairs:
                index1 = self.name_Index[pair[0]];
                index2 = self.name_Index[pair[1]];
                #Current state i.e whether (0,0) (0,1) (1,0) or (1,1)
                self.joint_prob[pair][(image[index1], image[index2])] = self.joint_prob[pair][(image[index1], image[index2])] + 1;
        #Normalize
        for pair in self.pairs:
            states = self.joint_prob[pair];
            for s in states.keys():
                states[s] = states[s] / float((len(self.image_object_map)));
        #Compute the mutual information.
        for pair in self.pairs:
            total = 0;
            states = self.joint_prob[pair];
            for s in states.keys():
                joint = states[s];
                if joint != 0:
                    #If joint was zero product will be zero hence no need to add it to the sum. Done to avoid log(0)
                    marginal_product = self.marginal_prob[pair[0]][s[0]] * self.marginal_prob[pair[1]][s[1]];
                    total = total + ( joint * math.log( (joint / marginal_product ) ) );
            #Store the mutual information
            self.edge_weight[pair] = total;
                
        #Build a Graph
        bg = nx.Graph();
        for name in self.names:
            bg.add_node(name, None);
        for pair in self.pairs:
            #Negate the edge weight because we need Maximum Spanning Tree
            bg.add_edge(pair[0], pair[1], weight = -1 * self.edge_weight[pair]);
        #Find the minimum spanning tree on this graph, It will be the Maximum Spanning Tree    
        T=nx.minimum_spanning_tree(bg);
        #Compute the Edge Potentials of this Pairwise MRF.
        edges = dict();
        
        for edge in T.edges():
            states = dict();
            #Possible Combinations
            if edge not in self.joint_prob.keys():
                #Swap the ordering
                temp = (edge[1],edge[0]);
                edge = temp;
            states[(0,0)] = self.joint_prob[edge][(0,0)] / float(self.marginal_prob[edge[0]][0] * self.marginal_prob[edge[1]][0] );
            states[(0,1)] = self.joint_prob[edge][(0,1)] / float(self.marginal_prob[edge[0]][0] * self.marginal_prob[edge[1]][1] ); 
            states[(1,0)] = self.joint_prob[edge][(1,0)] / float(self.marginal_prob[edge[0]][1] * self.marginal_prob[edge[1]][0] );
            states[(1,1)] = self.joint_prob[edge][(1,1)] / float(self.marginal_prob[edge[0]][1] * self.marginal_prob[edge[1]][1] ); 
            edges[(self.name_Index[edge[0]],self.name_Index[edge[1]])] = states;
        
        #Node potentials
        nodes = dict();
        for name in self.names:
                nodes[self.name_Index[name]] = self.marginal_prob[name];
        return (nodes, edges);
            
                
                
            
            
        

if __name__ == '__main__':
  args = list(sys.argv[1:])
  fnames = args[0];
  fObjectNames = args[1];  
  #Read the names
  with open(fnames) as f:
      names = ChowTiuIO(f.read()).parseNames();
  #Read the Image to Object Map
  with open(fObjectNames) as f:
      image_object_map = ChowTiuIO(f.read()).parseMatrix();
  
  ct = ChowTiuAlgo(names, image_object_map);
  nodes, edges = ct.learn();
  ChowTiuIO("chow-learned.uai").writeFile(nodes, edges);
  
