import sys;
import math;
import matplotlib.pyplot as plt
import networkx as nx
import copy
import numpy  
from sets import Set
from numpy import *
import random;
import Queue;
import operator;

class BPIO(object):
    def __init__(self, s):
        self.s = s; 
        
    def parseNames(self):
        names = self.s.rstrip().split('\n');
        return names; 
    
    def parseUAI(self, names_len):
        #Store a list of lines;
        lines = self.s.rstrip().split('\n');
        no_nodes =  int(lines[1]);
        temp_nodes = [];
        #Stores Node's Potentials as dict
        nodes = dict();
        for i in range(4, 4 + no_nodes):
            line = [int(x) for x in lines[i].split(' ')];
            temp_nodes.append(line[1]);
            nodes[line[1]] = None;
        n = 0;
        for i in range(int(lines[3]) + 4, int(lines[3]) + 4 + (3 * no_nodes)):
            if lines[i]:
                line = [x for x in lines[i].split(' ')];
                if len(line) > 1:
                    temp_dict = dict();
                    temp_dict[0] = float(line[1]);
                    temp_dict[1] = float(line[2])
                    nodes[ temp_nodes[n] ] = temp_dict;
                    n = n + 1;
        
        #Reading Edges
        temp_edges = [];    
        #Stores the Edge's Potentials as dict
        edges = dict();
        for i in range((no_nodes + 4), int(lines[3]) + 4):
            line = [int(x) for x in lines[i].split(' ')];
            temp_edges.append((line[1],line[2]));
            edges[(line[1],line[2])] = None;
        
        n = 0;
        i = int(lines[3]) + 4 + (3 * no_nodes);
        while i < len(lines):
            if lines[i]:
                line = [x for x in lines[i].split(' ')];
                if len(line) > 1:
                    line1 = [x for x in lines[i+1].split(' ')];
                    i = i + 1;
                    #Create a temp dict
                    temp_dict = dict();
                    temp_dict[(0,0)] = line[1];
                    temp_dict[(0,1)] = line[2];
                    temp_dict[(1,0)] = line1[1];
                    temp_dict[(1,1)] = line1[2];
                    edges[temp_edges[n]] = temp_dict;
                    n = n + 1;
            i = i + 1;
        
        return (nodes, edges);
        

class BeliefPropogation(object):
    #===========================================================================
    # Input
    # message_type : 0 - SumProduct ; 1 - MaxProduct
    # source : Source of the message
    # node_potential : Node potentials of the source node
    # edge_potential : Edge Potential from source to dest - Make sure ordering is same
    # input_messages : list of Messages Received. Every element of list will be a dict
    #===========================================================================
    def message(self, message_type, node_potential, edge_potential, input_messages = None):
        #Source is binary stated;
        calculated_msg = dict();
        for dest_state in range(0,2):
            calculated_msg[dest_state] = dict();
            for source_state in range(0,2):
                calculated_msg[dest_state][source_state] = 0;
                product = 1.0;
                #Node Potential
                product = product * node_potential[source_state];
                #Edge potential
                product = product * float(edge_potential[(source_state,dest_state)]);
                #Messages
                if input_messages:
                    for i in range(len(input_messages)):
                        product = product * input_messages[i][source_state];
                calculated_msg[dest_state][source_state] = product;
        #Preparing message to be returned based on message_type
        message_ret = dict();
        if message_type == 0:
            for dest_state in range(0,2):
                message_ret[dest_state] = sum( calculated_msg[dest_state][source_state] for source_state in calculated_msg[dest_state].keys());    
        else:
            for dest_state in range(0,2):
                message_ret[dest_state] = max( calculated_msg[dest_state][source_state] for source_state in calculated_msg[dest_state].keys());
        #Renormalization Trick
        total = sum (  message_ret[dest_state] for dest_state in message_ret.keys() );
        for dest_state in range(0,2):
            message_ret[dest_state] = message_ret[dest_state] / total;
        return message_ret;
    
    #Returns a list of leaves for a Tree rooted at root_node;
    def getLeaves(self, root_node):
        #Stack
        stack = [];
        #Marked nodes
        marked = [];
        #leaf Nodes;
        leaves = [];
        stack.append(root_node);    
        marked.append(root_node);
        while stack:
            current_node = stack.pop();
            neigbours = self.graph.neighbors(current_node)
            #Check if this is leaf node
            #Leaf node has only edge to its parent
            if len(neigbours) == 1 and neigbours[0] in marked:
                leaves.append(current_node);
            else:
                for ne in neigbours:
                    if ne not in marked:
                        marked.append(ne);
                        stack.append(ne);
        return leaves;
                    
    
    #Runs the algorithm
    def MessagePasser(self, message_type, nodes, edges):
        #Graph Object
        self.graph = nx.Graph();#Pass a Message
        for n in nodes.keys():
            self.graph.add_node(n, None);
        for e in edges.keys():
            self.graph.add_edge(e[0], e[1], None);
        #Randomly select a node as the root node
        root_node = random.sample( nodes.keys(), 1)[0];

        #--------------------------------------------------------- Backward Sweep from leaf to roots
        edges_seen = [];
        nodes_fired = dict();
        for n in nodes.keys():
            nodes_fired[n] = 0;

        current_nodes = Queue.Queue();
        #Nodes currently in the queue
        current_Q = [];

        leaf_nodes = self.getLeaves(root_node);        
        
        for leaf in leaf_nodes:
            current_nodes.put(leaf);
            current_Q.append(leaf);
            
        #Stores the Messages
        backward_messages = dict();
        for n in nodes.keys():
            backward_messages[n] = [];
            
        #Store the Paths
        backward_paths = dict();
        for n in nodes.keys():
            backward_paths[n] = set();
            
        while len(edges_seen) < len(edges.keys()):
            node = current_nodes.get();
            current_Q.remove(node);
            #If the no of messages received by this node is (no of neighbours - no of backward messages = 1)
            neighbours = self.graph.neighbors(node);

            if math.fabs( len(neighbours) - len(backward_messages[node]) ) == 1:
                nodes_fired[node] = 1;
                for ne in neighbours:
                    if (node,ne) not in edges_seen and (ne, node) not in edges_seen:
                        #Get Edge Potential to be sent
                        if (node, ne) in edges.keys():
                            ep = edges[(node, ne)];
                            #Mark the edge
                            edges_seen.append((node,ne));

                        else:
                            temp = edges[(ne,node)];
                            ep = dict();
                            #Need to swap probabilties
                            for t in temp.keys():
                                if t == (1,0) or t == (0,1):
                                    ep[t] = temp[( t[1], t[0] )];
                                else:
                                    ep[t] = temp[t];
                            #Mark the edge
                            edges_seen.append((ne,node));

                        #Pass a Message
                        message = self.message(message_type, nodes[node], ep, backward_messages[node]);
                        backward_messages[ne].append( message );                                                   
                        backward_paths[ne].add(node);
                        if nodes_fired[ne] == 0 and ne != root_node and ne not in current_Q:
                            current_nodes.put(ne);                
                            current_Q.append(ne);          
            else:
                #Put the node back
                if nodes_fired[node] == 0 and node not in current_Q:
                    current_nodes.put(node);
                    current_Q.append(node);
                
        #--------------------------------------------------------- Forward Sweep from root to leaf
                
        edges_seen = [];
        #Stores the Messages
        forward_messages = dict(); 
        for n in nodes.keys():
            if n == root_node:
                #Backward Messages received at the root should be pushed as forward ones now
                forward_messages[root_node] = backward_messages[root_node];
            else:
                forward_messages[n] = [];

        #List of current nodes;
        current_nodes = Queue.Queue();
        current_nodes.put(root_node);
    
        while len(edges_seen) < len(edges.keys()):
            node = current_nodes.get();
            #Explore all the edges for this node
            neighbours = backward_paths[node];
            for ne in neighbours:
                if (node,ne) not in edges_seen and (ne, node) not in edges_seen:
                    #Get Edge Potential to be sent
                    if (node, ne) in edges.keys():
                        ep = edges[(node, ne)];
                        #Mark the edge
                        edges_seen.append((node,ne));
                    else:
                        temp = edges[(ne,node)];
                        ep = dict();
                        #Need to swap probabilties
                        for t in temp.keys():
                            if t == (1,0) or t == (0,1):
                                ep[t] = temp[( t[1], t[0] )];
                            else:
                                ep[t] = temp[t];
                        edges_seen.append((ne,node));
                        
                    #Pass a Message  -- Even a Root node will have incoming messages that would have been received by the end of backward stage.
                    forward_messages[ne].append( self.message(message_type, nodes[node], ep, forward_messages[node]) );
                    #Put the Node
                    current_nodes.put(ne);  

                 
        #--------------------- Compute the Single Node Marginals / Max Marginals
        max_or_sum_marginals = dict();
        for n in nodes.keys():
            states = dict();
            sum = 0;
            for n_state in range(0,2):
                product = nodes[n][n_state];
                for i in range(len(forward_messages[n])):
                    product = product *  ( forward_messages[n][i][n_state] ) ;
                if n != root_node:
                    for i in range(len(backward_messages[n])):
                        product = product * ( backward_messages[n][i][n_state] ) ;
                #Save the State
                states[n_state] = product
                sum = sum + product;
            #Normalize the states
            for n_state in range(0,2):
                states[n_state] = states[n_state] / sum;
            max_or_sum_marginals[n] = states;
        return max_or_sum_marginals;
        
        

if __name__ == '__main__':
  args = list(sys.argv[1:])
  fnames = args[0];
  fObjectNames = [];
  fObjectNames.append(args[1]); 
  fObjectNames.append(args[2]);
  
  #Read the names
  with open(fnames) as f:
      names = BPIO(f.read()).parseNames();
     
  #Read the Image to Object Map
  for file in range(0,2):
      with open(fObjectNames[file]) as f:
          nodes, edges = BPIO(f.read()).parseUAI(len(names));
          
      bp = BeliefPropogation();

      max_marginal = bp.MessagePasser(1, nodes, edges);
      objects_present = [];
      for i in range(0,111):
          #Local Decoding
          if max(max_marginal[i].iteritems(), key=operator.itemgetter(1))[0] == 1:
              #If the max marginal is 1, object is present;
              objects_present.append(names[i]);

      print "MAP -- OBJECTS PRESENT FOR FILE " + str(fObjectNames[file]) + "   " + str(objects_present);
  
      sum_marginal = bp.MessagePasser(0, nodes, edges);
      object_present_with_prob1 = [];
      object_present_with_prob2 = [];
      for i in range(0,111):
        if sum_marginal[i][1] >= 0.8:
            object_present_with_prob1.append(names[i]);
        if sum_marginal[i][1] > 0.6 :
            object_present_with_prob2.append(names[i]);
      print "OBJECTS PRESENT WITH PROBABILITY GREATER THAN OR EQUAL TO 0.8 FOR FILE  " + str(fObjectNames[file]) + "   " + str(object_present_with_prob1);
      print "OBJECTS PRESENT WITH PROBABILITY GREATER THAN 0.6 FOR FILE " + str(fObjectNames[file]) + "   " + str(object_present_with_prob2);

  
  