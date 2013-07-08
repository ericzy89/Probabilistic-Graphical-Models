'''
    Author : Rahul Manghwani, NYU Courant
    Written : 5th April 2013
'''
import sys;
import math;
import networkx as nx
import copy
import numpy as np; 
from sets import Set
from numpy import *
import random;
import Queue;
import operator;
from collections import OrderedDict;

class MPLPIO(object):
    def __init__(self, s):
        self.s = s; 
        
    def parseNames(self):
        names = self.s.rstrip().split('\n');
        return names; 
    
    def convertToLog(self, value):
        return np.log(value)
        
    def parseUAILog(self):
        #Store a list of lines;
        lines = self.s.rstrip().split('\n');
        no_nodes =  int(lines[1]);
        #Reading States 
        node_states = dict();
        n = 0;
        for no in lines[2].split(' '):
            if no:
                node_states[n] = int(no);
                n = n + 1;
        #Reading Edges
        temp_edges = [];    
        #Stores the Edge's Potentials as dict
        edges = OrderedDict();
        for i in range(4, 4 + int(lines[3]) - no_nodes):
            line = [int(x) for x in lines[i].split(' ') if x];
            temp_edges.append((line[1],line[2]));
            edges[(line[1],line[2])] = None;
        #Stores Node's Potentials as dict
        nodes = dict();
        temp_nodes = [];
        for i in range(4 + int(lines[3]) - no_nodes, 4 + int(lines[3])):
            line = [int(x) for x in lines[i].split(' ') if x];
            temp_nodes.append(line[1]);
            nodes[line[1]] = None;
        #Get Edge Potentials
        k = 0;
        for i in range(5 + int(lines[3]),(5 + int(lines[3]) + (3 * len(temp_edges)))):
            if lines[i] and len(lines[i].split(' ')) > 1:
               temp_dict = OrderedDict();
               line = [float(x) for x in lines[i].split(' ') if x];
               index = 0;
               for x_i in range(node_states[temp_edges[k][0]]):
                   for x_j in range(node_states[temp_edges[k][1]]):
                       temp_dict[(x_i,x_j)] = line[index]
                       index = index + 1; 
               edges[temp_edges[k]] = temp_dict;
               k = k + 1;
        #Get Node Potentials
        k = 0;
        for i in range((5 + int(lines[3]) + (3 * len(temp_edges))), len(lines)):
            if lines[i] and len(lines[i].split(' ')) > 1:
                temp_dict = OrderedDict();
                line = [float(x) for x in lines[i].split(' ') if x];
                index = 0;
                for i in range(node_states[k]):
                    temp_dict[i] = line[index];
                    index = index + 1;
                nodes[temp_nodes[k]] = temp_dict;
                k = k + 1
        
        return (nodes, edges);
    
        
    def parseUAI(self):
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
                    temp_dict[0] = self.convertToLog(float(line[1]));
                    temp_dict[1] = self.convertToLog(float(line[2]));
                    nodes[ temp_nodes[n] ] = temp_dict;
                    n = n + 1;
        
        #Reading Edges
        temp_edges = [];    
        #Stores the Edge's Potentials as dict
        edges = OrderedDict();
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
                    temp_dict[(0,0)] = self.convertToLog( float(line[1]) );
                    temp_dict[(0,1)] = self.convertToLog( float(line[2]) );
                    temp_dict[(1,0)] = self.convertToLog( float(line1[1]) );
                    temp_dict[(1,1)] = self.convertToLog( float(line1[2]) );
                    edges[temp_edges[n]] = temp_dict;
                    n = n + 1;
            i = i + 1;
        
        return (nodes, edges);
        


class MPLPAlgo(object):    
    def findMax(self, nodes, edges, names, print_full_map, file_name, output):
        #Store the delta's as Dict from node to all the delta's that represent edges incoming into that node
        deltas = dict();       
        #Initialize 
        for e in edges:
            #Assume e is (i,j)        
            #Getting states of variable e[1]
            temp_xj = dict();
            for x_j in (nodes[e[1]]).keys():
                temp_xj[x_j] = 0;
            #Check if j is already stored
            if e[1] in deltas.keys():
               deltas[e[1]][e[0]] = temp_xj;
            else:
               outgoing_incomingStates = dict();
               outgoing_incomingStates[e[0]] = temp_xj
               deltas[e[1]] = outgoing_incomingStates;
               
            #Getting states of variable e[0]
            temp_xi = dict();
            for x_i in (nodes[e[0]]).keys():
                temp_xi[x_i] = 0; 
            #Check if i is already stored
            if e[0] in deltas.keys():
                deltas[e[0]][e[1]] = temp_xi;
            else:
                outgoing_incomingStates = dict();
                outgoing_incomingStates[e[1]] = temp_xi;
                deltas[e[0]] = outgoing_incomingStates;
           
        #Compute and iterate until small enough change in L(delta)
        L_delta_prev = 0.0;
        L_delta_next = sys.maxint;
        highest_max_theta = -1.0 * sys.maxint;
        min_l_delta = sys.maxint;
        iteration_max_theta = 1;
        best_map_ass = dict();
        iterations = 1;
        output.write( "Iteration\t\t\t\tInteger Solution(theta(x))\t\t\tL(delta)\t\t\tIntegral Gap\n");
        while True:
            #Go through all the edges
            for e in edges:
                #Assume e as i,j
                #Compute j->i, States of x^i;
                for x_i in (nodes[e[0]]).keys():
                    #Maximize the Factor Slave - Find the maximum value among all states x^j
                    max = -1 * sys.maxint;
                    for x_j in (nodes[e[1]]).keys():
                        #Processing term delta_j^(-i)
                        temp_sum = nodes[e[1]][x_j]
                        #All the edges from k->j such that k not equal to i
                        for k in deltas[e[1]].keys():
                            if k != e[0]:
                                temp_sum = temp_sum + deltas[e[1]][k][x_j];
                        #Processing term theta(i,j)
                        temp_sum = temp_sum + float(edges[e][(x_i,x_j)]);
                        #Check if maximum
                        if temp_sum > max:
                            max = temp_sum;
                    right = 0.5 * max;
                    #Computing the delta_i^(-j);
                    left = nodes[e[0]][x_i];
                    for k in deltas[e[0]].keys():
                        if k != e[1]:
                            left = left + deltas[e[0]][k][x_i];
                    left = -0.5 * left;
                    #Update delta(j->i)(x_i)
                    deltas[e[0]][e[1]][x_i] = left + right;
                
                #Compute i->j , States of x^j;
                for x_j in (nodes[e[1]]).keys():
                    #Maximize the Factor Slave - Find the maximum value among all states x^i
                    max = -1 * sys.maxint;
                    for x_i in (nodes[e[0]]).keys():
                        #Processing term delta_i^(-j)
                        temp_sum = nodes[e[0]][x_i];
                        #All the edges from k->i such that k not equal to j
                        for k in deltas[e[0]].keys():
                            if k != e[1]:
                                temp_sum = temp_sum + deltas[e[0]][k][x_i];
                        #Processing term theta(i,j)
                        temp_sum = temp_sum + float(edges[e][(x_i,x_j)]);
                        #Check if maximum
                        if temp_sum > max:
                            max = temp_sum;
                    right = 0.5 * max;
                    #Computing the delta_j^(-i)
                    left = nodes[e[1]][x_j];
                    for k in deltas[e[1]].keys():
                        if k != e[0]:
                            left = left + deltas[e[1]][k][x_j];
                    left = -0.5 * left;
                    #Update delta(i->j)(x_j)
                    deltas[e[1]][e[0]][x_j] = left + right;   
            #After all edges have been processed,Compute the value of dual objective
            dual_obj = self.computeDualObjective(nodes, edges, deltas);
            #Compute the Local Decoding
            map_ass = self.performLocalDecoding(deltas, nodes);
            integer_sol = self.computeIntegerSol(map_ass, nodes, edges);
            output.write(str(iterations).rjust(10)+"\t\t\t"+str(integer_sol).rjust(15) + "  \t\t\t\t" + str(dual_obj).rjust(20) + "   \t\t\t" + str(float(dual_obj - integer_sol)) +"\n");
            #Updating the Previous and Next Dual objectives
            L_delta_prev = L_delta_next;
            L_delta_next = dual_obj;
            #Storing the Min Delta
            if L_delta_next < min_l_delta:
                min_l_delta = L_delta_next;
            #Storing the Best Possible assignment 
            if integer_sol > highest_max_theta:
                highest_max_theta = integer_sol;
                best_map_ass = map_ass;
                iteration_max_theta = iterations;
            if L_delta_prev - L_delta_next <= 0.0002:
                break;
            if iterations % 10 == 0:
                print "."
            iterations = iterations + 1;
            
        #Print the MAP
        output.write( "\nNo of iterations  = " + str(iterations)+"\n");
        output.write( "\nMax-MAP(theta)\t\tMin-L(delta)\t\tIntegrality gap\n");
        output.write( str(highest_max_theta) + "\t\t" + str(min_l_delta)+ "\t\t" + str(min_l_delta - highest_max_theta)+"\n");
        

        
        if print_full_map:
            if file_name =="2dri.UAI.LG":
                output.write( "Iteration at which you find MAX(Theta) " + str(iteration_max_theta));
            output.write( "\nMAP Assignment " + str(best_map_ass));
        else:
            output.write( "\n \nMAP Assignment - Objects found : \n");
            for i in best_map_ass.keys():
                if i == 111:
                    break;
                else:
                    if best_map_ass[i] == 1:
                        output.write(names[i] + "\n");
            
    def computeDualObjective(self,nodes, edges, deltas):
        dual_obj = 0.0;
        #Going through all the 'I' Slaves
        for i in nodes:
            #Find the maximum for state of x_i
            max = -1 * sys.maxint;
            for x_i in nodes[i].keys():
                temp_sum = nodes[i][x_i];
                #For all edges j->i
                for j in deltas[i].keys():
                    temp_sum = temp_sum + deltas[i][j][x_i];
                if temp_sum > max:
                    max = temp_sum;
            dual_obj = dual_obj + max;
        #Going through all the 'F' Slaves
        for e in edges:
            #Assuming e as (i,j)
            max = -1 * sys.maxint;
            for x_i in nodes[e[0]].keys():
                for x_j in nodes[e[1]].keys():
                   temp_sum = float(edges[e][(x_i,x_j)]);
                   temp_sum = temp_sum - deltas[e[0]][e[1]] [x_i] - deltas[e[1]][e[0]][x_j];
                   if temp_sum > max:
                       max = temp_sum;
            #Add to the dual objective
            dual_obj = dual_obj + max;
        return dual_obj;  
    
    
    def computeIntegerSol(self, map_ass, nodes, edges):
        #Compute the MAP Assignment using local decoding
        sum = 0.0;
        for i in nodes:
            sum = sum + float(nodes[i][map_ass[i]]);
        for e in edges:
            sum = sum + float(edges[e][ (map_ass[e[0]], map_ass[e[1]]) ]);  
        return sum;    

    
    def performLocalDecoding(self, deltas, nodes):
        #Compute the MAP Assignment using local decoding
        map_ass = dict();
        for i in nodes:
            #Go through all the states and find the one having the maximum value
            max = -1.0 * sys.maxint;
            max_state = -1;
            for x_i in nodes[i].keys():
                temp_sum = nodes[i][x_i];
                #For all the delta's j->i    
                for j in deltas[i].keys():
                    temp_sum = temp_sum + deltas[i][j][x_i];
                if temp_sum > max:
                    max = temp_sum;
                    max_state = x_i;
            #Store the MAP for this variable
            map_ass[i] = max_state;
        return map_ass;    
        
        

if __name__ == '__main__':
  args = list(sys.argv[1:])
  if len(args) != 2:
     print "python MPLP.py <names.txt> <file_name>"
     sys.exit(1)
  fnames = args[0];
  fObjectNames = [];
  fObjectNames.append(args[1]); 
  mplp = MPLPAlgo();
  #Read the names
  with open(fnames) as f:
      names = MPLPIO(f.read()).parseNames();
  
  #Write output to a file
  output = open(args[1]+".out", "w");  
  
     
  #Read the Image to Object Map
  with open(fObjectNames[0]) as f:
      if args[1] == "2dri.UAI.LG" or args[1] == "1exm.UAI.LG":
          nodes, edges = MPLPIO(f.read()).parseUAILog();
          mplp.findMax(nodes, edges, names, True, args[1], output);
      else:   
          nodes, edges = MPLPIO(f.read()).parseUAI();
          mplp.findMax(nodes, edges, names, False, args[1], output);
      
  output.close();
  print "Output File Generated"   
  
  
  
