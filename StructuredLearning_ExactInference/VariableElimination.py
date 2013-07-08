'''
Author : Rahul Manghwani, NYU Courant
Written : 5th Feb 2013
'''

import sys;
import bifParser;
import matplotlib.pyplot as plt
import networkx as nx  
import copy;
import numpy;
from copy import deepcopy;
from numpy import *;
from sets import Set;

class BuildGraph(object):
    def __init__(self,vars):
        self.BayesianNet = nx.DiGraph();
        self.MarkovNet = nx.Graph();
        #Indexed as factors[nodeName] = list of factors
        self.factors = dict();
        #Map from variable to factors where it exists
        self.var_Factors = dict();
        for var in vars:
            self.var_Factors[var.name] = [];
        #Maintain a map from variable name to variable object
        self.var_Obj = dict();
        for var in vars:
            self.var_Obj[var.name] = var;
            evidence = ();
            for cv in var.cond_vars :
                self.BayesianNet.add_edge(cv, var, None);
                evidence = evidence + (cv,);
            #print "Variable " + var.name + " Evidence : " + str(evidence) + "\n";
            #Make a factor 
            query = ();
            query =  query + (var,);
            f = bifParser.Factor(query, evidence);
            self.var_Factors[var.name].append(f);
            for e in evidence:
                self.var_Factors[e.name].append(f);
    
    def buildMarkovNetwork  (self):
        #Converts Bayesian Network into Markov network
        #Go through a Node and check if it's parents are connected in the directed graph.
        #If no, add an edge between them
        for node in nx.nodes(self.BayesianNet):
            parents = [];
            #Moralize parents of this node
            for parent in node.cond_vars:
                parents.append(parent.name);
                for otherParent in node.cond_vars:
                    if parent.name != otherParent.name:
                        self.MarkovNet.add_edge(parent.name, otherParent.name, None);
                #Add edges between this parent and the node
                self.MarkovNet.add_edge(parent.name, node.name, None);
            #Build a Factor 
            #Make a factor between this node and its parents
            #Get Parents
            factor = [];
            if not parents:
                factor.append( tuple([node.name])  );
                if node.name in self.factors.keys():
                    self.factors[node.name].extend(factor);
                else:
                    self.factors[node.name] = list(factor);
            else:    
                factor.append( tuple ([node.name] + list(parents) ) );
                #Add it node's map
                if node.name in self.factors.keys():
                    self.factors[node.name].extend(factor);
                else:
                    self.factors[node.name] = list(factor);
                for p in parents:
                    if p in self.factors.keys():
                        self.factors[p].extend(factor);
                    else:
                        self.factors[p] = list(factor);

    def MinFill(self,nodes):
        min_fill = sys.maxint;
        best_node = None;
        best_node_factors = None;
        
        #Create a Copy of the Graph 
        for n in nodes:
            #Number of fill edges for this node
            fill_node = 0;
            #Get the factors for this node 
            n_factors = self.factors[n];
            
            #Make a Tau Factor 
            tau_factor = set();
            for n_fact in n_factors:
                for e in n_fact:
                    if e != n and (e in nodes):
                        tau_factor.add(e);
            
            #print "Trying" + str(tau_factor) + " For " + str(n) + "\n";
            
            #To Store Unique factors for this node
            unique_pairs = set();
            #Identify how many fill edges will need to be added
            for p1 in tau_factor:
                for p2 in tau_factor:
                    if p1 != p2 and ((p1,p2) not in unique_pairs and (p2,p1) not in unique_pairs) and self.MarkovNet.has_edge(p1, p2) == False:
                        #Edge does not exist between this pair. Add this fill edge
                        unique_pairs.add((p1,p2));
                        fill_node = fill_node + 1;
            #Check if this the minimum one
            if fill_node < min_fill:
                min_fill = fill_node;
                best_node = n; 
                best_node_factors = tau_factor;
        #Add edges for the selected node
        #print "Varibale " + str(best_node) + " node factors " + str(best_node_factors); 
        for p1 in best_node_factors:
                for p2 in best_node_factors:
                    if p1 != p2 and self.MarkovNet.has_edge(p1, p2) == False:
                        print "Adding Edge between " + str(p1) + " and " + str(p2) + " for removed node " + str(best_node) + "\n";
                        self.MarkovNet.add_edge(p1, p2, None);
        #Clique generated for this iteration 
        clique = best_node_factors;
        #print "Clique " + str(clique) + " Best Node " + best_node;
        #Add the newly generated Tau factor to all the concerned nodes
        for tf in best_node_factors:
            self.factors[tf].append( tuple(list(best_node_factors) ) );
        clique.add(best_node);

        #Return the best Node
        return (best_node, min_fill, clique);
            
    
    def findEliminationOrdering(self):    
        #UnMarkedNodes
        unMarkedNodes = [];
        unMarkedNodes.extend(self.factors.keys());
        #Optimal Ordering
        optimal_ordering = [];
        #No of Fill Edges Added
        fill_total = 0;
        #Largest Clique
        largest_clique = set();
        for _ in range(len(unMarkedNodes)):
            next_best, min_fill, clique = self.MinFill(unMarkedNodes);
            #Book- Kepping
            if len(clique) > len(largest_clique):
                largest_clique = clique;
            fill_total = fill_total + min_fill;
            #Add to optimal Ordering
            optimal_ordering.append(next_best);
            #Mark the node
            unMarkedNodes.remove(next_best);
        print "Optimal Ordering " + str(optimal_ordering);
        print "Total no of fill edges added : " + str(fill_total);
        print "Variable in the Largest Clique : " + str(largest_clique);
        print "Induced Width : " + str(len(largest_clique) - 1);
        return optimal_ordering;
    
        
    def generateStates(self,tau_factors, index, states, totalStateSpace, last_mul):
       if index == len(tau_factors):
           #Generate States
           return states;
       else:
           #Decide the granularity
           granularity = totalStateSpace / ( last_mul * tau_factors[index].nstates);
           i = 0;
           for _ in range( (totalStateSpace / (granularity * tau_factors[index].nstates ) )):
               for state in tau_factors[index].state_names:
                   for _ in range((granularity)):
                       states[i] = states[i] + (state,);
                       i = i + 1;
           return self.generateStates(tau_factors, index+ 1, states, totalStateSpace, (last_mul * tau_factors[index].nstates));
           
    
    def getIndexState(self, variable, state):
        i = 0;
        for sn in variable.state_names:
            if sn == state:
                return i;
            i = i + 1;
        
    def getTauVarIndex(self, tau_factor, var):
        i = 0;
        for v in tau_factor:
            if v.name == var.name:
                return i;
            i = i + 1;
    
    def checkListContains(self, object, name):
        for o in object:
            if o.name == name:
                return True
        return False;
            
    def checkAllTau(self, tau_factors):
        for tf in tau_factors:
            if tf.tau == False:
                return False;
        return True;
    
    
    def evalConstant(self, factors, evidence, eliminated_var,intermediate_state):
        sum = 0;
        to_remove = False;
        factor_to_remove = None;
        for e_state in self.var_Obj[eliminated_var].state_names:
            product = 1;
            for f in factors:
                temp_state = ();
                if f.evidence:
                    #All the evidences will either be the variable being eliminated or Observed variables
                    for ev in f.evidence:
                        temp_state = temp_state + (evidence[ev.name],);
                    #Query/Numerator will never have more than one variable
                    product = product * ((f.query)[0]).probabilities[temp_state][self.getIndexState(self.var_Obj[eliminated_var],e_state)];
                else:
                    #This is variable being eliminated 
                    if self.checkAllTau(f.query) == True:
                        product = product * intermediate_state[f.query][tuple([e_state])];
                        to_remove = True;
                        factor_to_remove = f;
                    else:
                        product = product * ((f.query)[0]).probabilities[e_state];
            #All factors are evaluated
            sum = sum + product;
        if to_remove == True:
            for k in intermediate_state.keys():
                if factor_to_remove.query == k:
                    del intermediate_state[k];
                    factors.remove(factor_to_remove);
                    
        return (sum, intermediate_state, factors);    
    
    def calIndividualQueryProb(self, constants, intermediate_factors, query_observed, evidence, query_state):
        product = 1;
        #Multiply all the constants
        for c in constants:
            product = product * c;
        for f in query_observed:
            temp_state = ();
            if f.evidence:
                for ev in f.evidence: 
                    if ev.name in evidence.keys():
                        temp_state = temp_state + (evidence[ev.name],);
                    else:
                        temp_state = temp_state + (query_state[ev.name],);
                product = product * ((f.query)[0]).probabilities[temp_state][self.getIndexState((f.query)[0] ,query_state[ (f.query)[0].name  ] )];
            else:
                product = product * ((f.query)[0]).probabilities[query_state[ (f.query)[0].name  ]];
        #For Intermediate Factors
        for f in intermediate_factors.keys():
            temp_state = ();
            for m in f:
                temp_state = temp_state + (query_state[m.name],);
            product = product * intermediate_factors[f][temp_state];    
        return product;
        
    
    def listCopy(self, mylist):
        copier = [];
        for my in mylist:
            copier.append(my);
        return copier;
    
    def calculateProbabilities(self,ordering,evidence,query):
        #Find the elements to be eliminated
        for e in evidence.keys():
            ordering.remove(e);
        for q in query.keys():
            ordering.remove(q);
        #Variables Already eliminated 
        eliminated = [];
        #Stores the state of the intermediate tau variables 
        intermediate_state = dict();
        #Constants generated during elimination process. Somtimes , a factor will only contain variable to be eiliminated and/or evidence. So its a constant
        constants = [];
        #----------------------------- print "To be Eliminated" + str(ordering);
        # Variable Elimination Ordering 
        for o in ordering:
            #print o;
            #Get the factors and decide the state space of the resulting factor
            factors = self.var_Factors[o];
            factor_temp = [];
            #Tau factors used 
            tau_factors_used = [];
            #Remove any factors that have already been eliminated
            for f in factors:
                #Check query
                found = False; 
                for q in f.query:
                    if q.name in eliminated:
                        found = True;
                        break;
                for e in f.evidence:
                    if e.name in eliminated:
                        found = True;
                        break;
                if found == False:
                    factor_temp.append(f);      
            factors = factor_temp;    
            #Tau Factor Variables. Evidence variables should not be included.
            tau_factors = [];
            total_state_space = 1;
            #Tau factors that will be used in this iteration
            #---------------------------------------------------- print factors;
            for f in factors:
                if f.query in intermediate_state.keys() and self.checkAllTau(f.query) == True:
                    tau_factors_used.append(f);
            #print "Using  =--- " + str(tau_factors_used) + "\n";
            #shape = ();
            for fact in factors:
                #Get the Query and evidence variables for this factor
                Q = fact.query;
                E = fact.evidence;
                #Decide the state space. Exclude Evidence and Variable being eliminated
                for q in Q:
                    if q.name not in evidence.keys() and q.name != o and self.checkListContains(tau_factors, q.name) == False:
                        #temp = q.copy();
                        tau_factors.append(q);
                        total_state_space = total_state_space * q.nstates;
                        #shape = shape + (q.nstates,);
                for e in E:
                    if e.name not in evidence.keys() and e.name != o and self.checkListContains(tau_factors, e.name) == False:
                        #temp = e.copy();
                        tau_factors.append(e);
                        total_state_space = total_state_space * e.nstates;
                        #shape = shape + (e.nstates,);
            #Allocate an array to store the state 
            #tau_factor_state = zeros(total_state_space);
            #tau_factor_state.resize(shape);
            tau_factor_state = dict();
            states = []; 
            for _ in range(total_state_space):
                states.append(())
            #If the variable being eliminated will result in a constant 
            if not tau_factors:
                result, intermediate_state, factors = self.evalConstant(factors, evidence, o, intermediate_state);
                constants.append(result);
                #Scan for other similar entries
                count = 0;
                for k in intermediate_state.keys():
                    for f in factors:
                        if f.query == k:
                            count = count +  1;
                            
                while count > 0:
                    result, intermediate_state, factors = self.evalConstant(factors, evidence, o, intermediate_state);
                    constants.append(result);
                    count = count - 1;
                    
                eliminated.append(o);
                continue;  
    
            #Generate the states
            states = self.generateStates(tau_factors, 0, states, total_state_space, 1);
            #------------------- print "Tau Factors " + str(tau_factors) + "\n";
            #print "total " + str(total_state_space) + "len " + str(len(states)) + "States : " +  str(states) + "\n";
            #Run the variable elimination algorithm 
            for state in states:
                #No of additions will be number of states of Eliminated variable
                sum = 0;
                #print "---------- State of Tau ----- " + str(state);
                for e_state in self.var_Obj[o].state_names:
                    product = 1;
                    #print "State REmoved " + str(e_state);
                    for f in factors:
                        #print f;
                        #print intermediate_state.keys();
                        if f.query in intermediate_state.keys() and self.checkAllTau(f.query) == True:
                            #print "Here" + str(f.query);
                            #constantsThis is tau factor
                            temp_state = ();
                            for q in f.query:
                                if q.name == o:
                                    temp_state = temp_state + (e_state,);
                                else:
                                    temp_state = temp_state + (state[self.getTauVarIndex(tau_factors,q)],);
                            #print intermediate_state.keys()
                            #print intermediate_state[f.query];
                            #Lookup and muliply the probability
                            
                            product = product * intermediate_state[f.query][temp_state];
                        else:
                            #print "Inside Factors" + str(f);
                            #Analyze the factor
                            temp_state = ();
                            if f.evidence:
                                #print "Evidence Found"
                                #Prepare a state to query 
                                for ev in f.evidence:
                                        #This is observed variable. State is Known
                                    if ev.name in evidence.keys():
                                        temp_state = temp_state + (evidence[ev.name],);
                                    elif self.checkListContains(tau_factors,ev.name) == True:
                                        #This is a tau variable 
                                        temp_state = temp_state + (state[self.getTauVarIndex(tau_factors,ev)],);
                                    else:
                                        #This is a variable which is being eliminated
                                        temp_state = temp_state + (e_state,);
                            else:
                                #No Evidence
                                #print "No Evidence"
                                temp_state = temp_state + (e_state,);    
                            #print "temp State" + str(temp_state);
                            #What Type of factor it is :
                            if self.checkListContains(f.query,o) == True:
                                # Eliminated variable is in query/numerator
                                if f.evidence:
                                    #print "Tau " + str(tau_factors) + "Query " + str(f.query[0]) + "Trying to remove" + str(o);
                                    #print ((f.query)[0]).probabilities;
                                    product = product * ((f.query)[0]).probabilities[temp_state][self.getIndexState(self.var_Obj[o],e_state)];
                                else:
                                    product = product * ((f.query)[0]).probabilities[temp_state[0]];
                            elif self.checkListContains(f.evidence,o) == True:
                                #Eliminated variable is in evidence/denominator
                                #print "Tau " + str(tau_factors) + "Query " + str(f.query[0]) + "Trying to remove" + str(o);
                                if (f.query)[0].name in evidence.keys():
                                    temp_index = self.getIndexState(  self.var_Obj[(f.query)[0].name]  , evidence[(f.query)[0].name]);
                                    product = product * ((f.query)[0]).probabilities[temp_state][temp_index];
                                else:
                                    temp_index = self.getIndexState(  f.query[0] , state[self.getTauVarIndex(tau_factors,f.query[0])] );
                                    product = product * ((f.query)[0]).probabilities[temp_state][temp_index];
                        #print product; 
                    #All factors processed! 
                    sum = sum + product;
                    #print "Sum " + str(sum);
                    #print "Sum " + str(sum);
                #Save the Value for this state
                tau_factor_state[state] = sum;
            eliminated.append(o);
            #print tau_factor_state;
            #Add the newly generated intermediate node to all the lists
            tau_factors_copied = [];
            for tf in tau_factors:
                temp = tf.copy();
                temp.tau = True;
                tau_factors_copied.append(temp);
           
            new_f = bifParser.Factor( tuple(tau_factors_copied), ());
            for tf in tau_factors_copied:
                self.var_Factors[tf.name].append(new_f);
            
            #Save the state
            intermediate_state[tuple(tau_factors_copied)] = tau_factor_state;
            #Clean up -- Remove the tau factor that was used up in this iteration
            for tfu in tau_factors_used:
                for k in intermediate_state.keys():
                    if tfu.query == k:
                        del intermediate_state[k];
            #print intermediate_state.keys();
            
        #------- print "Intermediate State ---------" + str(intermediate_state);
        
        #For all the observed variables 
        for e in evidence.keys():
            factors = self.var_Factors[e];
            tu = [];
            for f in factors:
                found = False; 
                for q in f.query:
                    if q.name in eliminated or q.name in query:
                        found = True;
                        break;
                for ev in f.evidence:
                    if ev.name in eliminated or ev.name in query:
                        found = True;
                        break;
                if found == False:
                    tu.append(f); 
            if tu:  
                result, intermediate_state, tu = self.evalConstant(tu, evidence, e, intermediate_state);
                constants.append(result);
        #---------- print "Constants During Elimination -----" + str(constants);

        #For all the factors which contain only observed and/or query variables
        query_observed = [];
        for q in query.keys():
            factors = self.var_Factors[q];
            for f in factors:
                found = False; 
                for q in f.query:
                    if q.name in eliminated or q.tau == True:
                        found = True;
                        break;
                for ev in f.evidence:
                    if ev.name in eliminated or ev.tau == True:
                        found = True;
                        break;
                if found == False:
                    query_observed.append(f); 
        #------------------------------------------------- print query_observed;
                
        #Now generating all the possible states for query variables for renormalization
        states = [];
        query_var = [];
        total_state_space = 1; 
        for k in query.keys():
            query_var.append(self.var_Obj[k]);
            total_state_space = total_state_space * self.var_Obj[k].nstates;
        for _ in range(total_state_space):
            states.append(())
        states = self.generateStates(query_var, 0, states, total_state_space, 1);
        sum = 0;
        desired_prob = 0;
        for s in states:
            #Prepare a dict
            state_dict = dict();
            i = 0;
            for k in query.keys():
                state_dict[k] = s[i];
                i = i + 1;
            temp_sum = self.calIndividualQueryProb(constants, intermediate_state, query_observed, evidence, state_dict);
            if state_dict == query:
                desired_prob = temp_sum;
            sum = sum + temp_sum;
        desired_prob = desired_prob / float(sum);
        return desired_prob;
            

            
if __name__ == '__main__':
  args = list(sys.argv[1:])
  fname = args.pop(0);
  with open(fname) as f:
    vars = bifParser.BIFParser(f.read()).parse()
bg1 = BuildGraph(vars);
bg2 = BuildGraph(vars);
bg3 = BuildGraph(vars);
bg4 = BuildGraph(vars);
bg1.buildMarkovNetwork();
bg2.buildMarkovNetwork();
bg3.buildMarkovNetwork();
bg4.buildMarkovNetwork();

optimal_ordering = bg1.findEliminationOrdering();



evidence1 = dict();  query1 = dict();
evidence2 = dict();  query2 = dict();
evidence3 = dict();  query3 = dict();
evidence4 = dict();  query4 = dict();

query1["strokevolume"] = "high";
evidence1["errcauter"] = "true"; evidence1["hypovolemia"] = "true"; evidence1["pvsat"] = "normal";
evidence1["disconnect"] = "true"; evidence1["minvolset"] = "low";

query2["hrbp"] = "normal";
evidence2["lvedvolume"] = "normal"; evidence2["anaphylaxis"] = "true"; evidence2["press"] = "zero";
evidence2["venttube"] = "zero"; evidence2["bp"] = "high";

query3["lvfailure"] = "false";
evidence3["hypovolemia"] = "true"; evidence3["minvolset"] = "low"; evidence3["ventlung"] = "normal";
evidence3["bp"] = "normal"; 
 
 
query4["pvsat"] = "normal"; query4["cvp"] = "normal";
evidence4["lvedvolume"] = "high";
evidence4["anaphylaxis"] = "false";
evidence4["press"] = "zero";


print " Probability of conditional query 1 is  " + str(bg1.calculateProbabilities(bg1.listCopy(optimal_ordering),evidence1 , query1));
print " Probability of conditional query 2 is " + str(bg2.calculateProbabilities(bg2.listCopy(optimal_ordering),evidence2 , query2));
print " Probability of conditional query 3 is " + str(bg3.calculateProbabilities(bg3.listCopy(optimal_ordering),evidence3 , query3));
print " Probability of conditional query 4 is " + str(bg4.calculateProbabilities(bg4.listCopy(optimal_ordering),evidence4 , query4));
