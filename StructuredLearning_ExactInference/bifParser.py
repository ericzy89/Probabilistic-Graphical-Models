import sys
import numpy as np

class Variable(object):
    def __init__(self, id, name, state_names):
        self.id = id
        self.name = name
        self.state_names = state_names
        self.nstates = len(state_names)
        self.cond_vars = []
        self.probs = None
        #Better way of storing probabilities
        self.probabilities = dict();
        self.tau = False;
        self.restrict_ind = None

    def restrict_state(self, state_name):
        ind = self.state_names.index(state_name)
        self.restrict_ind = ind
        self.state_names = self.state_names[ind:ind+1]
        self.probs = self.probs[..., ind:ind+1]
        self.nstates = 1
    
    def copy(self):
        v = Variable(self.id,self.name,self.state_names);
        v.cond_vars = self.cond_vars;
        v.probs = self.probs;
        v.probabilities = self.probabilities;
        v.tau = self.tau;
        return v;
    
    def __repr__(self):
        return 'var:' + self.name + " tau " + str(self.tau);


class Factor(object):
    def __init__(self, query, evidence):
        self.query    = query;
        self.evidence = evidence;
        
    def __repr__(self):
        return 'fac:' + str(self.query) + " evide " + str(self.evidence) + "\n";

class BIFParser(object):
    def __init__(self, s):
        for c in ',;|()[]':
            s = s.replace(c, ' ')
        self.toks = s.lower().split()
        self.vars = {}
        self.nvars = 0
        self._block_parsers = {
            'network': self._parse_network,
            'variable': self._parse_variable,
            'probability': self._parse_probability,
            }

    def parse(self):
        while self.toks:
            t = self._next()
            self._block_parsers[t]()
        return self.vars.values()

    def _next(self, assert_tok=None):
        next = self.toks.pop(0)
        if assert_tok:
            assert next == assert_tok
        return next

    def _next_to(self, end_tok):
        end = self.toks.index(end_tok)
        t = self.toks[:end]
        del self.toks[:end+1]
        return t

    def _next_n(self, n):
        t = self.toks[:n]
        del self.toks[:n]
        return t

    def _parse_network(self):
        name = self._next()
        self._next('{')
        self._next('}')

    def _parse_variable(self):
        name = self._next()
        self._next('{')
        self._next('type')
        self._next('discrete')
        nstates = int(self._next())
        self._next('{')
        state_names = self._next_to('}')
        assert len(state_names) == nstates
        self.vars[name] = Variable(self.nvars, name, state_names)
        self.nvars += 1
        self._next('}')

    def _parse_probability(self):
        var = self.vars[self._next()]
        assert var.probs is None # shoudln't have probs defined twice
        cond_vars = [self.vars[x] for x in self._next_to('{')]
        var.cond_vars = cond_vars
        if cond_vars:
            # conditional prob
            ncond = len(cond_vars)
            dims = [v.nstates for v in cond_vars] + [var.nstates]
            probs = np.zeros(dims)
            while self.toks[0] != '}':
                cond_states = self._next_n(ncond)
                state_probs = map(float, self._next_n(var.nstates))
                state = tuple([v.state_names.index(s)
                               for (v, s) in zip(cond_vars, cond_states)])
                probs[state] = state_probs
                var.probabilities[tuple(cond_states)] = state_probs;
            var.probs = probs
        else:
            # terminal prob
            self._next('table')
            var.probs = np.array(map(float, self._next_n(var.nstates)))
            #Store the State in var.probabilities
            i = 0;
            for s in var.state_names:
                var.probabilities[s] = var.probs[i];
                i = i + 1; 
        self._next('}')

