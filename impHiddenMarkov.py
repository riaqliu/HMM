from impProbabilityClasses import PMatrix, PVector
from itertools import product
from functools import reduce

class HiddenMarkovModel:
    def __init__(self, transmissionMatrix, emissionMatrix, initialTransmission):
        self.tM = transmissionMatrix
        self.eM = emissionMatrix
        self.iT = initialTransmission

        self.states = self.iT.states
        self.observables = self.eM.observables
    
    @classmethod
    def initialize(cls, states: list, observables: list):
        return cls(\
            PMatrix(states, states), \
            PMatrix(states, observables), \
            PVector(states)\
            )
    
    
    def score(self, observations: list) -> float:
        def mul(x, y): return x * y

        score = 0
        chains = list(product(*(self.states,) * len(observations)))
        
        for c in chains:
            expanded_chain = list(zip(c, [self.tM.states[0]] + list(c)))
            expanded_obser = list(zip(observations, c))
            
            p_observations = list(map(lambda x: self.eM.df.loc[x[1], x[0]], expanded_obser))
            p_hidden_state = list(map(lambda x: self.tM.df.loc[x[1], x[0]], expanded_chain))
            p_hidden_state[0] = self.iT[c[0]]
            
            score += reduce(mul, p_observations) * reduce(mul, p_hidden_state)
        return score


# TODO: implement methods to answer problem 2 and 3 in
# https://medium.com/@kangeugine/hidden-markov-model-7681c22f5b9