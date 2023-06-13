from impProbabilityClasses import PMatrix, PVector
from itertools import product
from functools import reduce

class HiddenMarkovModel:
    def __init__(self, transmissionMatrix:PMatrix, emissionMatrix:PMatrix, initialTransmission:PVector):
        self.tM:PMatrix = transmissionMatrix
        self.eM:PMatrix = emissionMatrix
        self.iT:PVector = initialTransmission

        self.states = self.iT.states
        self.observables = self.eM.observables
    
    @classmethod
    def initialize(cls, states: list, observables: list):
        return cls(
            PMatrix(states, states), 
            PMatrix(states, observables), 
            PVector(states)
            )
    
    def extract_chain(self, c:list, observations, func) -> tuple[list]:
        expandedChain = list(zip(c, [self.tM.states[0]] + list(c)))
        expandedObser = list(zip(observations, c))
        pObservations = list(map(lambda x: self.eM.df.loc[x[1], x[0]], expandedObser))
        pHiddenState = list(map(lambda x: self.tM.df.loc[x[1], x[0]], expandedChain))
        pHiddenState[0] = self.iT[c[0]]

        x = reduce(func, pObservations) * reduce(func, pHiddenState)

        return (expandedChain, expandedObser, pObservations, pHiddenState, x)
    
    def score(self, observations: list) -> float:
        score = 0
        chains = list(product(*(self.states,) * len(observations)))
        def mul(x, y): return x * y
        for c in chains:
            x = self.extract_chain(c,observations, mul)[-1]
            score += x
        return score
    
    def decode(self, observations:list) -> tuple[float, list]:
        def mul(x,y): return x * y

        chains = list(product(*(self.states,) * len(observations)))
        pList = list()
        for c in chains:
            _, expandedObser, _, _, x = self.extract_chain(c,observations, mul)
            seq = list(zip(*expandedObser))[1]
            pList.append((x, seq))
        return max(pList, key=lambda k : k[0])

# TODO: implement methods to answer problem 3 in
# https://medium.com/@kangeugine/hidden-markov-model-7681c22f5b9

if __name__ == "__main__":
    pass