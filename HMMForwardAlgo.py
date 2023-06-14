from __future__ import annotations
from impHiddenMarkov import HiddenMarkovModel
from impProbabilityClasses import PVector, PMatrix
from itertools import product



class FAHiddenMarkovModel(HiddenMarkovModel):
    def score(self, observations: list) -> float:
        return sum([self.iT[x] * self.eM.df.loc[x,observations[0]] * self.forwardAlgorithm(observations[1:],x) for x in self.states])
    
    def forwardAlgorithm(self, observations: list, prev: str) -> float:
        if len(observations) == 0:
            return 1
        return sum([self.tM.df.loc[prev,x] * self.eM.df.loc[x,observations[0]] * self.forwardAlgorithm(observations[1:],x) for x in self.states])




if __name__ == "__main__":
    def score(hmm:HiddenMarkovModel, observations:list[str]) -> None:
        print(f"Probability of the observation {observations} is {hmm.score(observations):0.5f}.")    


    # Hidden States
    r1 = PVector({'Rainy':0.7, 'Sunny':0.3})
    s1 = PVector({'Rainy':0.4, 'Sunny':0.6})

    # Observable States
    r2 = PVector({'Walk':0.1, 'Shop':0.4, 'Clean':0.5})
    s2 = PVector({'Walk':0.6, 'Shop':0.3, 'Clean':0.1})

    # Start
    start = PVector({'Rainy':0.6, 'Sunny':0.4})

    # Matrices
    Hidden = PMatrix({'Rainy': r1, 'Sunny': s1})
    Observable = PMatrix({'Rainy': r2, 'Sunny': s2})

    hmm = FAHiddenMarkovModel(Hidden, Observable, start)
    # score(hmm,['Walk']) 
    score(hmm,['Clean','Clean','Clean']) 
