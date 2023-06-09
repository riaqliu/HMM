from __future__ import annotations
import numpy as np
import pandas as pd


class PVector:
    def __init__(self, probabilities: dict):
        self.states = sorted(probabilities)
        self.values = np.array(list(map(lambda x: 
            probabilities[x], self.states))).reshape(1, -1)
        
    @classmethod
    def initialize(cls, states: list):
        size = len(states)
        rand = np.random.rand(size) / (size**2) + 1 / size
        rand /= rand.sum(axis=0)
        return cls(dict(zip(states, rand)))

    def __getitem__(self, state: str) -> float:
        return float(self.values[0, self.states.index(state)])
    

    
class PMatrix:
    def __init__(self, pVectors: dict[PVector]) -> None:
        self.states = sorted(pVectors)
        self.observables = pVectors[self.states[0]].states
        self.values = np.stack([pVectors[x].values
                                for x in self.states
                                ]).squeeze()

    @classmethod
    def initialize(cls, states:list, observables: list):
        size = len(states)
        rand = np.random.rand(size, len(observables)) / (size**2) + 1 / size
        rand /= rand.sum(axis=1).reshape(-1,1)
        aggr = [dict(zip(observables, rand[i, :])) for i in range(size)]
        pvec = [PVector(x) for x in aggr]
        return cls(dict(zip(states, pvec)))
    
    @property
    def df(self):
        return pd.DataFrame(self.values, columns=self.observables, index=self.states)
