from __future__ import annotations
import numpy as np
import pandas as pd


class PVector:
    def __init__(self, probabilities: dict):
        self.states = sorted(probabilities)
        self.values = np.array(list(map(lambda x: 
            probabilities[x], self.states))).reshape(1, -1)

    def __getitem__(self, state: str) -> float:
        return float(self.values[0, self.states.index(state)])
    

    
class PMatrix:
    def __init__(self, pVectors: dict[PVector]) -> None:
        self.states = sorted(pVectors)
        self.observables = pVectors[self.states[0]].states
        self.values = np.stack([pVectors[x].values
                                for x in self.states
                                ]).squeeze()
    
    @property
    def df(self):
        return pd.DataFrame(self.values, columns=self.observables, index=self.states)
