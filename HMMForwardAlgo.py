from __future__ import annotations
from impHiddenMarkov import HiddenMarkovModel



class FAHiddenMarkovModel(HiddenMarkovModel):
    def score(self, observations: list) -> float:
        observations = ['Clean','Clean','Clean']
        
        #Initialization

        return self.forwardAlgorithm(observations)
    
    def forwardAlgorithm(self, observations: list) -> list:

        return 




if __name__ == "__main__":
    pass