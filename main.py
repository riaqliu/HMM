from impProbabilityClasses import PVector, PMatrix

from impHiddenMarkov import HiddenMarkovModel


def score(hmm:HiddenMarkovModel, observations:list[str]) -> None:
    print(f"Probability of the observation {observations} is {hmm.score(observations):0.5f}.")    

def decode(hmm:HiddenMarkovModel, observations:list[str]) -> None:
    probability, weather = hmm.decode(observations)
    print(f"The weather was most likely {weather} with {probability*100.0:.5f}% probability")


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

hmm = HiddenMarkovModel(Hidden, Observable, start)

# Probability that the first observation is 'Walk'
score(hmm,['Walk']) 

# Probability that the first observation is 'Shop'
score(hmm,['Shop']) 

# Probability that the first observation is 'Clean'
score(hmm,['Clean']) 

# Probability that the sequence 'Clean', 'Clean', 'Clean' occurs
score(hmm,['Clean','Clean','Clean']) 

# Given the observation 'Shop', 'Clean', 'Walk', find the most likely weather
decode(hmm,['Shop','Clean','Walk'])

# Given observation 'Clean', 'Clean', 'Clean', find the most likely weather
decode(hmm,['Clean','Clean','Clean'])

# Given observation 'Clean', 'Clean', 'Clean', find the most likely weather
decode(hmm,['Shop','Shop','Walk'])

# Given observation 'Clean', 'Clean', 'Clean', find the most likely weather
decode(hmm,['Walk','Clean','Walk'])
