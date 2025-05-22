# Author: Ali Aral Takak / Student ID: 22001758
# Import the required libraries.
import os
import csv
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
from skmultiflow.data import AGRAWALGenerator, SEAGenerator, FileStream
from skmultiflow.trees import HoeffdingTreeClassifier
from skmultiflow.meta import AdaptiveRandomForestClassifier as ARF
from skmultiflow.lazy import SAMKNNClassifier as SAMkNN
from skmultiflow.drift_detection import DDM, EDDM, ADWIN

# File and model configurations.
np.random.seed(22001758)
STREAM_LENGTH    = 100_000
SEGMENTS         = 20
WINDOW_SIZE      = 100
PASSIVE_THRESH   = 0.65
LEARNER_COUNT    = 5
DATA_DIR         = '/Users/aral/Documents/Bilkent Archive/GE 461 - Introduction to Data Science/Assignment 5/Datasets'
OUTPUT_DIR       = 'outputs'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Define active and passive ensemble classes.
class ActiveEnsemble:
    def __init__(self, learner_count=LEARNER_COUNT, window_size=WINDOW_SIZE, detectors=None):
        if detectors is None: detectors = [DDM, EDDM, ADWIN]
        self.learners  = [HoeffdingTreeClassifier() for _ in range(learner_count)]
        self.detectors = [detectors[i % len(detectors)]() for i in range(learner_count)]
        self.windows   = [deque(maxlen=window_size) for _ in range(learner_count)]

    def predict(self, X):
        votes   = [clf.predict(X)[0] for clf in self.learners]
        weights = [(np.mean(w) if w else 1.0/len(self.learners)) for w in self.windows]
        agg = {}
        for v,wt in zip(votes, weights):
            agg[v] = agg.get(v,0) + wt
        return max(agg, key=agg.get)

    def partial_fit(self, X, y):
        for i,(clf,det,win) in enumerate(zip(self.learners,self.detectors,self.windows)):
            pred    = clf.predict(X)[0]
            correct = int(pred==y[0])
            det.add_element(0 if correct else 1)
            win.append(correct)
            if det.detected_change():
                self.learners[i]  = HoeffdingTreeClassifier()
                self.detectors[i] = type(det)()
                self.windows[i]   = deque(maxlen=win.maxlen)
            clf.partial_fit(X, y, classes=[0,1])

class PassiveEnsemble:
    def __init__(self, learner_count=LEARNER_COUNT, window_size=WINDOW_SIZE, threshold=PASSIVE_THRESH):
        self.learners  = [HoeffdingTreeClassifier() for _ in range(learner_count)]
        self.windows   = [deque(maxlen=window_size) for _ in range(learner_count)]
        self.threshold = threshold

    def predict(self, X):
        votes   = [clf.predict(X)[0] for clf in self.learners]
        weights = [(np.mean(w) if w else 1.0/len(self.learners)) for w in self.windows]
        agg = {}
        for v,wt in zip(votes, weights):
            agg[v] = agg.get(v,0) + wt
        return max(agg, key=agg.get)

    def partial_fit(self, X, y):
        for i,(clf,win) in enumerate(zip(self.learners,self.windows)):
            pred    = clf.predict(X)[0]
            correct = int(pred==y[0])
            win.append(correct)
            if len(win)==win.maxlen and np.mean(win)<self.threshold:
                self.learners[i] = HoeffdingTreeClassifier()
                self.windows[i]  = deque(maxlen=win.maxlen)
            clf.partial_fit(X, y, classes=[0,1])
