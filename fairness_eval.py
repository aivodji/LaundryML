import pandas as pd
import numpy as np
import seaborn as sns


import matplotlib.pyplot as plt

from collections import namedtuple

class FairnessEvaluator(namedtuple('FairnessEvaluator', 'minority majority label')):
    def demographic_parity_discrimination(self):
        yes_majority_prop = np.mean(np.logical_and(self.majority == 1, self.label == 1))
        yes_minority_prop = np.mean(np.logical_and(self.minority == 1, self.label == 1))

        dem_parity = abs(yes_majority_prop - yes_minority_prop)

        #print(dem_parity)

        return dem_parity

