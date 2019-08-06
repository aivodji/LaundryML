import pandas as pd
import numpy as np
import seaborn as sns


import matplotlib.pyplot as plt

from collections import namedtuple


class ConfusionMatrix(namedtuple('ConfusionMatrix', 'minority majority label truth')):
    def get_matrix(self):
        
        TP = np.logical_and(self.label == 1, self.truth == 1)
        FP = np.logical_and(self.label == 1, self.truth == 0)
        FN = np.logical_and(self.label == 0, self.truth == 1)
        TN = np.logical_and(self.label == 0, self.truth == 0)

        #maj
        TP_maj = np.logical_and(TP == 1, self.majority == 1)
        FP_maj = np.logical_and(FP == 1, self.majority == 1)
        FN_maj = np.logical_and(FN == 1, self.majority == 1)
        TN_maj = np.logical_and(TN == 1, self.majority == 1)

        nTP_maj = np.sum(TP_maj)
        nFN_maj = np.sum(FN_maj)
        nFP_maj = np.sum(FP_maj)
        nTN_maj = np.sum(TN_maj)

        nPPV_maj = float(nTP_maj) / max((nTP_maj + nFP_maj), 1)
        nTPR_maj = float(nTP_maj) / max((nTP_maj + nFN_maj), 1)

        nFDR_maj = float(nFP_maj) / max((nFP_maj + nTP_maj), 1)
        nFPR_maj = float(nFP_maj) / max((nFP_maj + nTN_maj), 1)

        nFOR_maj = float(nFN_maj) / max((nFN_maj + nTN_maj), 1)
        nFNR_maj = float(nFN_maj) / max((nFN_maj + nTP_maj), 1)

        nNPV_maj = float(nTN_maj) / max((nTN_maj + nFN_maj), 1)
        nTNR_maj = float(nTN_maj) / max((nTN_maj + nFP_maj), 1)

        #min
        TP_min = np.logical_and(TP == 1, self.minority == 1)
        FP_min = np.logical_and(FP == 1, self.minority == 1)
        FN_min = np.logical_and(FN == 1, self.minority == 1)
        TN_min = np.logical_and(TN == 1, self.minority == 1)

        
        nTP_min = np.sum(TP_min)
        nFN_min = np.sum(FN_min)
        nFP_min = np.sum(FP_min)
        nTN_min = np.sum(TN_min)

        nPPV_min = float(nTP_min) / max((nTP_min + nFP_min), 1)
        nTPR_min = float(nTP_min) / max((nTP_min + nFN_min), 1)

        nFDR_min = float(nFP_min) / max((nFP_min + nTP_min), 1)
        nFPR_min = float(nFP_min) / max((nFP_min + nTN_min), 1)

        nFOR_min = float(nFN_min) / max((nFN_min + nTN_min), 1)
        nFNR_min = float(nFN_min) / max((nFN_min + nTP_min), 1)

        nNPV_min = float(nTN_min) / max((nTN_min + nFN_min), 1)
        nTNR_min = float(nTN_min) / max((nTN_min + nFP_min), 1)

        matrix_maj = {
            'TP' : nTP_maj,
            'FP' : nFP_maj,
            'FN' : nFN_maj,
            'TN' : nTN_maj,
            'PPV' : nPPV_maj,
            'TPR' : nTPR_maj,
            'FDR' : nFDR_maj,
            'FPR' : nFPR_maj,
            'FOR' : nFOR_maj,
            'FNR' : nFNR_maj,
            'NPV' : nPPV_maj,
            'TNR' : nTNR_maj}

        matrix_min = {
            'TP' : nTP_min,
            'FP' : nFP_min,
            'FN' : nFN_min,
            'TN' : nTN_min,
            'PPV' : nPPV_min,
            'TPR' : nTPR_min,
            'FDR' : nFDR_min,
            'FPR' : nFPR_min,
            'FOR' : nFOR_min,
            'FNR' : nFNR_min,
            'NPV' : nPPV_min,
            'TNR' : nTNR_min}

        return matrix_maj, matrix_min

class FairnessMetric(namedtuple('FairnessMetric', 'cm_majority cm_minority')):
    def statistical_parity(self):
        statistical_parity_maj = float(self.cm_majority['TP'] + self.cm_majority['FP']) / max((self.cm_majority['TP'] + self.cm_majority['FP'] + self.cm_majority['FN'] + self.cm_majority['TN']), 1)
        statistical_parity_min = float(self.cm_minority['TP'] + self.cm_minority['FP']) / max((self.cm_minority['TP'] + self.cm_minority['FP'] + self.cm_minority['FN'] + self.cm_minority['TN']), 1)
        return np.fabs(statistical_parity_maj - statistical_parity_min)
    
    def predictive_parity(self):
        return np.fabs(self.cm_majority['PPV'] - self.cm_minority['PPV'])

    def predictive_equality(self):
        return np.fabs(self.cm_majority['FPR'] - self.cm_minority['FPR'])

    def equal_opportunity(self):
        return np.fabs(self.cm_majority['FNR'] - self.cm_minority['FNR'])



