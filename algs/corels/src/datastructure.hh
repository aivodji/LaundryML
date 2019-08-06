struct confusion_matrix {
    int nTP;
    int nFP;
    int nFN;
    int nTN;
    double nPPV;
    double nTPR;
    double nFDR;
    double nFPR;
    double nFOR;
    double nFNR;
    double nNPV;
    double nTNR;
};

struct confusion_matrix_groups {
    confusion_matrix minority;
    confusion_matrix majority;
};

struct fairness_metrics {
    double statistical_parity;
    double predictive_parity;
    double predictive_equality;
    double equal_opportunity;
    //double conditional_procedure_accuracy_equality;
    //double conditional_use_accuracy_equality;
    //double overall_accuracy_equality;
    //double treatment_equality;
};
