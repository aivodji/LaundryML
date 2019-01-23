struct fairness_metrics {
    double overall_accuracy_equality;
    double statistical_parity_sum;
    double statistical_parity_max;
    double conditional_procedure_accuracy_equality_sum;
    double conditional_procedure_accuracy_equality_max;
    double conditional_use_accuracy_equality_sum;
    double conditional_use_accuracy_equality_max;
    double treatment_equality;
    double total_fairness;
};

