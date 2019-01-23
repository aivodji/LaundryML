#include "queue.hh"
#include <algorithm>
#include <iostream>
#include <sys/resource.h>
#include <stdio.h>


fairness_metrics compute_fairness_metrics(CacheTree* tree, VECTOR parent_not_captured, VECTOR captured){

    fairness_metrics all_metrics;

    VECTOR not_captured;
    int num_not_captured;
    rule_vinit(tree->nsamples(), &not_captured);

    rule_vandnot(not_captured, parent_not_captured, captured, tree->nsamples(), &num_not_captured);

    // true positives, false negatives, true negatives, and false positives tables
    VECTOR A, B, D, C;
    rule_vinit(tree->nsamples(), &A);
    rule_vinit(tree->nsamples(), &B);
    rule_vinit(tree->nsamples(), &D);
    rule_vinit(tree->nsamples(), &C);

    mpz_and(A, tree->label(0).truthtable, captured);
    mpz_and(B, tree->label(0).truthtable, not_captured);
    mpz_and(D, tree->label(1).truthtable, not_captured);
    mpz_and(C, tree->label(1).truthtable, captured);

    
    // true positives, false negatives, true negatives, and false positives tables for majority group
    VECTOR A_maj, B_maj, D_maj, C_maj;
    rule_vinit(tree->nsamples(), &A_maj);
    rule_vinit(tree->nsamples(), &B_maj);
    rule_vinit(tree->nsamples(), &D_maj);
    rule_vinit(tree->nsamples(), &C_maj);
    mpz_and(A_maj, A, tree->rule(1).truthtable);
    mpz_and(B_maj, B, tree->rule(1).truthtable);
    mpz_and(D_maj, D, tree->rule(1).truthtable);
    mpz_and(C_maj, C, tree->rule(1).truthtable);

    // true positives, false negatives, true negatives, and false positives tables for minority group
    VECTOR A_min, B_min, D_min, C_min;
    rule_vinit(tree->nsamples(), &A_min);
    rule_vinit(tree->nsamples(), &B_min);
    rule_vinit(tree->nsamples(), &D_min);
    rule_vinit(tree->nsamples(), &C_min);
    mpz_and(A_min, A, tree->rule(2).truthtable);
    mpz_and(B_min, B, tree->rule(2).truthtable);
    mpz_and(D_min, D, tree->rule(2).truthtable);
    mpz_and(C_min, C, tree->rule(2).truthtable);

    int nA_maj = mpz_popcount(A_maj);
    int nB_maj = mpz_popcount(B_maj);
    int nD_maj = mpz_popcount(D_maj);
    int nC_maj = mpz_popcount(C_maj);


    int nA_min = mpz_popcount(A_min);
    int nB_min = mpz_popcount(B_min);
    int nD_min = mpz_popcount(D_min);
    int nC_min = mpz_popcount(C_min);

    // overall accuracy equality
    double overall_accuracy_equality_maj = (double) (nA_maj + nD_maj)/(nA_maj + nB_maj + nC_maj + nD_maj);
    double overall_accuracy_equality_min = (double) (nA_min + nD_min)/(nA_min + nB_min + nC_min + nD_min);
    all_metrics.overall_accuracy_equality = abs(overall_accuracy_equality_maj - overall_accuracy_equality_min);

    // statistical parity
    double statistical_parity_maj_0 = (double) (nA_maj + nC_maj)/(nA_maj + nB_maj + nC_maj + nD_maj);
    double statistical_parity_maj_1 = (double) (nB_maj + nD_maj)/(nA_maj + nB_maj + nC_maj + nD_maj);
    double statistical_parity_min_0 = (double) (nA_min + nC_min)/(nA_min + nB_min + nC_min + nD_min);
    double statistical_parity_min_1 = (double) (nB_min + nD_min)/(nA_min + nB_min + nC_min + nD_min);
    all_metrics.statistical_parity_sum = abs(statistical_parity_maj_0 - statistical_parity_min_0) + abs(statistical_parity_maj_1 - statistical_parity_min_1);
    all_metrics.statistical_parity_max = max(abs(statistical_parity_maj_0 - statistical_parity_min_0), abs(statistical_parity_maj_1 - statistical_parity_min_1));

    // conditional procedure accuracy equality
    double conditional_procedure_accuracy_equality_maj_0 = (double) (nA_maj)/(nA_maj + nB_maj);
    double conditional_procedure_accuracy_equality_maj_1 = (double) (nD_maj)/(nC_maj + nD_maj);
    double conditional_procedure_accuracy_equality_min_0 = (double) (nA_min)/(nA_min + nB_min);
    double conditional_procedure_accuracy_equality_min_1 = (double) (nD_min)/(nC_min + nD_min);
    all_metrics.conditional_procedure_accuracy_equality_sum = abs(conditional_procedure_accuracy_equality_maj_0 - conditional_procedure_accuracy_equality_min_0) + abs(conditional_procedure_accuracy_equality_maj_1 - conditional_procedure_accuracy_equality_min_1);
    all_metrics.conditional_procedure_accuracy_equality_max = max(abs(conditional_procedure_accuracy_equality_maj_0 - conditional_procedure_accuracy_equality_min_0), abs(conditional_procedure_accuracy_equality_maj_1 - conditional_procedure_accuracy_equality_min_1));


    // conditional use accuracy equality
    double conditional_use_accuracy_equality_maj_0 = (double) (nA_maj)/(nA_maj + nC_maj);
    double conditional_use_accuracy_equality_maj_1 = (double) (nD_maj)/(nB_maj + nD_maj);
    double conditional_use_accuracy_equality_min_0 = (double) (nA_min)/(nA_min + nC_min);
    double conditional_use_accuracy_equality_min_1 = (double) (nD_min)/(nB_min + nD_min);
    all_metrics.conditional_use_accuracy_equality_sum = abs(conditional_use_accuracy_equality_maj_0 - conditional_use_accuracy_equality_min_0) + abs(conditional_use_accuracy_equality_maj_1 - conditional_use_accuracy_equality_min_1);
    all_metrics.conditional_use_accuracy_equality_max = max(abs(conditional_use_accuracy_equality_maj_0 - conditional_use_accuracy_equality_min_0), abs(conditional_use_accuracy_equality_maj_1 - conditional_use_accuracy_equality_min_1));


    // treatment equality
    double treatment_maj;
    if (nB_maj != 0){
        treatment_maj = (double) nC_maj / nB_maj;
    } else {
        treatment_maj = 0.0;
    }

    double treatment_min;
    if (nB_min != 0){
        treatment_min = (double) nC_min / nB_min;
    } else {
        treatment_min = 0.0;
    }

    all_metrics.treatment_equality = abs(treatment_maj - treatment_min);
    //all_metrics.treatment_equality = max(treatment_maj, treatment_min);




    rule_vfree(&not_captured);

    rule_vfree(&A);
    rule_vfree(&B);
    rule_vfree(&D);
    rule_vfree(&C);

    rule_vfree(&A_maj);
    rule_vfree(&B_maj);
    rule_vfree(&D_maj);
    rule_vfree(&C_maj);

    rule_vfree(&A_min);
    rule_vfree(&B_min);
    rule_vfree(&D_min);
    rule_vfree(&C_min);

    //return (statistical_parity < 0.01) ? 0 : statistical_parity;

    return all_metrics;

}


double computefairness(CacheTree* tree, VECTOR parent_not_captured, VECTOR captured){

    VECTOR not_captured;
    int num_not_captured;
    rule_vinit(tree->nsamples(), &not_captured);

    rule_vandnot(not_captured, parent_not_captured, captured, tree->nsamples(), &num_not_captured);

    // true positives, false negatives, true negatives, and false positives tables
    VECTOR TP, FN, TN, FP;
    rule_vinit(tree->nsamples(), &TP);
    rule_vinit(tree->nsamples(), &FN);
    rule_vinit(tree->nsamples(), &TN);
    rule_vinit(tree->nsamples(), &FP);

    /*mpz_and(TP, tree->label(0).truthtable, captured);
    mpz_and(FN, tree->label(0).truthtable, not_captured);
    mpz_and(TN, tree->label(1).truthtable, not_captured);
    mpz_and(FP, tree->label(1).truthtable, captured);*/

    mpz_and(TP, tree->label(0).truthtable, captured);
    mpz_and(FN, tree->label(0).truthtable, not_captured);
    mpz_and(TN, tree->label(1).truthtable, not_captured);
    mpz_and(FP, tree->label(1).truthtable, captured);

    /*int nTP = mpz_popcount(TP);
    int nFN = mpz_popcount(FN);
    int nTN = mpz_popcount(TN);
    int nFP = mpz_popcount(FP);*/


    // true positives, false negatives, true negatives, and false positives tables for majority group
    VECTOR TP_maj, FN_maj, TN_maj, FP_maj;
    rule_vinit(tree->nsamples(), &TP_maj);
    rule_vinit(tree->nsamples(), &FN_maj);
    rule_vinit(tree->nsamples(), &TN_maj);
    rule_vinit(tree->nsamples(), &FP_maj);
    mpz_and(TP_maj, TP, tree->rule(1).truthtable);
    mpz_and(FN_maj, FN, tree->rule(1).truthtable);
    mpz_and(TN_maj, TN, tree->rule(1).truthtable);
    mpz_and(FP_maj, FP, tree->rule(1).truthtable);

    int nTP_maj = mpz_popcount(TP_maj);
    int nFN_maj = mpz_popcount(FN_maj);
    int nTN_maj = mpz_popcount(TN_maj);
    int nFP_maj = mpz_popcount(FP_maj);


    // true positives, false negatives, true negatives, and false positives tables for minority group
    VECTOR TP_min, FN_min, TN_min, FP_min;
    rule_vinit(tree->nsamples(), &TP_min);
    rule_vinit(tree->nsamples(), &FN_min);
    rule_vinit(tree->nsamples(), &TN_min);
    rule_vinit(tree->nsamples(), &FP_min);
    mpz_and(TP_min, TP, tree->rule(2).truthtable);
    mpz_and(FN_min, FN, tree->rule(2).truthtable);
    mpz_and(TN_min, TN, tree->rule(2).truthtable);
    mpz_and(FP_min, FP, tree->rule(2).truthtable);

    int nTP_min = mpz_popcount(TP_min);
    int nFN_min = mpz_popcount(FN_min);
    int nTN_min = mpz_popcount(TN_min);
    int nFP_min = mpz_popcount(FP_min);


    /*printf("TP =========> %d\n", nTP);
    printf("FN =========> %d\n", nFN);
    printf("TN =========> %d\n", nTN);
    printf("FP =========> %d\n", nFP);

    printf("TP_maj =========> %d\n", nTP_maj);
    printf("FN_maj =========> %d\n", nFN_maj);
    printf("TN_maj =========> %d\n", nTN_maj);
    printf("FP_maj =========> %d\n", nFP_maj);

    printf("TP_min =========> %d\n", nTP_min);
    printf("FN_min =========> %d\n", nFN_min);
    printf("TN_min =========> %d\n", nTN_min);
    printf("FP_min =========> %d\n", nFP_min);*/

    double ratioA_maj = (double) (nTP_maj + nFP_maj) / (nTP_maj + nFN_maj + nTN_maj + nFP_maj);
    double ratioB_maj = (double) (nFN_maj + nTN_maj) / (nTP_maj + nFN_maj + nTN_maj + nFP_maj);

    double ratioA_min = (double) (nTP_min + nFP_min) / (nTP_min + nFN_min + nTN_min + nFP_min);
    double ratioB_min = (double) (nFN_min + nTN_min) / (nTP_min + nFN_min + nTN_min + nFP_min);

    double statistical_parity = 0.5*abs(ratioA_maj - ratioA_min) + 0.5*abs(ratioB_maj - ratioB_min);

    rule_vfree(&not_captured);

    rule_vfree(&TP);
    rule_vfree(&FN);
    rule_vfree(&TN);
    rule_vfree(&FP);

    rule_vfree(&TP_maj);
    rule_vfree(&FN_maj);
    rule_vfree(&TN_maj);
    rule_vfree(&FP_maj);

    rule_vfree(&TP_min);
    rule_vfree(&FN_min);
    rule_vfree(&TN_min);
    rule_vfree(&FP_min);

    //return (statistical_parity < 0.01) ? 0 : statistical_parity;

    return statistical_parity;

}


Queue::Queue(std::function<bool(Node*, Node*)> cmp, char const *type)
    : q_(new q (cmp)), type_(type) {}


/*
 * Performs incremental computation on a node, evaluating the bounds and inserting into the cache,
 * queue, and permutation map if appropriate.
 * This is the function that contains the majority of the logic of the algorithm.
 *
 * parent -- the node that is going to have all of its children evaluated.
 * parent_not_captured -- the vector representing data points NOT captured by the parent.
 */
void evaluate_children(CacheTree* tree, Node* parent, tracking_vector<unsigned short, DataStruct::Tree> parent_prefix,
        VECTOR parent_not_captured, Queue* q, PermutationMap* p, double beta) {

    VECTOR captured, captured_zeros, not_captured, not_captured_zeros, not_captured_equivalent;
    int num_captured, c0, c1, captured_correct;
    int num_not_captured, d0, d1, default_correct, num_not_captured_equivalent;
    

    bool prediction, default_prediction;
    double lower_bound, objective, parent_lower_bound, lookahead_bound;
    double parent_equivalent_minority;
    double equivalent_minority = 0.;
    int nsamples = tree->nsamples();
    int nrules = tree->nrules();
    double c = tree->c();
    double threshold = c * nsamples;
    rule_vinit(nsamples, &captured);
    rule_vinit(nsamples, &captured_zeros);
    rule_vinit(nsamples, &not_captured);
    rule_vinit(nsamples, &not_captured_zeros);
    rule_vinit(nsamples, &not_captured_equivalent);
    int i, len_prefix;
    len_prefix = parent->depth() + 1;
    parent_lower_bound = parent->lower_bound();
    parent_equivalent_minority = parent->equivalent_minority();

    double t0 = timestamp();

    for (i = 1; i < nrules; i++) {
        //if ( (i==1) || (i==2))
            //continue;
        double t1 = timestamp();
        // check if this rule is already in the prefix
        if (std::find(parent_prefix.begin(), parent_prefix.end(), i) != parent_prefix.end())
            continue;
        // captured represents data captured by the new rule
        rule_vand(captured, parent_not_captured, tree->rule(i).truthtable, nsamples, &num_captured);

        // lower bound on antecedent support
        if ((tree->ablation() != 1) && (num_captured < threshold)) 
            continue;
        rule_vand(captured_zeros, captured, tree->label(0).truthtable, nsamples, &c0);
        c1 = num_captured - c0;
        if (c0 > c1) {
            prediction = 0;
            captured_correct = c0;
        } else {
            prediction = 1;
            captured_correct = c1;
        }

        // lower bound on accurate antecedent support
        if ((tree->ablation() != 1) && (captured_correct < threshold))
            continue;
        // subtract off parent equivalent points bound because we want to use pure lower bound from parent
        lower_bound = parent_lower_bound - parent_equivalent_minority + (double)(num_captured - captured_correct) / nsamples + c;
        logger->addToLowerBoundTime(time_diff(t1));
        logger->incLowerBoundNum();
        if (lower_bound >= tree->min_objective()) // hierarchical objective lower bound
	        continue;
        double t2 = timestamp();
        rule_vandnot(not_captured, parent_not_captured, captured, nsamples, &num_not_captured);
        rule_vand(not_captured_zeros, not_captured, tree->label(0).truthtable, nsamples, &d0);
        d1 = num_not_captured - d0;
        if (d0 > d1) {
            default_prediction = 0;
            default_correct = d0;
        } else {
            default_prediction = 1;
            default_correct = d1;
        }

    
        double misc = (double)(num_not_captured - default_correct) / nsamples;

        //double unfairness = computefairness(tree, parent_not_captured, captured);
        //double unfairness = compute_fairness_metrics(tree, parent_not_captured, captured).statistical_parity_sum;
        double unfairness = compute_fairness_metrics(tree, parent_not_captured, captured).statistical_parity_sum;
        
        
        //double unfairness = compute_fairness_metrics(tree, parent_not_captured, captured).statistical_parity_max;
        
        //double unfairness = compute_fairness_metrics(tree, parent_not_captured, captured).overall_accuracy_equality_max;

        //objective = (1 - beta)*(0.05 - misc)*(0.05 - misc) + beta*unfairness  + lower_bound;

        //printf("=====================> misc %f\n", misc);

        objective =  (1 - beta)*misc + beta*unfairness + lower_bound;

        //objective =  misc + beta*unfairness + lower_bound;




        logger->addToObjTime(time_diff(t2));
        logger->incObjNum();
        if (objective < tree->min_objective()) {
            //printf("min(objective): %1.5f -> %1.5f, length: %d, cache size: %zu\n", tree->min_objective(), objective, len_prefix, tree->num_nodes());

            logger->setTreeMinObj(objective);
            tree->update_min_objective(objective);
            tree->update_opt_rulelist(parent_prefix, i);
            tree->update_opt_predictions(parent, prediction, default_prediction);
            //print_final_rulelist_debug(tree->opt_rulelist(), tree->opt_predictions(), false, tree->getRules(), tree->getLabels());
            // dump state when min objective is updated
            logger->dumpState();
        }
        // calculate equivalent points bound to capture the fact that the minority points can never be captured correctly
        if (tree->has_minority()) {
            rule_vand(not_captured_equivalent, not_captured, tree->minority(0).truthtable, nsamples, &num_not_captured_equivalent);
            equivalent_minority = (double)(num_not_captured_equivalent) / nsamples;
            lower_bound += equivalent_minority;
        }
        if (tree->ablation() != 2)
            lookahead_bound = lower_bound + c;
        else
            lookahead_bound = lower_bound;
        // only add node to our datastructures if its children will be viable
        if (lookahead_bound < tree->min_objective()) {
            double t3 = timestamp();
            // check permutation bound
            Node* n = p->insert(i, nrules, prediction, default_prediction,
                                   lower_bound, objective, parent, num_not_captured, nsamples,
                                   len_prefix, c, equivalent_minority, tree, not_captured, parent_prefix);
            logger->addToPermMapInsertionTime(time_diff(t3));
            // n is NULL if this rule fails the permutaiton bound
            if (n) {
                double t4 = timestamp();
                tree->insert(n);
                logger->incTreeInsertionNum();
                logger->incPrefixLen(len_prefix);
                logger->addToTreeInsertionTime(time_diff(t4));
                double t5 = timestamp();
                q->push(n);
                logger->setQueueSize(q->size());
                if (tree->calculate_size())
                    logger->addQueueElement(len_prefix, lower_bound, false);
                logger->addToQueueInsertionTime(time_diff(t5));
            }
        } // else:  objective lower bound with one-step lookahead
        
    
    }
    // ------ end of for loop

    rule_vfree(&captured);
    rule_vfree(&captured_zeros);
    rule_vfree(&not_captured);
    rule_vfree(&not_captured_zeros);
    rule_vfree(&not_captured_equivalent);

    logger->addToRuleEvalTime(time_diff(t0));
    logger->incRuleEvalNum();
    logger->decPrefixLen(parent->depth());
    if (tree->calculate_size())
        logger->removeQueueElement(len_prefix - 1, parent_lower_bound, false);
    if (parent->num_children() == 0) {
        tree->prune_up(parent);
    } else {
        parent->set_done();
        tree->increment_num_evaluated();
    }
}

/*
 * Explores the search space by using a queue to order the search process.
 * The queue can be ordered by DFS, BFS, or an alternative priority metric (e.g. lower bound).
 */
int bbound(CacheTree* tree, size_t max_num_nodes, Queue* q, PermutationMap* p, double beta) {
    bool print_queue = 0;
    size_t num_iter = 0;
    int cnt;
    double min_objective;
    VECTOR captured, not_captured;
    rule_vinit(tree->nsamples(), &captured);
    rule_vinit(tree->nsamples(), &not_captured);

    size_t queue_min_length = logger->getQueueMinLen();

    double start = timestamp();
    logger->setInitialTime(start);
    logger->initializeState(tree->calculate_size());
    int verbosity = logger->getVerbosity();
    // initial log record
    logger->dumpState();         

    min_objective = 1.0;
    tree->insert_root();
    logger->incTreeInsertionNum();
    q->push(tree->root());
    logger->setQueueSize(q->size());
    logger->incPrefixLen(0);
    // log record for empty rule list
    logger->dumpState();
    while ((tree->num_nodes() < max_num_nodes) && !q->empty()) {
        double t0 = timestamp();
        std::pair<Node*, tracking_vector<unsigned short, DataStruct::Tree> > node_ordered = q->select(tree, captured);
                

        logger->addToNodeSelectTime(time_diff(t0));
        logger->incNodeSelectNum();
        if (node_ordered.first) {
            double t1 = timestamp();
            // not_captured = default rule truthtable & ~ captured

            rule_vandnot(not_captured, tree->rule(0).truthtable, captured, tree->nsamples(), &cnt);

            evaluate_children(tree, node_ordered.first, node_ordered.second, not_captured, q, p, beta);
    
    

            logger->addToEvalChildrenTime(time_diff(t1));
            logger->incEvalChildrenNum();

            if (tree->min_objective() < min_objective) {
                min_objective = tree->min_objective();
                if (verbosity >= 10)
                    printf("before garbage_collect. num_nodes: %zu, log10(remaining): %zu\n", 
                            tree->num_nodes(), logger->getLogRemainingSpaceSize());
                logger->dumpState();
                tree->garbage_collect();
                logger->dumpState();
                if (verbosity >= 10)
                    printf("after garbage_collect. num_nodes: %zu, log10(remaining): %zu\n", tree->num_nodes(), logger->getLogRemainingSpaceSize());
            }
        }
        logger->setQueueSize(q->size());
        if (queue_min_length < logger->getQueueMinLen()) {
            // garbage collect the permutation map: can be simplified for the case of BFS
            queue_min_length = logger->getQueueMinLen();
            //pmap_garbage_collect(p, queue_min_length);
        }
        ++num_iter;
        if ((num_iter % 10000) == 0) {
            if (verbosity >= 10)
                printf("iter: %zu, tree: %zu, queue: %zu, pmap: %zu, log10(remaining): %zu, time elapsed: %f\n",
                       num_iter, tree->num_nodes(), q->size(), p->size(), logger->getLogRemainingSpaceSize(), time_diff(start));
        }
        if ((num_iter % logger->getFrequency()) == 0) {
            // want ~1000 records for detailed figures
            logger->dumpState();
        }
    }
    logger->dumpState(); // second last log record (before queue elements deleted)
    if (verbosity >= 1)
        printf("iter: %zu, tree: %zu, queue: %zu, pmap: %zu, log10(remaining): %zu, time elapsed: %f\n",
               num_iter, tree->num_nodes(), q->size(), p->size(), logger->getLogRemainingSpaceSize(), time_diff(start));
    if (q->empty())
        printf("Exited because queue empty\n");
    else
        printf("Exited because max number of nodes in the tree was reached\n");

    size_t tree_mem = logger->getTreeMemory(); 
    size_t pmap_mem = logger->getPmapMemory(); 
    size_t queue_mem = logger->getQueueMemory(); 
    printf("TREE mem usage: %zu\n", tree_mem);
    printf("PMAP mem usage: %zu\n", pmap_mem);
    printf("QUEUE mem usage: %zu\n", queue_mem);

    // Print out queue
    ofstream f;
    if (print_queue) {
        char fname[] = "queue.txt";
        printf("Writing queue elements to: %s\n", fname);
        f.open(fname, ios::out | ios::trunc);
        f << "lower_bound objective length frac_captured rule_list\n";
    }

    // Clean up data structures
    printf("Deleting queue elements and corresponding nodes in the cache,"
            "since they may not be reachable by the tree's destructor\n");
    printf("\nminimum objective: %1.10f\n", tree->min_objective());
    Node* node;
    double min_lower_bound = 1.0;
    double lb;
    size_t num = 0;
    while (!q->empty()) {
        node = q->front();
        q->pop();
        if (node->deleted()) {
            tree->decrement_num_nodes();
            logger->removeFromMemory(sizeof(*node), DataStruct::Tree);
            delete node;
        } else {
            lb = node->lower_bound() + tree->c();
            if (lb < min_lower_bound)
                min_lower_bound = lb;
            if (print_queue) {
                std::pair<tracking_vector<unsigned short, DataStruct::Tree>, tracking_vector<bool, DataStruct::Tree> > pp_pair = node->get_prefix_and_predictions();
                tracking_vector<unsigned short, DataStruct::Tree> prefix = std::move(pp_pair.first);
                tracking_vector<bool, DataStruct::Tree> predictions = std::move(pp_pair.second);
                f << node->lower_bound() << " " << node->objective() << " " << node->depth() << " "
                  << (double) node->num_captured() / (double) tree->nsamples() << " ";
                for(size_t i = 0; i < prefix.size(); ++i) {
                    f << tree->rule_features(prefix[i]) << "~"
                      << predictions[i] << ";";
                }
                f << "default~" << predictions.back() << "\n";
                num++;
            }
        }
    }
    printf("minimum lower bound in queue: %1.10f\n\n", min_lower_bound);
    if (print_queue)
        f.close();
    // last log record (before cache deleted)
    logger->dumpState();

    rule_vfree(&captured);
    rule_vfree(&not_captured);
    return num_iter;
}
