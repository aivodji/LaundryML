#include "queue.hh"
#include <algorithm>
#include <iostream>
#include <sys/resource.h>
#include <stdio.h>

// todo check division
confusion_matrix_groups computeModelFairness(int nsamples,
                                            const tracking_vector<unsigned short, DataStruct::Tree>& rulelist,
                                            const tracking_vector<bool, DataStruct::Tree>& preds,                  
                                            rule_t * rules,
                                            rule_t * labels,
                                            int maj_pos,
                                            int min_pos){
    // datastructures to store the results
    confusion_matrix_groups cmg;
    confusion_matrix cm_minority;
    confusion_matrix cm_majority;

    VECTOR captured_it;
    VECTOR not_captured_yet;
    VECTOR captured_zeros;
    VECTOR preds_prefix;
    int nb;
    int nb2;
    int pm;
    rule_vinit(nsamples, &captured_it);
    rule_vinit(nsamples, &not_captured_yet);
    rule_vinit(nsamples, &preds_prefix);
    rule_vinit(nsamples, &captured_zeros);
    // Initially not_captured_yet is full of ones
    rule_vor(not_captured_yet,labels[0].truthtable, labels[1].truthtable, nsamples ,&nb);
    // Initially preds_prefix is full of zeros
    rule_vand(preds_prefix, labels[0].truthtable, labels[1].truthtable, nsamples, &nb);

    tracking_vector<unsigned short, DataStruct::Tree>::iterator it;
    for (size_t i = 0; i < rulelist.size(); ++i) {
        rule_vand(captured_it, not_captured_yet, rules[rulelist[i]].truthtable, nsamples, &nb);
        rule_vandnot(not_captured_yet, not_captured_yet, captured_it, nsamples, &pm);
        rule_vand(captured_zeros, captured_it, labels[0].truthtable, nsamples, &nb2);
        if(nb2 <= (nb - nb2)) { //then prediction is 1
            rule_vor(preds_prefix, preds_prefix, captured_it, nsamples, &nb);
        }
    }
    if(preds.back() == 1) { // else it is already OK
        rule_vor(preds_prefix, preds_prefix, not_captured_yet, nsamples, &pm);
    }

    //std::cout << "============= " << mpz_popcount(preds_prefix) << std::endl;

    // true positives, false negatives, true negatives, and false positives
    VECTOR TP, FP, FN, TN;
    rule_vinit(nsamples, &TP);
    rule_vinit(nsamples, &FP);
    rule_vinit(nsamples, &FN);
    rule_vinit(nsamples, &TN);

    rule_vand(TP, preds_prefix, labels[1].truthtable, nsamples, &pm);
    rule_vand(FP, preds_prefix, labels[0].truthtable, nsamples, &pm);
    rule_vandnot(FN, labels[1].truthtable, preds_prefix, nsamples, &pm);
    rule_vandnot(TN, labels[0].truthtable, preds_prefix, nsamples, &pm);

    // true positives, false negatives, true negatives, and false positives for majority group
    VECTOR TP_maj, FP_maj, FN_maj, TN_maj;
    rule_vinit(nsamples, &TP_maj);
    rule_vinit(nsamples, &FP_maj);
    rule_vinit(nsamples, &FN_maj);
    rule_vinit(nsamples, &TN_maj);

    mpz_and(TP_maj, TP, rules[maj_pos].truthtable);
    mpz_and(FP_maj, FP, rules[maj_pos].truthtable);
    mpz_and(FN_maj, FN, rules[maj_pos].truthtable);
    mpz_and(TN_maj, TN, rules[maj_pos].truthtable);

    // true positives, false negatives, true negatives, and false positives for minority group
    VECTOR TP_min, FP_min, FN_min, TN_min;
    rule_vinit(nsamples, &TP_min);
    rule_vinit(nsamples, &FP_min);
    rule_vinit(nsamples, &FN_min);
    rule_vinit(nsamples, &TN_min);

    mpz_and(TP_min, TP, rules[min_pos].truthtable);
    mpz_and(FP_min, FP, rules[min_pos].truthtable);
    mpz_and(FN_min, FN, rules[min_pos].truthtable);
    mpz_and(TN_min, TN, rules[min_pos].truthtable);

    // stats for majority
    int nTP_maj = mpz_popcount(TP_maj);
    int nFP_maj = mpz_popcount(FP_maj);

    int nFN_maj = mpz_popcount(FN_maj);
    int nTN_maj = mpz_popcount(TN_maj);

    double nPPV_maj = (double) nTP_maj / max((nTP_maj + nFP_maj), 1);
    double nTPR_maj = (double) nTP_maj / max((nTP_maj + nFN_maj), 1);

    double nFDR_maj = (double) nFP_maj / max((nFP_maj + nTP_maj), 1);
    double nFPR_maj = (double) nFP_maj / max((nFP_maj + nTN_maj), 1);

    double nFOR_maj = (double) nFN_maj / max((nFN_maj + nTN_maj), 1);
    double nFNR_maj = (double) nFN_maj / max((nFN_maj + nTP_maj), 1);

    double nNPV_maj = (double) nTN_maj / max((nTN_maj + nFN_maj), 1);
    double nTNR_maj = (double) nTN_maj / max((nTN_maj + nFP_maj), 1);


    cm_majority.nTP = nTP_maj;
    cm_majority.nFP = nFP_maj;
    cm_majority.nFN = nFN_maj;
    cm_majority.nTN = nTN_maj;
    cm_majority.nPPV = nPPV_maj;
    cm_majority.nTPR = nTPR_maj;
    cm_majority.nFDR = nFDR_maj;
    cm_majority.nFPR = nFPR_maj;
    cm_majority.nFOR = nFOR_maj;
    cm_majority.nFNR = nFNR_maj;
    cm_majority.nNPV = nNPV_maj;
    cm_majority.nTNR = nTNR_maj;


    // stats for minority
    int nTP_min = mpz_popcount(TP_min);
    int nFP_min = mpz_popcount(FP_min);
    int nFN_min = mpz_popcount(FN_min);
    int nTN_min = mpz_popcount(TN_min);

    double nPPV_min = (double) nTP_min / max((nTP_min + nFP_min), 1);
    double nTPR_min = (double) nTP_min / max((nTP_min + nFN_min), 1);
    double nFDR_min = (double) nFP_min / max((nFP_min + nTP_min), 1);
    double nFPR_min = (double) nFP_min / max((nFP_min + nTN_min), 1);
    double nFOR_min = (double) nFN_min / max((nFN_min + nTN_min), 1);
    double nFNR_min = (double) nFN_min / max((nFN_min + nTP_min), 1);
    double nNPV_min = (double) nTN_min / max((nTN_min + nFN_min), 1);
    double nTNR_min = (double) nTN_min / max((nTN_min + nFP_min), 1);



    cm_minority.nTP = nTP_min;
    cm_minority.nFP = nFP_min;
    cm_minority.nFN = nFN_min;
    cm_minority.nTN = nTN_min;
    cm_minority.nPPV = nPPV_min;
    cm_minority.nTPR = nTPR_min;
    cm_minority.nFDR = nFDR_min;
    cm_minority.nFPR = nFPR_min;
    cm_minority.nFOR = nFOR_min;
    cm_minority.nFNR = nFNR_min;
    cm_minority.nNPV = nNPV_min;
    cm_minority.nTNR = nTNR_min;

    cmg.majority = cm_majority;
    cmg.minority = cm_minority;


    rule_vfree(&captured_it);
    rule_vfree(&not_captured_yet);
    rule_vfree(&captured_zeros);
    rule_vfree(&preds_prefix);
    rule_vfree(&TP);
    rule_vfree(&FP);
    rule_vfree(&FN);
    rule_vfree(&TN);
    rule_vfree(&TP_maj);
    rule_vfree(&FP_maj);
    rule_vfree(&FN_maj);
    rule_vfree(&TN_maj);
    rule_vfree(&TP_min);
    rule_vfree(&FP_min);
    rule_vfree(&FN_min);
    rule_vfree(&TN_min);


    return cmg;

}

confusion_matrix_groups compute_confusion_matrix(tracking_vector<unsigned short, 
                                                DataStruct::Tree> parent_prefix, 
                                                CacheTree* tree, 
                                                VECTOR parent_not_captured, 
                                                VECTOR captured,
                                                int index, 
                                                int maj_pos,
                                                int min_pos,
                                                int prediction, 
                                                int default_prediction){

    // datastructures to store the results
    confusion_matrix_groups cmg;
    confusion_matrix cm_minority;
    confusion_matrix cm_majority;



    VECTOR not_captured;
    int num_not_captured;
    rule_vinit(tree->nsamples(), &not_captured);
    rule_vandnot(not_captured, parent_not_captured, captured, tree->nsamples(), &num_not_captured);

    VECTOR captured_it;
    VECTOR not_captured_yet;
    VECTOR captured_zeros;
    VECTOR preds_prefix;
    int nb;
    int nb2;
    int pm;
    rule_vinit(tree->nsamples(), &captured_it);
    rule_vinit(tree->nsamples(), &not_captured_yet);
    rule_vinit(tree->nsamples(), &preds_prefix);
    rule_vinit(tree->nsamples(), &captured_zeros);

    // Initially not_captured_yet is full of ones
    rule_vor(not_captured_yet, tree->label(0).truthtable, tree->label(1).truthtable, tree->nsamples(),&nb);
    // Initially preds_prefix is full of zeros
    rule_vand(preds_prefix, tree->label(0).truthtable, tree->label(1).truthtable, tree->nsamples(),&nb);
    tracking_vector<unsigned short, DataStruct::Tree>::iterator it;

    for (it = parent_prefix.begin(); it != parent_prefix.end(); it++) {
        //printf("precedent rules : %s\n", tree->rule(*it).features);
        rule_vand(captured_it, not_captured_yet, tree->rule(*it).truthtable, tree->nsamples(), &nb);
        rule_vandnot(not_captured_yet, not_captured_yet, captured_it, tree->nsamples(), &pm);
        rule_vand(captured_zeros, captured_it, tree->label(0).truthtable, tree->nsamples(), &nb2);
        if(nb2 <= (nb - nb2)) { //then prediction is 1
            rule_vor(preds_prefix, preds_prefix, captured_it, tree->nsamples(), &nb);
        }
    }

    rule_vandnot(not_captured_yet, not_captured_yet, captured, tree->nsamples(), &pm);

    if(default_prediction == 1) { // else it is already OK
        rule_vor(preds_prefix, preds_prefix, not_captured, tree->nsamples(), &pm);
    }

    if(prediction == 1) { // else it is already OK
        rule_vor(preds_prefix, preds_prefix, captured, tree->nsamples(), &pm);
    }

    // true positives, false negatives, true negatives, and false positives
    VECTOR TP, FP, FN, TN;
    rule_vinit(tree->nsamples(), &TP);
    rule_vinit(tree->nsamples(), &FP);
    rule_vinit(tree->nsamples(), &FN);
    rule_vinit(tree->nsamples(), &TN);

    rule_vand(TP, preds_prefix, tree->label(1).truthtable, tree->nsamples(), &pm);
    rule_vand(FP, preds_prefix, tree->label(0).truthtable, tree->nsamples(), &pm);
    rule_vandnot(FN, tree->label(1).truthtable, preds_prefix, tree->nsamples(), &pm);
    rule_vandnot(TN, tree->label(0).truthtable, preds_prefix, tree->nsamples(), &pm);

    // true positives, false negatives, true negatives, and false positives for majority group
    VECTOR TP_maj, FP_maj, FN_maj, TN_maj;
    rule_vinit(tree->nsamples(), &TP_maj);
    rule_vinit(tree->nsamples(), &FP_maj);
    rule_vinit(tree->nsamples(), &FN_maj);
    rule_vinit(tree->nsamples(), &TN_maj);

    mpz_and(TP_maj, TP, tree->rule(maj_pos).truthtable);
    mpz_and(FP_maj, FP, tree->rule(maj_pos).truthtable);
    mpz_and(FN_maj, FN, tree->rule(maj_pos).truthtable);
    mpz_and(TN_maj, TN, tree->rule(maj_pos).truthtable);

    // true positives, false negatives, true negatives, and false positives for minority group
    VECTOR TP_min, FP_min, FN_min, TN_min;
    rule_vinit(tree->nsamples(), &TP_min);
    rule_vinit(tree->nsamples(), &FP_min);
    rule_vinit(tree->nsamples(), &FN_min);
    rule_vinit(tree->nsamples(), &TN_min);

    mpz_and(TP_min, TP, tree->rule(min_pos).truthtable);
    mpz_and(FP_min, FP, tree->rule(min_pos).truthtable);
    mpz_and(FN_min, FN, tree->rule(min_pos).truthtable);
    mpz_and(TN_min, TN, tree->rule(min_pos).truthtable);

    // stats for majority
    int nTP_maj = mpz_popcount(TP_maj);
    int nFP_maj = mpz_popcount(FP_maj);

    int nFN_maj = mpz_popcount(FN_maj);
    int nTN_maj = mpz_popcount(TN_maj);

    double nPPV_maj = (double) nTP_maj / max((nTP_maj + nFP_maj), 1);
    double nTPR_maj = (double) nTP_maj / max((nTP_maj + nFN_maj), 1);

    double nFDR_maj = (double) nFP_maj / max((nFP_maj + nTP_maj), 1);
    double nFPR_maj = (double) nFP_maj / max((nFP_maj + nTN_maj), 1);

    double nFOR_maj = (double) nFN_maj / max((nFN_maj + nTN_maj), 1);
    double nFNR_maj = (double) nFN_maj / max((nFN_maj + nTP_maj), 1);

    double nNPV_maj = (double) nTN_maj / max((nTN_maj + nFN_maj), 1);
    double nTNR_maj = (double) nTN_maj / max((nTN_maj + nFP_maj), 1);

    cm_majority.nTP = nTP_maj;
    cm_majority.nFP = nFP_maj;
    cm_majority.nFN = nFN_maj;
    cm_majority.nTN = nTN_maj;

    cm_majority.nPPV = nPPV_maj;
    cm_majority.nTPR = nTPR_maj;
    cm_majority.nFDR = nFDR_maj;
    cm_majority.nFPR = nFPR_maj;
    cm_majority.nFOR = nFOR_maj;
    cm_majority.nFNR = nFNR_maj;
    cm_majority.nNPV = nNPV_maj;
    cm_majority.nTNR = nTNR_maj;


    // stats for minority
    int nTP_min = mpz_popcount(TP_min);
    int nFP_min = mpz_popcount(FP_min);
    int nFN_min = mpz_popcount(FN_min);
    int nTN_min = mpz_popcount(TN_min);

    double nPPV_min = (double) nTP_min / max((nTP_min + nFP_min), 1);
    double nTPR_min = (double) nTP_min / max((nTP_min + nFN_min), 1);
    double nFDR_min = (double) nFP_min / max((nFP_min + nTP_min), 1);
    double nFPR_min = (double) nFP_min / max((nFP_min + nTN_min), 1);
    double nFOR_min = (double) nFN_min / max((nFN_min + nTN_min), 1);
    double nFNR_min = (double) nFN_min / max((nFN_min + nTP_min), 1);
    double nNPV_min = (double) nTN_min / max((nTN_min + nFN_min), 1);
    double nTNR_min = (double) nTN_min / max((nTN_min + nFP_min), 1);

    cm_minority.nTP = nTP_min;
    cm_minority.nFP = nFP_min;
    cm_minority.nFN = nFN_min;
    cm_minority.nTN = nTN_min;

    cm_minority.nPPV = nPPV_min;
    cm_minority.nTPR = nTPR_min;
    cm_minority.nFDR = nFDR_min;
    cm_minority.nFPR = nFPR_min;
    cm_minority.nFOR = nFOR_min;
    cm_minority.nFNR = nFNR_min;
    cm_minority.nNPV = nNPV_min;
    cm_minority.nTNR = nTNR_min;

    cmg.majority = cm_majority;
    cmg.minority = cm_minority;


    rule_vfree(&not_captured);
    rule_vfree(&captured_it);
    rule_vfree(&not_captured_yet);
    rule_vfree(&captured_zeros);
    rule_vfree(&preds_prefix);
    rule_vfree(&TP);
    rule_vfree(&FP);
    rule_vfree(&FN);
    rule_vfree(&TN);
    rule_vfree(&TP_maj);
    rule_vfree(&FP_maj);
    rule_vfree(&FN_maj);
    rule_vfree(&TN_maj);
    rule_vfree(&TP_min);
    rule_vfree(&FP_min);
    rule_vfree(&FN_min);
    rule_vfree(&TN_min);

    return cmg;
}


fairness_metrics compute_fairness_metrics(confusion_matrix_groups cmg){
    fairness_metrics metrics;

    // statistical_parity
    double statistical_parity_maj = (double) (cmg.majority.nTP + cmg.majority.nFP) / 
                                                max((cmg.majority.nTP + cmg.majority.nFP + cmg.majority.nFN + cmg.majority.nTN),1);
                                
    double statistical_parity_min = (double) (cmg.minority.nTP + cmg.minority.nFP) / 
                                                max((cmg.minority.nTP + cmg.minority.nFP + cmg.minority.nFN + cmg.minority.nTN),1);
                                
                                
    metrics.statistical_parity =  fabs(statistical_parity_maj - statistical_parity_min);

    // predictive parity
    metrics.predictive_parity = fabs(cmg.majority.nPPV - cmg.minority.nPPV);

    // predictive equality
    metrics.predictive_equality = fabs(cmg.majority.nFPR - cmg.minority.nFPR);

    // equal opportunity
    metrics.equal_opportunity = fabs(cmg.majority.nFNR - cmg.minority.nFNR);

    return metrics;
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
void evaluate_children(CacheTree* tree,
                        Node* parent, 
                        tracking_vector<unsigned short, 
                        DataStruct::Tree> parent_prefix,
                        VECTOR parent_not_captured, 
                        Queue* q, 
                        PermutationMap* p, 
                        double beta,
                        int fairness,
                        int maj_pos,
                        int min_pos) {

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
        double unfairness = 0.0;

        confusion_matrix_groups cmg = compute_confusion_matrix(parent_prefix, tree, parent_not_captured, captured, i,
                                                                                 maj_pos, min_pos, prediction, default_prediction);

        fairness_metrics fm = compute_fairness_metrics(cmg);
        
        switch (fairness)
        {
            case 1:
                unfairness = fm.statistical_parity;
                break;
            case 2:
                unfairness = fm.predictive_parity;
                break;
            case 3:
                unfairness = fm.predictive_equality;
                break;
            case 4:
                unfairness = fm.equal_opportunity;
                break;
            default:
                break;
        }
        
        // compute the objective function
        objective =  (1 - beta)*misc + beta*unfairness + lower_bound;

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
int bbound(CacheTree* tree,
                size_t max_num_nodes,
                Queue* q, 
                PermutationMap* p, 
                double beta,
                int fairness,
                int maj_pos,
                int min_pos) {

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
            
            // call of evaluate children
            evaluate_children(tree, node_ordered.first, node_ordered.second, not_captured, q, p, beta, fairness, maj_pos, min_pos);
    
    
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
