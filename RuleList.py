import functools 
@functools.total_ordering
class RuleList(object):
    def __init__(self, unfairness=0.0, fidelity=1.0, quality=1.0, beta="reg=0.0"):
        self.unfairness = unfairness
        self.fidelity = fidelity
        self.beta = beta
        self.quality = 1.0

    def __str__(self):
        output = 'unfairnes: %20s  fidelity:%20s  beta:%20s' % (self.unfairness, self.fidelity, self.beta)
        #return  "unfairnes: " + str(self.unfairness) + " fidelity: " + str(self.fidelity) + " beta: " + self.beta
        return output

    def __eq__ (self, other):
        return self.unfairness == other.unfairness and self.fidelity == other.fidelity 
    
    def __lt__ (self, other):
        if self.fidelity == other.fidelity:
            return self.unfairness < other.unfairness
        return self.fidelity < other.fidelity

    def distance_to_target(self, target):
        dist = (self.unfairness - target.unfairness)**2 + (self.fidelity - target.fidelity)**2
        self.quality = dist
        return dist



@functools.total_ordering
class RuleList2(object):
    def __init__(self, unfairness_train=0.0, fidelity_train=1.0, quality_train=1.0, unfairness_test=0.0, fidelity_test=1.0, quality_test=1.0, beta="reg=0.0"):
        self.unfairness_train = unfairness_train
        self.fidelity_train = fidelity_train
        self.quality_train = 1.0

        self.unfairness_test = unfairness_test
        self.fidelity_test = fidelity_test
        self.quality_test = 1.0

        self.beta = beta
        

    def __str__(self):
        output = 'fidelity_train: %20s  fidelity_test:%20s' % (self.fidelity_train, self.fidelity_test)
        output2 = ' unfairnes_train: %20s  unfairness_test:%20s  params:%20s' % (self.unfairness_train, self.unfairness_test, self.beta)
        #return  "unfairnes: " + str(self.unfairness) + " fidelity: " + str(self.fidelity) + " beta: " + self.beta
        return output + output2

    def __eq__ (self, other):
        return self.unfairness_train == other.unfairness_train and self.fidelity_train == other.fidelity_train
    
    def __lt__ (self, other):
        if self.fidelity_train == other.fidelity_train:
            return self.unfairness_train < other.unfairness_train
        return self.fidelity_train < other.fidelity_train

    def distance_to_target(self, target):
        dist = (self.unfairness_train - target.unfairness_train)**2 + (self.fidelity_train - target.fidelity_train)**2
        dist2 = (self.unfairness_test - target.unfairness_test)**2 + (self.fidelity_test - target.fidelity_test)**2
        self.quality_train = dist
        self.quality_test = dist2
        return dist, dist2