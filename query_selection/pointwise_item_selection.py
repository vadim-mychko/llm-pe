'''
pointwise_item_selection.py contains different pointwise item selection methods. Each 
method is passed a list of beliefs about item utility in the form of beta-bernoulli 
parameters. Each is expected to return the index of the item on which to query, based 
on its selection method.
'''
import math

# Select the item with the highest expected utility.
def greedy(beliefs):
    scores = [(util['alpha'] / (util['alpha'] + util['beta']) ) for util in beliefs]
        
    top_idx = sorted(range(len(scores)), key=lambda i: scores[i])[-1:]

    return top_idx

def random(beliefs):
    raise NotImplementedError

def entropy_reduction(beliefs):
    raise NotImplementedError

def ucb(beliefs):
    temp = 1.96 #TODO: Correctly implement temperature
    expected_values = [(util['alpha'] / (util['alpha'] + util['beta']) ) for util in beliefs]
    var = [(
            (util['alpha'] * util['beta'])
              / (
                  math.pow((util['alpha'] + util['beta']), 2) * 
                  (util['alpha'] + util['beta'] + 1)
                )
            )
            for util in beliefs]
    
    top_idx = sorted(range(len(beliefs)), key=lambda i: expected_values[i] + temp*math.sqrt(var[i]))[-1:]

    return top_idx

def thompson_sampling(beliefs):
    raise NotImplementedError

ITEM_SELECTION_CLASSES = {
    'greedy': greedy,
    'entropy_reduction': entropy_reduction,
    'ucb': ucb,
}

# TODO: Fill out this file