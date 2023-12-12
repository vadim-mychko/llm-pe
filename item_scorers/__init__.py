
from item_scorers.llm_logprob_scorer import LLMLogprobScorer
#from item_scorers.entailment_scorer import EntailmentScorer
#from item_scorers.ce_scorer import CEScorer
#from item_scorers.dense_retrieva_scorer import DenseRetrievalScorer

ITEM_SCORER_CLASSES = {
    'LLMLogprobScorer': LLMLogprobScorer,
#    'EntailmentScorer': EntailmentScorer,
#    'CEScorer': CEScorer,
#    'DenseRetrievalScorer': DenseRetrievalScorer,
}
