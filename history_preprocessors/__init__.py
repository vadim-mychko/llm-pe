from history_preprocessors.history_concatenator import HistoryConcatenator
from history_preprocessors.aspect_preprocessor import AspectPreprocessor
#from item_scorers.entailment_scorer import EntailmentScorer
#from item_scorers.ce_scorer import CEScorer
#from item_scorers.dense_retrieva_scorer import DenseRetrievalScorer

HISTORY_PREPROCESSOR_CLASSES = {
    'HistoryConcatenator': HistoryConcatenator,
    'AspectPreprocessor': AspectPreprocessor,
#    'EntailmentScorer': EntailmentScorer,
#    'CEScorer': CEScorer,
#    'DenseRetrievalScorer': DenseRetrievalScorer,
}
