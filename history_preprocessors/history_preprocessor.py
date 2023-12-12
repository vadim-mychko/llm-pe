
from abc import ABC, abstractmethod
from utils.logging import setup_logging

class HistoryPreprocessor(ABC):
    
    def __init__(self, config):
        
        self.config = config
        self.logger = setup_logging(self.__class__.__name__, self.config)

    @abstractmethod
    def preprocess(self,history) -> str:
        pass

