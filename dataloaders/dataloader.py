from abc import ABC, abstractmethod
from utils.logging import setup_logging

class DataLoader(ABC):
    
    def __init__(self, config):
        
        self.config = config
        self.logger = setup_logging(self.__class__.__name__, self.config)

    def __len__(self):
        pass

    @abstractmethod
    def get_data(self) -> list:
        """
        Reads query data from a file path in the config and retrieves a list of queries ordered alphabetically.
        """
        pass
