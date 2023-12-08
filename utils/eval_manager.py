import pytrec_eval
import yaml
from utils.logging import setup_logging


class EvalManager():

    def __init__(self,config):
        with open(config, "r") as config_file:
            self.config = yaml.safe_load(config_file)
        self.logger = setup_logging(self.__class__._name__, self.config)