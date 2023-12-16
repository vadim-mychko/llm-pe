"""
Base Class for LLM Wrappers.
"""
import abc
import logging
from utils.logging import setup_logging


class LLMBase(abc.ABC):
    def __init__(self, config):
        self.config = config
        self.total_tokens_used = 0
        self.total_cost = 0
        self.log_probabilities = []
        self.logger = setup_logging(self.__class__.__name__, self.config)

    @abc.abstractmethod
    def make_request(self, prompt: str, temperature=0.0) -> str:
        """
        Make a request to the LLM.

        Args:
            prompt: An input to the LLM.
            temperature: The temperature to use for the LLM.
            max_tokens: The maximum number of tokens to generate.

        Returns:
            str: The generated text response.
        """
        raise NotImplementedError("This method must be implemented by a subclass.")

