"""
Base Class for LLM Wrappers.
"""
import abc
import logging
import os
import openai
from utils.logging import setup_logging


class LLMBase(abc.ABC):
    def __init__(self, config):
        self.config = config
        self.total_tokens_used = 0
        self.total_cost = 0
        self.log_probabilities = []
        self.API_KEY = os.environ['OPENAI_API_KEY']
        openai.api_key = self.API_KEY

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

    def get_total_tokens_used(self) -> int:
        """
        Get the total number of tokens used by the LLM.

        Returns:
            int: The total number of tokens used.
        """
        return self.total_tokens_used
    
    def get_total_cost(self) -> float:
        """
        Get the total cost of LLM requests.

        Returns:
            float: The total cost of LLM requests.
        """
        return self.total_cost

