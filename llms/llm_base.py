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
    def make_request(self, prompt: str, temperature=0.0, max_tokens=256) -> str:
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


#[Anton Jun 27]: TODO: not all openai subclasses are able to return log probabilities. Instead of having a method in the parent class, 
#create an interface for this method, and update the UML. Also - I'm not sure a list is the best return type - 
#perhaps it is best to define the return type later once we know how the Completions log probs will be used.
#Example interface:
'''
class LogProbabilityProvider(abc.ABC):
    @abc.abstractmethod
    def get_log_probabilities(self) -> list:
        """
        Get the log probabilities of the most recent response.

        Returns:
            list: The log probabilities.
        """
'''
'''
    def get_log_probabilities(self) -> list:
        """
        Get the log probabilities of the most recent response.

        Returns:
            list: The log probabilities.
        """
        return self.log_probabilities
'''