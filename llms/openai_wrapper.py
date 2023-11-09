import os
from typing import Optional

import openai

from llms.llm_base import LLMBase

class GPTCompletion(LLMBase):
    def __init__(self, config):
        super().__init__(config)

    def make_request(self, prompt: str, temperature: Optional[float] = 0, max_tokens=256, logprobs=0) -> str:
        """
        Make a request to Open AI's GPT Completion LLM.

        Args:
            prompt: An input to the LLM.
            temperature: The temperature to use for the LLM.
            max_tokens: The maximum number of tokens to generate.

        Returns:
            str: The generated text response.
        """
        response = openai.Completion.create(
            model=self.config["llm"]["model"],
            temperature=temperature,
            prompt=prompt,
            max_tokens=max_tokens,
            logprobs=logprobs
        )
    

        tokens_used = response["usage"]["total_tokens"]
        cost_of_response = tokens_used * 0.000002
        logprobs = response['choices'][0]['logprobs']['top_logprobs'] #TODO: Since we want more than just top, change this

        self.total_tokens_used += tokens_used
        self.total_cost += cost_of_response
        self.log_probabilities = logprobs
        return response['choices'][0]['text']
    
    def get_log_probabilities(self):
        """
        Get the log probabilities of the most recent response.

        Returns:
            list: The log probabilities.
        """
        return self.log_probabilities
    

class GPTChatCompletion(LLMBase):
    def __init__(self, config):
        super().__init__(config)
        
    def make_request(self, prompt: str, temperature: Optional[float] = 0, max_tokens=2000) -> str:
        """
        Make a request to Open AI's GPT Chat Completion LLM.

        Args:
            prompt: An input to the LLM.
            temperature: The temperature to use for the LLM.
            max_tokens: The maximum number of tokens to generate.

        Returns:
            str: The generated text response.
        """
        model_name = self.config["llm"]["model"]

        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ]

        response = openai.ChatCompletion.create(
            model=model_name,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )

        tokens_used = response["usage"]["total_tokens"]
        cost_of_response = tokens_used * 0.000002

        self.total_tokens_used += tokens_used
        self.total_cost += cost_of_response

        return response.choices[0].message['content']