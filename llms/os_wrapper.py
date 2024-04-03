import os
from typing import Optional
import os
import openai
import time
import sys
from llms.llm_base import LLMBase
from transformers import AutoTokenizer, AutoModelForCausalLM

class OSWrapper(LLMBase):
    def __init__(self, config):
        super().__init__(config)

        self.tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
        self.model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1")
        

    def make_request(self, prompt: str, temperature: Optional[float] = 0, logprobs=0) -> str:
        """
        Make a request to Open AI's GPT Completion LLM.

        Args:
            prompt: An input to the LLM.
            temperature: The temperature to use for the LLM.

        Returns:
            str: The generated text response.
        """

        inputs = self.tokenizer(prompt, return_tensors="pt")
        generate_ids = self.model.generate(inputs.input_ids, max_length=30) # Temporary restriction
        response = self.tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

        return response
