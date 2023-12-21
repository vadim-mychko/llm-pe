import os
from typing import Optional
import os
import openai
from llms.llm_base import LLMBase

class GPTCompletion(LLMBase):
    def __init__(self, config):
        super().__init__(config)


        self.API_KEY = os.environ['OPENAI_API_KEY']
        openai.api_key = self.API_KEY

    def make_request(self, prompt: str, temperature: Optional[float] = 0, logprobs=0) -> str:
        """
        Make a request to Open AI's GPT Completion LLM.

        Args:
            prompt: An input to the LLM.
            temperature: The temperature to use for the LLM.

        Returns:
            str: The generated text response.
        """
        response = openai.Completion.create(
            model=self.config["llm"]["model"],
            temperature=temperature,
            prompt=prompt,
            logprobs=logprobs
        )


        if logprobs:
        #logprobs for only the first generated token and alternatives (e.g, 'True', 'False', 'Unc').
        #format: {token: logprob, ... }
            self.logprobs = response['choices'][0]['logprobs']['top_logprobs'][0]
        else:
            self.logprobs = None


        return response['choices'][0]['text']
   
    

class GPTChatCompletion(LLMBase):
    def __init__(self, config):
        super().__init__(config)
        
    def make_request(self, prompt: str, temperature: float = 0.0, logprobs=0, top_logprobs = 3) -> str:
        """
        Make a request to Open AI's GPT Chat Completion LLM.

        Args:
            prompt: An input to the LLM.
            temperature: The temperature to use for the LLM.

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
            temperature=temperature,
            logprobs = bool(logprobs),
            top_logprobs = top_logprobs
        )


        if bool(logprobs):
        #logprobs for only the first generated token and alternatives (e.g, 'True', 'False', 'Unc').
        #format: {token: logprob, ... }
            top_logprobs = response.choices[0].logprobs['content'][0]['top_logprobs']
            self.logprobs = {e['token']: e['logprob'] for e in top_logprobs}
        else:
            self.logprobs = None
        print(self.logprobs)
        return response.choices[0].message['content']
