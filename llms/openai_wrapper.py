import os
from typing import Optional
import os
import openai
import time
import sys
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

        attempts = 0
        while attempts < 3:
            try:
                response = openai.Completion.create(
                    model=self.config["llm"]["model"],
                    temperature=temperature,
                    prompt=prompt,
                    logprobs=logprobs
                )
            except openai.error.RateLimitError as e:
                print("OpenAI Rate Limit Error... waiting 30 seconds and trying again")
                attempts += 1
                time.sleep(10) # Wait 30 seconds
                continue
            except openai.error.ServiceUnavailableError as e:
                print("OpenAI Service Unavailable Error... waiting 30 seconds and trying again")
                attempts += 1
                time.sleep(30) # Wait 30 seconds
                continue
            break

        if attempts >= 3:
            print("Received 3 repeated OpenAI Service Unavailable Errors... ending execution")


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

        attempts = 0
        while attempts < 3:
            try:
                response = openai.ChatCompletion.create(
                    model=model_name,
                    messages=messages,
                    temperature=temperature,
                    logprobs = bool(logprobs)
                )
            except openai.error.RateLimitError as e:
                print("OpenAI Rate Limit Error... waiting 10 seconds and trying again", file=sys.stderr)
                attempts += 1
                time.sleep(10) # Wait 30 seconds
                continue
            except openai.error.ServiceUnavailableError as e:
                print("OpenAI Service Unavailable Error... waiting 30 seconds and trying again", file=sys.stderr)
                attempts += 1
                time.sleep(30) # Wait 30 seconds
                continue
            except openai.error.APIError as e:
                print("OpenAI API Error... waiting 10 seconds and trying again", file=sys.stderr)
                attempts += 1
                time.sleep(10) # Wait 30 seconds
                continue
            break

        if attempts >= 3:
            print("Received 3 repeated OpenAI Service Unavailable Errors... ending execution", file=sys.stderr)


        if bool(logprobs):
        #logprobs for only the first generated token and alternatives (e.g, 'True', 'False', 'Unc').
        #format: {token: logprob, ... }
            top_logprobs = response.choices[0].logprobs['content'][0]['top_logprobs']
            self.logprobs = {e['token']: e['logprob'] for e in top_logprobs}
        else:
            self.logprobs = None

        return response.choices[0].message['content']
