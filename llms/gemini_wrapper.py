from typing import Optional
import os
import openai
import time
import sys
from llms.llm_base import LLMBase
from utils.key_shuffler import KeyShuffler
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from google.api_core.exceptions import GoogleAPICallError, RetryError, InvalidArgument


class Gemini(LLMBase):
    def __init__(self, config):
        super().__init__(config)

        self.key_file = "./utils/api_keys.json"
        self.key_shuffler = KeyShuffler(self.key_file)

        self.GEMINI_API_KEY = self.key_shuffler.get_next_key()
        genai.configure(api_key = self.GEMINI_API_KEY)

        self.model_name = config["llm"]["gemini_model"]
        self.model = genai.GenerativeModel(self.model_name)
        self.safety_settings = {HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,}



    def make_request(self, prompt: str, temperature = 0) -> str:
        """
        Make a language generation request to Gemini.
        """


        attempts = 0
        while True:
            try:
                response = self.model.generate_content(prompt, safety_settings = self.safety_settings)
                self.logger.debug(response)
            except (GoogleAPICallError, RetryError, InvalidArgument) as e:
                if attempts >= 3:
                    print("Received 3 repeated Gemini API Errors... changing API key")
                    self.GEMINI_API_KEY = self.key_shuffler.get_next_key()
                    genai.configure(api_key = self.GEMINI_API_KEY)
                    attempts = 0
                else:
                    print(f"Gemini API Error {e} \n ... waiting 60 seconds")
                    attempts += 1
                    time.sleep(60) 
                continue
            break

        self.logger.debug(response)    
        return response.text
   
    

# class GPTChatCompletion(LLMBase):
#     def __init__(self, config):
#         super().__init__(config)
        
#     def make_request(self, prompt: str, temperature: float = 0.0, logprobs=0, top_logprobs = 3) -> str:
#         """
#         Make a request to Open AI's GPT Chat Completion LLM.

#         Args:
#             prompt: An input to the LLM.
#             temperature: The temperature to use for the LLM.

#         Returns:
#             str: The generated text response.
#         """
#         model_name = self.config["llm"]["model"]

#         messages = [
#             {"role": "system", "content": "You are a helpful assistant."},
#             {"role": "user", "content": prompt},
#         ]

#         attempts = 0
#         while attempts < 3:
#             try:
#                 response = openai.ChatCompletion.create(
#                     model=model_name,
#                     messages=messages,
#                     temperature=temperature,
#                     logprobs = bool(logprobs)
#                 )
#             except openai.error.RateLimitError as e:
#                 print("OpenAI Rate Limit Error... waiting 10 seconds and trying again", file=sys.stderr)
#                 attempts += 1
#                 time.sleep(10) # Wait 30 seconds
#                 continue
#             except openai.error.ServiceUnavailableError as e:
#                 print("OpenAI Service Unavailable Error... waiting 30 seconds and trying again", file=sys.stderr)
#                 attempts += 1
#                 time.sleep(30) # Wait 30 seconds
#                 continue
#             except openai.error.APIError as e:
#                 print("OpenAI API Error... waiting 10 seconds and trying again", file=sys.stderr)
#                 attempts += 1
#                 time.sleep(10) # Wait 30 seconds
#                 continue
#             break

#         if attempts >= 3:
#             print("Received 3 repeated OpenAI Service Unavailable Errors... ending execution", file=sys.stderr)


#         if bool(logprobs):
#         #logprobs for only the first generated token and alternatives (e.g, 'True', 'False', 'Unc').
#         #format: {token: logprob, ... }
#             top_logprobs = response.choices[0].logprobs['content'][0]['top_logprobs']
#             self.logprobs = {e['token']: e['logprob'] for e in top_logprobs}
#         else:
#             self.logprobs = None

#         return response.choices[0].message['content']
