import os
from typing import Optional
import os
import time
import sys
from llms.llm_base import LLMBase
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline

class OSWrapper(LLMBase):
    def __init__(self, config):
        super().__init__(config)

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

        # Hard coding Mistral for now
        model_path = "/model-weights/Mixtral-8x7B-Instruct-v0.1"
        # model_path = "/model-weights/Mistral-7B-v0.1"
        # model_path = "/model-weights/gemma-2b"

        cuda_device = torch.device("cuda")

        self.pipeline = pipeline(task="text-generation",
                                 model=model_path,
                                 tokenizer=model_path,
                                 device_map="auto",
                                 torch_dtype=torch.bfloat16,
                                 return_full_text=False,
                                 model_kwargs={"quantization_config": bnb_config}
                                 )

        # self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        # self.model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True,
        #         quantization_config=bnb_config,
        #         torch_dtype=torch.bfloat16,
        #         device_map='auto',
        #     )

        self.device = "cuda" # Hard-coding this for now
        # self.model.to(self.device) 
        

    def make_request(self, prompt: str, temperature: Optional[float] = 0, logprobs=0, max_tokens=50) -> str:
        """
        Make a request to Open AI's GPT Completion LLM.

        Args:
            prompt: An input to the LLM.
            temperature: The temperature to use for the LLM.

        Returns:
            str: The generated text response.
        """

        # import pdb;pdb.set_trace()

        # model_inputs = self.tokenizer([prompt], return_tensors="pt").to(self.device)

        # generated_ids = self.model.generate(**model_inputs, max_new_tokens=10, do_sample=True)
        # import pdb; pdb.set_trace()
        response_dict = self.pipeline(prompt, do_sample=True, max_new_tokens=max_tokens)
        response = response_dict[0]['generated_text']
        # response = self.tokenizer.batch_decode(generated_ids)[0]

        return response
