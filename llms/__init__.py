from llms.llm_base import LLMBase
from llms.openai_wrapper import GPTChatCompletion, GPTCompletion


LLM_CLASSES = {
    'GPTChat': GPTChatCompletion,
    'GPT': GPTCompletion
}