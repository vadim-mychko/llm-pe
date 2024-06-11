from llms.llm_base import LLMBase
from llms.openai_wrapper import GPTChatCompletion, GPTCompletion
from llms.gemini_wrapper import Gemini
from llms.os_wrapper import OSWrapper


LLM_CLASSES = {
    'GPTChatCompletion': GPTChatCompletion,
    'GPTCompletion': GPTCompletion,
    'Mistral': OSWrapper,
    'Gemini': Gemini
}