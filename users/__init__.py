from users.base_user import UserSim
from users.llm_user import LLMUserSim

# Used to create an instance of the appropriate user class based on the name provided in config.

USER_CLASSES = {
    'LLM': LLMUserSim,
    'Base': UserSim,
}