from pe_modules.base_pe_module import BasePEModule
from pe_modules.mono_llm_pe_module import MonoLLMPEModule
from pe_modules.dt_pe_module import DTPEModule

# Used to create an instance of the appropriate pe_module class based on the name provided in config.

PE_MODULE_CLASSES = {
    'Base': BasePEModule,
    'MonoLLM': MonoLLMPEModule,
    'DT': DTPEModule,
}