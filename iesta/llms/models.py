

### HELPERS

# END OF HELPERS


from abc import abstractmethod
import dataclasses
from typing import ClassVar
from transformers import pipeline
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain import HuggingFacePipeline
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

import iesta.llms.prompts as prompts
from langchain.prompts.prompt import PromptTemplate


class IestaLLM():
    name = ""

    def __init__(self):
        self.llm = self.load_langchain_llm()

    @abstractmethod
    def load_langchain_llm(self, model_name_path: str = None):
        pass

    @abstractmethod
    def get_prompt_template(self, instruction: str, 
                            new_system_prompt: str ):
        print("should never be called")
        pass


class ChatGpt(IestaLLM):
    name = "ChatGpt"

    def get_prompt_template(self, instruction: str,
                            new_system_prompt: str = prompts.IESTA_SYSTEM_PROMPT ):
        
        system_message_prompt = SystemMessagePromptTemplate.from_template(
            new_system_prompt
        )
        human_message_prompt = HumanMessagePromptTemplate.from_template(
            instruction
        )

        prompt = ChatPromptTemplate.from_messages(
             [system_message_prompt, human_message_prompt]
            )
        print(f"*** prompt WITHIN {prompt}")
        return prompt

    def load_langchain_llm(self, model_name_path: str = None):
        self.model_name_path = "gpt-3.5-turbo" if model_name_path is None else model_name_path
        print(f"loading model {self.model_name_path}")
        
        llm = ChatOpenAI(
            model_name=self.model_name_path,
            temperature=0
            )
        return llm


class LlamaV2(IestaLLM):
 
    name = "llamav2"

    B_INST: ClassVar = "[INST]"
    E_INST: ClassVar = "[/INST]"
    B_SYS: ClassVar = "<<SYS>>\n"
    E_SYS: ClassVar = "\n<</SYS>>\n\n"

    def get_prompt_template(self, instruction, new_system_prompt: str = prompts.IESTA_SYSTEM_PROMPT ) :
        SYSTEM_PROMPT = LlamaV2.B_SYS + new_system_prompt + LlamaV2.E_SYS
        prompt_str = LlamaV2.B_INST + SYSTEM_PROMPT + instruction + LlamaV2.E_INST

        prompt_template = PromptTemplate(
            template=prompt_str,
            input_variables=["text"],
            )

        return prompt_template

    def load_langchain_llm(self, model_name_path: str = None) -> HuggingFacePipeline:

        self.model_name_path = "meta-llama/Llama-2-7b-chat-hf" if model_name_path is None else model_name_path
        print(f"loading model {self.model_name_path}")

        tokenizer = AutoTokenizer.from_pretrained(self.model_name_path,)

        model = AutoModelForCausalLM.from_pretrained(self.model_name_path,
                                                     device_map='auto',
                                                     torch_dtype=torch.float16,
                                                     )
        pipe = pipeline("text-generation",
                        model=model,
                        tokenizer=tokenizer,
                        torch_dtype=torch.bfloat16,
                        device_map="auto",
                        max_new_tokens=2048,
                        do_sample=True,
                        top_k=30,
                        num_return_sequences=1,
                        eos_token_id=tokenizer.eos_token_id,
                        )
        
        llm = HuggingFacePipeline(pipeline=pipe,
                                  model_kwargs={
                                      'temperature': 0
                                      })

        return llm


    # For future reference - NOT USED IN IESTA
    DEFAULT_SYSTEM_PROMPT = """\
    You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

    If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."""
    
