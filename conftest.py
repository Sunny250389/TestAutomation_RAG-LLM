import os
import requests
import pytest
from langchain_deepseek import ChatDeepSeek
from ragas import SingleTurnSample
from ragas.llms import LangchainLLMWrapper

os.environ["DEEPSEEK_API_KEY"] = "sk-30745902272040a49a4f4ab90e823150"


@pytest.fixture
def llm_wrapper():
    llm = ChatDeepSeek(model="deepseek-chat", temperature=0)
    langchain_llm = LangchainLLMWrapper(llm)
    return langchain_llm