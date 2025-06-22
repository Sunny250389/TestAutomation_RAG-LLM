# pytest
import os
import requests
import pytest
# from langchain_openai import ChatOpenAI
from langchain_deepseek import ChatDeepSeek
from ragas import SingleTurnSample
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import LLMContextPrecisionWithoutReference


@pytest.mark.integration
@pytest.mark.asyncio
async def test_context_precision_api():
    # power of LLM + method metric -> score
    # llm = ChatOpenAI(model="gpt-4", temperature=0)
    os.environ["DEEPSEEK_API_KEY"] = "sk-30745902272040a49a4f4ab90e823150"
    llm = ChatDeepSeek(model="deepseek-chat", temperature=0)
    langchain_llm = LangchainLLMWrapper(llm)
    context_precision = LLMContextPrecisionWithoutReference(llm=langchain_llm)
    question = "How many articles are there in the Selenium Webdriver python course"

    #feed data
    responseDict = requests.post("https://rahulshettyacademy.com/rag-llm/ask",
                                 json={
                                     "question": question,
                                     "chat_history": []
                                 }).json()
    print(responseDict)

    sample = SingleTurnSample(
        user_input=question,
        response=responseDict["answer"],
        retrieved_contexts=[responseDict["retrieved_docs"][0]["page_content"],
                            responseDict["retrieved_docs"][1]["page_content"],
                            responseDict["retrieved_docs"][2]["page_content"]]
    )

    # score
    score = await context_precision.single_turn_ascore(sample)
    print(score)
