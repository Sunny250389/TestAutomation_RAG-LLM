# pytest
import os
import requests
import pytest
# from langchain_openai import ChatOpenAI
from langchain_deepseek import ChatDeepSeek
from ragas import SingleTurnSample
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import LLMContextRecall

from utils import get_llm_response, load_test_data



@pytest.mark.asyncio
@pytest.mark.parametrize("getData",
                         load_test_data(), indirect=True)
async def test_context_recall_api(llm_wrapper, getData):
    context_recall = LLMContextRecall(llm=llm_wrapper)
    # score
    score = await context_recall.single_turn_ascore(getData)
    print(score)
    assert score > 0.7


@pytest.fixture()
def getData(request):
    test_data = request.param
    responseDict = get_llm_response(test_data)
    print(responseDict)

    #This section will keep on changing based on the metrics which is being used
    sample = SingleTurnSample(
        user_input=test_data["question"],
        retrieved_contexts=[responseDict["retrieved_docs"][0]["page_content"],
                            responseDict["retrieved_docs"][1]["page_content"],
                            responseDict["retrieved_docs"][2]["page_content"]],
        reference=test_data["reference"]
    )
    return sample