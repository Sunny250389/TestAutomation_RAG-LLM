# pytest
import os
import asyncio
import pytest
# from langchain_openai import ChatOpenAI
from langchain_deepseek import ChatDeepSeek
from ragas import SingleTurnSample
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import LLMContextPrecisionWithoutReference


@pytest.mark.integration
@pytest.mark.asyncio
async def test_context_precision():
    # power of LLM + method metric -> score
    # llm = ChatOpenAI(model="gpt-4", temperature=0)
    os.environ["DEEPSEEK_API_KEY"] = "sk-30745902272040a49a4f4ab90e823150"
    llm = ChatDeepSeek(model="deepseek-chat", temperature=0)
    langchain_llm = LangchainLLMWrapper(llm)
    context_precision = LLMContextPrecisionWithoutReference(llm=langchain_llm)

    sample = SingleTurnSample(
        user_input="How many articles are there in the Selenium webdriver python course?",
        response="There are 23 articles in the course.",
        retrieved_contexts=["Complete Understanding on Selenium Python API Methods with real time Scenarios on LIVE "
                            "Websites\n\"Last but not least\" you can clear any Interview and can Lead Entire Selenium "
                            "Python Projects from Design Stage\nThis course includes:\n17.5 hours on-demand "
                            "video\nAssignments\n23 articles\n9 downloadable resources\nAccess on mobile and"
                            " TV\nCertificate of completion\nRequirements",
                            "What you'll learn\n*****By the end of this course,You will be Mastered on Selenium "
                            "Webdriver with strong Core JAVA basics\n****You will gain the ability to design "
                            "PAGEOBJECT, DATADRIVEN&HYBRID Automation FRAMEWORKS from scratch\n*** InDepth "
                            "understanding of real time Selenium CHALLENGES with 100 + examples\n*Complete knowledge"
                            " on TestNG, MAVEN,ANT, JENKINS,LOG4J, CUCUMBER, "
                            "HTML REPORTS,EXCEL API, GRID PARALLEL TESTING"]
    )

    # score
    score = await context_precision.single_turn_ascore(sample)
    print(score)
