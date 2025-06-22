import json
import requests


def load_test_data():
    test_data_path = r"C:\Users\Sunny Kumar\Projects\TestAutomation_RAG-LLM\testdata\test2_data.json"
    with open(test_data_path) as f:
        return json.load(f)

def get_llm_response(test_data):
    responseDict = requests.post("https://rahulshettyacademy.com/rag-llm/ask",
                                 json={
                                     "question": test_data["question"],
                                     "chat_history": []
                                 }).json()
    return responseDict