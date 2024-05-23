import os
from dotenv import load_dotenv
load_dotenv()

from promptflow.core import Prompty, AzureOpenAIModelConfiguration
from promptflow.evals.evaluators import ChatEvaluator

model_config = AzureOpenAIModelConfiguration(
    azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY")
)

chat_history=[
    {"role": "user", "content": "Does Azure OpenAI support customer managed keys?"},
    {"role": "assistant", "content": "Yes, customer managed keys are supported by Azure OpenAI."}
]
chat_input="Do other Azure AI services support this too?"

prompty = Prompty.load("chat.prompty", model={'configuration': model_config})
response = prompty(chat_history=chat_history, chat_input=chat_input)

conversation = chat_history
conversation += [
    {"role": "user", "content": chat_input},
    {"role": "assistant", "content": response}
]

chat_eval = ChatEvaluator(model_config=model_config)
score = chat_eval(conversation=conversation)

print(score)