from langchain_core.messages import SystemMessage,HumanMessage,AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv


load_dotenv()

model=ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite",max_tokens=50)

messages=[
    SystemMessage(content="You are a helpful assistant"),
    HumanMessage(content="Tell me about langchain")
]

response=model.invoke(messages)
messages.append(AIMessage(content=response.content[0]["text"]))
print(messages)