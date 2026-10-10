from langchain_core.messages import SystemMessage,HumanMessage,AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv


load_dotenv()

model=ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite")

chat_history=[
    SystemMessage(content="You are a helpful AI Assistant")
]

while True:
    user_input=input('You:')
    chat_history.append(HumanMessage(content=user_input))
    if user_input=="exit":
        break
    response=model.invoke(chat_history)
    chat_history.append(AIMessage(content=response.content[0]["text"]))
    print(response.content[0]["text"])


print(chat_history)
    