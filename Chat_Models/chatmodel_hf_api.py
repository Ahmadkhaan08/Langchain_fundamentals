from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from dotenv import load_dotenv

load_dotenv()

llm=HuggingFaceEndpoint(
    repo_id="openai/gpt-oss-safeguard-20b",
    task="text-generation"
)

model=ChatHuggingFace(llm=llm)
response=model.invoke("Who is the Prime Minister of Pakistan?")
print(response.content)