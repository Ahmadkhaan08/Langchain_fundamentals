from langchain_huggingface import ChatHuggingFace,HuggingFacePipeline

llm=HuggingFacePipeline(
    model_id="openai/gpt-oss-safeguard-20b",
    task="text-generation",
    pipeline_kwargs=dict(
        temperature=0.5,
        max_new_token=100
    )
)

model=ChatHuggingFace(llm=llm)
response=model.invoke("Who is the Prime Minister of Pakistan?")
print(response.content)