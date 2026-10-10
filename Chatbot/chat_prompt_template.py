from langchain_core.prompts import ChatPromptTemplate

prompt_template=ChatPromptTemplate([
    ('system','You are a helpul {domain} expert'),
    ('human','Tell me about {topic}')
])

res=prompt_template.invoke({'domain':'cricket','topic':'beamer'})
print(res)