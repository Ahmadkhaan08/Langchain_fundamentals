from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder


# chat template
chat_template=ChatPromptTemplate([
    ('system','You are a helpful customer sales agent'),
    MessagesPlaceholder(variable_name='chat_history'),
    ('human','{query}')
])

chat_history=[]
# load chats
with open('chat_history.txt') as f:
    chat_history.extend(f.readlines())
    
print(chat_history)

# prompt
prompt=chat_template.invoke({'chat_history':chat_history,'query':'where is my refund?'})
print(prompt)