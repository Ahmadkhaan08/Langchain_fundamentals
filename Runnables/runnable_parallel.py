from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence,RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

prompt1=PromptTemplate(
    template="Generate a tweet about {topic}",
    input_variables=['topic']
)

prompt2=PromptTemplate(
    template="Generate a Linkedin post about {topic}",
    input_variables=['topic']
)

model=ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite")

parser=StrOutputParser()

parellel_chain=RunnableParallel({
    'tweet':RunnableSequence(prompt1,model,parser),
    'linkedin':RunnableSequence(prompt2,model,parser)
})
res=parellel_chain.invoke({'topic':'Cricket'})
# print(res)
print("Tweet:",res['tweet'])
print("Linkedin Post:",res['linkedin'])