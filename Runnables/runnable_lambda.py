from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence,RunnableParallel,RunnablePassthrough,RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

def word_count(text):
    return len(text.split())

prompt=PromptTemplate(
    template="Write a joke on {topic}",
    input_variables=['topic']
)

model=ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite")

parser=StrOutputParser()


joke_chain=RunnableSequence(prompt,model,parser)
parallel_chain=RunnableParallel({
    'joke':RunnablePassthrough(),
    'word_count':RunnableLambda(word_count)
})
chain=RunnableSequence(joke_chain,parallel_chain)
res=chain.invoke({'topic':'Cricket'})
final_res="""\n {} word_count:{}""".format(res['joke'],res['word_count'])
# print(res)
print(final_res)