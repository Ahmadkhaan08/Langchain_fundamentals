from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv  import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

load_dotenv()

model=ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite")

# prompt 1:
prompt1=PromptTemplate(
    template="Write a detailed report on {topic}",
    input_variables=['topic']
)

# prompt 2:
prompt2=PromptTemplate(
    template="Write a 5 line summary on the following text. /n {text}",
    input_variables=['text']
)

parser=StrOutputParser()

chain= prompt1 | model | parser | prompt2 | model | parser

result = chain.invoke({'topic':'Black Hole'})

print(result)