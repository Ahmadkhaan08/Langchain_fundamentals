from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser


load_dotenv()

template1=PromptTemplate(
    template="Generate a detail report on {topic}",
    input_variables=['topic']
)

template2=PromptTemplate(
    template="Generate a 5 pointer summary from the following text \n {text}",
    input_variables=['text']
)

model=ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite")

parser=StrOutputParser()

chain=template1 | model | parser | template2 | model | parser
result=chain.invoke({'topic':'Unemployment in Pakistan due to AI'})
print(result)

# chain.get_graph().print_ascii()