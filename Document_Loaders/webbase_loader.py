from langchain_community.document_loaders import WebBaseLoader
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

model=ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite")

prompt=PromptTemplate(
    template="Answer the following question \n {question} from the given \n {text}",
    input_variables=['question','text']
)

parser=StrOutputParser()


url='https://www.geeksforgeeks.org/cpp/object-oriented-programming-in-cpp/'
loader=WebBaseLoader(url)

docs=loader.load()

chain=prompt | model | parser
res=chain.invoke({'question':'what are constructors','text':docs[0].page_content})


print(res)