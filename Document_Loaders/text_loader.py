from langchain_community.document_loaders import TextLoader
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

model=ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite")

prompt=PromptTemplate(
    template="Write the crux of the following paragraph. \n {paragraph}",
    input_variables=['paragraph']
)

loader=TextLoader("cricket.txt")
docs=loader.load()

parser=StrOutputParser()

chain=prompt | model | parser

res=chain.invoke({'paragraph':docs[0].page_content})

print(res)

# print(docs)
# print(docs[0].page_content)
# print(docs[0].metadata)
