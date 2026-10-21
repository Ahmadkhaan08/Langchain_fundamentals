from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda,RunnableBranch
from pydantic import BaseModel,Field
from typing import Literal
from langchain_core.output_parsers import PydanticOutputParser

load_dotenv()



model=ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite")
parser=StrOutputParser()

class Review(BaseModel):
    sentiment:Literal["positive","negative"]=Field(description="Give the sentiment of the feedback")
    
pydantic_parser=PydanticOutputParser(pydantic_object=Review)

template1=PromptTemplate(
    template="Classify the sentiment of the following feedback text into positive or negative \n {feedback} \n {format_instructions}",
    input_variables=['feedback'],
    partial_variables={'format_instructions':pydantic_parser.get_format_instructions()}
)

classifier_chain=template1 | model | pydantic_parser

template2 = PromptTemplate(
    template="Write an appropriate response to this positive feedback \n {feedback}",
    input_variables=["feedback"],
)

template3 = PromptTemplate(
    template="Write an appropriate response to this negative feedback \n {feedback}",
    input_variables=["feedback"],
)



branch_chain=RunnableBranch(
    (lambda x:x.sentiment=="positive",template2 | model | parser),
    (lambda x:x.sentiment=="negative",template3 | model | parser),
    RunnableLambda(lambda x:"could not find sentiment")
)

chain=classifier_chain | branch_chain | parser

res=chain.invoke({'feedback':'The T-shirt is comfortable, stylish, and made of excellent-quality fabric.'})
print(res)
