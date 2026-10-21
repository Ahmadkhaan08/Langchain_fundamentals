from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv  import load_dotenv
from langchain_classic.output_parsers import StructuredOutputParser,ResponseSchema
from langchain_core.prompts import PromptTemplate

load_dotenv()

model=ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite")

schema=[
    ResponseSchema(name='fact_1',description="fact_1 about the topic"),
    ResponseSchema(name='fact_2',description="fact_1 about the topic"),
    ResponseSchema(name='fact_3',description="fact_1 about the topic")
]

parser=StructuredOutputParser.from_response_schemas(schema)

template=PromptTemplate(
    template="Give 3 facts about the {topic} \n {format_instructions}",
    input_variables=['topic'],
    partial_variables={'format_instructions':parser.get_format_instructions()}
)

# prompt=template.invoke({'topic':'Black Hole'})

# result=model.invoke(prompt)

# final_result=parser.parse(result.content[0]["text"])

# print(final_result)

chain= template | model | parser 

result = chain.invoke({'topic':'Black Hole'})

print(result)