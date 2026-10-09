from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()

embeddings=GoogleGenerativeAIEmbeddings(model="gemini-embedding-2-preview")
documents = [
    "Babar Azam is a Pakistani cricketer known for his elegant cover drives and consistent batting.",
    "Imran Khan is a legendary former Pakistani captain famous for his leadership and leading Pakistan to the 1992 World Cup victory.",
    "Wasim Akram, also known as the 'Sultan of Swing', is widely regarded as one of the greatest fast bowlers in cricket history.",
    "Shahid Afridi is famous for his aggressive batting, record-breaking sixes, and effective leg-spin bowling.",
    "Shaheen Shah Afridi is a Pakistani fast bowler known for his lethal opening spells and sharp yorkers."
]
query="tell me about imran khan"

doc_vector=embeddings.embed_documents(documents)
query_vector=embeddings.embed_query(query)

scores=cosine_similarity([query_vector],doc_vector)[0]

index,score=sorted(list(enumerate(scores)),key=lambda x:x[1])[-1]

print(query)
print(documents[index])
print("Similarity score is: ", score )
