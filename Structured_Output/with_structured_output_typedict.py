from langchain_google_genai import ChatGoogleGenerativeAI
from typing import TypedDict,Annotated,Optional,Literal
from dotenv import load_dotenv

load_dotenv()


model=ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite")

# schemas
class Review(TypedDict):
    key_themes:Annotated[list[str],"Write down all the key themes discussed in the review"]
    summary:Annotated[str,"A brief summary of the review"]
    sentiment:Annotated[Literal["pos","neg"],"Return sentiment of the review either positive,negitive or neutral"]
    pros:Annotated[Optional[list[str]],"Write down all the pros inside a list"]
    cons:Annotated[Optional[list[str]],"Write down all the cons inside a list"]

structured_model=model.with_structured_output(Review)    
# response=structured_model.invoke("""The hardware is great, but the software feels bloated.
# There are too many pre-installed apps that I can't remove. Also, the UI looks outdated
# compared to other brands. Hoping for a software update to fix this.""")

response=structured_model.invoke("""I recently upgraded to the Google Pixel 9 Pro XL, and I must say, it's an absolute powerhouse! The Google Tensor G4 processor makes everything lightning fast—whether I'm gaming, multitasking, or editing photos. The 5060mAh battery easily lasts a full day even with heavy use, and the 37W fast charging is a lifesaver.

The Gemini AI integration is a great touch for summarizing notes and quick assistance, though I don't use it constantly. What really blew me away is the 50MP camera—the night sight mode is stunning, capturing crisp, vibrant images even in low light. Zooming up to 30x Super Res Zoom actually works well for distant objects, but anything beyond that loses noticeable sharpness.

However, the weight and flat-edge design make it a bit uncomfortable for one-handed use. Also, the charging speed still lags behind Chinese flagships—why does a flagship take over an hour to top off? The $1,100 price tag is also a hard pill to swallow.

Pros:
Smooth software experience and intelligent on-device AI features
Stunning 50MP camera with incredible low-light performance
Long battery life with dependable full-day endurance
Clean UI and guaranteed 7 years of major updates

Cons:
Bulky and heavy—not great for one-handed use
Charging speeds are slower compared to competitors
Expensive entry price for the base storage model""")


print(response)
print(response['sentiment'])