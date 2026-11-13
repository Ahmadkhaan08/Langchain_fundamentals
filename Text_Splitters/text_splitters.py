from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

loader=PyPDFLoader(file_path="../Document_Loaders/terminologies.pdf")

docs=loader.load()
# print(docs[0].page_content)

text="""
Class
A class in C++ is a user-defined data type that acts as a blueprint for creating objects. It defines the data members and member functions that describe the properties and behavior of the objects.

A class is a user-defined data type.
It contains data members and member functions.
Multiple objects can be created from a single class.
Example: A Car represents a class (blueprint), while BMW, Mercedes, and Audi are objects (instances) created from that class.

Class
Object
An object in C++ is an instance of a class that represents a real-world entity. It is used to access the data members and member functions defined in the class.

State: Represents the current values of an object's data members.
Behavior: Represents the actions an object can perform through its member functions.
Identity: Every object has a unique memory location that distinguishes it from other objects, even if they contain the same data.
Example: Dog is a class, while Tommy is an object (instance) of that class."""

splitter=CharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=0,
    separator=""
)

chunks=splitter.split_documents(docs)
# chunks=splitter.split_text(text)
print(chunks[0])
