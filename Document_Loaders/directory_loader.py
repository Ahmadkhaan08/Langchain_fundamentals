from langchain_community.document_loaders import DirectoryLoader,PyPDFLoader

loader=DirectoryLoader(
    path=r"E:\docs",
    glob="**/*.pdf",
    loader_cls=PyPDFLoader
)

docs=loader.load() # load all docs at once
# docs=loader.lazy_load() # load a single doc at once

for doc in docs:
    print(doc.metadata)

# # print(len(docs))
# print(docs[0].page_content)
# print(docs[0].metadata)
