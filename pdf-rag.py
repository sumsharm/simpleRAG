# Load the PDF
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import PromptTemplate
from langchain_community.document_loaders.csv_loader import CSVLoader

doc_path = "./data/constitution.pdf"
# doc_path = "./data/student.csv"
model = "llama3.2:latest"


pages = []

if doc_path:
    loader = PyPDFLoader(doc_path)
    # loader = CSVLoader(doc_path)
    for page in loader.load():
        pages.append(page)
    print("done loading....")
    # print(pages[0:50])
else:
    print("Upload a PDF/any file")

# Extract text from PDF files and split into smaller chunk
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200,chunk_overlap=150)
chunks = text_splitter.split_documents(pages)
print("Done splitting")
# print(f"Number of chunks {len(chunks)}")
# print(f"Chunk data {chunks[0]}")


# Add to vector database
vector_db = Chroma.from_documents(
    documents=chunks,
    embedding=OllamaEmbeddings(model="nomic-embed-text:latest"),
    collection_name="simple-rag"
)
print("Done adding to vector database.")

# Retrival
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers.multi_query import MultiQueryRetriever

llm = ChatOllama(model=model)

QUERY_PROMPT = PromptTemplate(
    input_variables=["question"],
    template="""You are an AI language model assistant. Your task is to generate five
    different versions of the given user question to retrieve relevant documents from
    a vector database. By generating multiple perspectives on the user question, your
    goal is to help the user overcome some of the limitations of the distance-based
    similarity search. Provide these alternative questions separated by newlines.
    Original question: {question}""",
)

retriever = MultiQueryRetriever.from_llm(
    vector_db.as_retriever(),
    llm,
    prompt=QUERY_PROMPT
)

# RAG prompt
template = """Answer the question based ONLY on the following context:
{context}
Question: {question}
"""

prompt = ChatPromptTemplate.from_template(template)
chain = (
    {"context": retriever, "question":RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

while True:

    response = chain.invoke(input("Ask me about?"))
    print(response)
    option = input("Do you wish to continue?(y/n)")
    if option.lower() != 'y':
        break
