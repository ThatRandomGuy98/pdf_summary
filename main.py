from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.tools import tool
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain


from pathlib import Path
from typing import List
PATH = Path(r"C:\Users\delga\Documents\trabalho\universidade\Mestrado\dissertation\Generative AI for Personalized Meal Planning A Recommendation System for Retail Customers.pdf")

#---------------------------------------------------------------------------------------------------
def pdf_loader(path: Path) -> List[Document]:
    if not path.exists():
        raise FileNotFoundError(f"PDF not found on this path -> {path}")
    
    loader = PyPDFLoader(str(path))
    return loader.load()

docs = pdf_loader(path=PATH)
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.split_documents(docs)
print(f"Split the thesis into {len(chunks)} chunks")
# print(chunks[:30])

#---------------------------------------------------------------------------------------------------
#I got some chunks, now to turn them into vectors
embeddings = OllamaEmbeddings(model="nomic-embed-text")
PERSIST_DIR = "thesis_chroma_db"
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory=PERSIST_DIR
)
retriever = vectorstore.as_retriever(search_kwargs={'k': 5})

llm = ChatOllama(model="llama3.1", temperature=0.5)
prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a thesis assistant. Answer using only the provided context."),
    ("human",
     "Question: {input}\n\nContext:\n{context}")
])
doc_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, doc_chain)
print(
    "\nAsk questions about the thesis."
    "\nType 'exit' or 'quit' to stop.\n"
    )


#---------------------------------------------------------------------------------------------------
while True:
    try:
        question = input("You> ").strip()
    except (KeyboardInterrupt, EOFError):
        print("\nSee you soon :)")
        break

    if not question:
        continue

    if question.lower() in {"exit", "quit", "q"}:
        print("See you soon :)")
        break

    result = rag_chain.invoke({"input": question})

    print("\n--- ANSWER ---\n")
    print(result["answer"])
    print("\n" + "-" * 40 + "\n")


        




