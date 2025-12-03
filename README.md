## Development of a PDF-Based Question-Answering Chatbot Using LangChain

### AIM:
To design and implement a question-answering chatbot capable of processing and extracting information from a provided PDF document using LangChain, and to evaluate its effectiveness by testing its responses to diverse queries derived from the document's content.

### PROBLEM STATEMENT:
To create an automated system that can understand and answer questions based on the specific content within a PDF document (Ex-2.pdf). The system must go beyond the general knowledge of a language model and provide answers grounded in the text of the provided file.

### DESIGN STEPS:

#### STEP 1:
Document Loading, We use the PyPDFLoader from LangChain to read the Ex-2.pdf file. This process transforms each page of the PDF into a Document object. 

#### STEP 2:
Document Splitting and Embedding, we will use a RecursiveCharacterTextSplitter to break the text into smaller, overlapping chunks. Each chunk is then converted into a numerical vector using OpenAIEmbeddings.

#### STEP 3:
Vector Storage and Retrieval, The generated embeddings are stored in a vector database (in this case, an in-memory Chroma vector store). This database is then used to create a retriever. 

#### STEP 4:
Question-Answering Chain Construction, Finally, we construct a Retrieval-Augmented Generation (RAG) chain.

### PROGRAM:
```
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain.vectorstores import DocArrayInMemorySearch
from langchain.document_loaders import TextLoader
from langchain.chains import RetrievalQA,  ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import TextLoader
from langchain.document_loaders import PyPDFLoader

def load_db(file, chain_type, k):
    # load documents
    loader = PyPDFLoader(file)
    documents = loader.load()
    # split documents
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    docs = text_splitter.split_documents(documents)
    # define embedding
    embeddings = OpenAIEmbeddings()
    # create vector database from data
    db = DocArrayInMemorySearch.from_documents(docs, embeddings)
    # define retriever
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": k})
    # create a chatbot chain. Memory is managed externally.
    qa = ConversationalRetrievalChain.from_llm(
        llm=ChatOpenAI(model_name=llm_name, temperature=0), 
        chain_type=chain_type, 
        retriever=retriever, 
        return_source_documents=True,
        return_generated_question=True,
    )
    return qa

```
### OUTPUT:
<img width="926" height="592" alt="image" src="https://github.com/user-attachments/assets/103423b5-acd8-4710-ab2a-d43c925bbf82" />


### RESULT:
The question-answering chatbot was successfully designed and implemented using the LangChain framework.
