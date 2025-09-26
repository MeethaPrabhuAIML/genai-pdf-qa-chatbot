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
import os
import openai
import sys
sys.path.append('../..')

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file

openai.api_key  = os.environ['OPENAI_API_KEY']
from langchain.document_loaders import PyPDFLoader
loader = PyPDFLoader("Ex-2.pdf")
pages = loader.load()

len(pages)

page = pages[1]
<img width="928" height="405" alt="Screenshot 2025-09-26 112050" src="https://github.com/user-attachments/assets/8d5d4b56-a97c-4778-91b5-9dae5a33b22d" />

print(page.page_content[0:500])

page.metadata
```

### OUTPUT:
<img width="928" height="405" alt="Screenshot 2025-09-26 112050" src="https://github.com/user-attachments/assets/c1027cc3-14c7-4670-9edc-9ddfbaab5a00" />


### RESULT:
The question-answering chatbot was successfully designed and implemented using the LangChain framework.
