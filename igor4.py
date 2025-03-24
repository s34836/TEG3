print("jeden")
import os
import time
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
os.environ["TOKENIZERS_PARALLELISM"] = "false"

print("dwa")
def fetch_conext(query):
    search_results = vectorstore.similarity_search_with_score(query, k=5)

    context = ""
    # Print the results
    print("\nSearch Results:")
    for i, (doc, score) in enumerate(search_results):
        print(f"Result {i+1} (Score: {score:.4f}):")
        print(f"Source: Page {doc.metadata.get('page', 'Unknown')}")
        print(f"Content: {doc.page_content}...\n")
        context += doc.page_content + ". "
    
    return context


# Load environment variables from custom path
dotenv_path = os.path.join('..', 'config', '.env')
load_dotenv(dotenv_path)
api_key = os.getenv('OPENAI_API_KEY_TEG')
os.environ['OPENAI_API_KEY'] = api_key
print("trzy")
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
print("Using local HuggingFace embedding model: sentence-transformers/all-MiniLM-L6-v2")
print("cztery")
vectorstore = FAISS.load_local('faiss_index', embedding_model, allow_dangerous_deserialization=True)
print("✅ Vector store loaded successfully")


user_query = "What is RAG?"
system_message = "You are helpful personal assistant. While responding use only information from received context. If there is no clear answer from the context give user information about Failure"
context = fetch_conext(user_query)
human_message = f'USER QUERY: {user_query} \n --- \n CONTEXT: {context}'

chat = ChatOpenAI(
    model_name="gpt-4o-mini",
    openai_api_key=os.environ['OPENAI_API_KEY'],
)

response = chat.invoke([
    SystemMessage(content=system_message),
    HumanMessage(content=human_message)
])

print(response.content)
