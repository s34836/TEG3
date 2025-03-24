# app.py
from flask import Flask, request, jsonify
import os
from dotenv import load_dotenv
import openai
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
os.environ["TOKENIZERS_PARALLELISM"] = "false"

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


class OpenAIService:
    def __init__(self, api_key, default_model="gpt-4o-mini", default_temperature=0.7):
        self.api_key = api_key
        self.default_model = default_model
        self.default_temperature = default_temperature
        self.client = openai.OpenAI(api_key=self.api_key)

    def get_response(self, human_message, system_message=None, model=None, temperature=None):
        model = model or self.default_model
        temperature = temperature if temperature is not None else self.default_temperature

        messages = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        messages.append({"role": "user", "content": human_message})

        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature
            )
            return response.choices[0].message.content.strip()

        except Exception as e:
            return f"Error communicating with OpenAI API: {str(e)}"

app = Flask(__name__)

# Load environment variables
dotenv_path = os.path.join('../', 'config', '.env')
load_dotenv(dotenv_path)
api_key = os.getenv('OPENAI_API_KEY_TEG')
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
print("Using local HuggingFace embedding model: sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.load_local('faiss_index', embedding_model, allow_dangerous_deserialization=True)
print("✅ Vector store loaded successfully")

default_model = os.getenv('DEFAULT_MODEL', 'gpt-4o-mini')
default_temperature = float(os.getenv('DEFAULT_TEMPERATURE', '0.7'))

openai_service = OpenAIService(
    api_key=api_key,
    default_model=default_model,
    default_temperature=default_temperature
)

@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.json
    
    if not data or 'human_message' not in data:
        return jsonify({'error': 'Human message is required'}), 400
    
    human_message = data.get('human_message')
    system_message = data.get('system_message')
    model = data.get('model', default_model)
    temperature = data.get('temperature', default_temperature)
    context = fetch_conext(human_message)
    human_message = f'USER QUERY: {human_message} \n --- \n CONTEXT: {context}'

    response = openai_service.get_response(
        human_message=human_message,
        system_message=system_message,
        model=model,
        temperature=temperature
    )
    print(response)
    if response:
        return jsonify({'response': response})
    else:
        return jsonify({'error': 'Failed to get response from OpenAI'}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy'})

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True)