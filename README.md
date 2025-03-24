# TEG
python -m venv .venv
windows:
.venv\Scripts\activate
mac:
source ./.venv/bin/Activate

pip install -r requirements.txt

python prepare.py --pdf data/RAG.pdf --index faiss_index
python prepare.py --pdf data/RAG.pdf --index semantic_index --semantic_split
python prepare.py --pdf data/RAG.pdf --index semantic_index --semantic_split
python prepare.py --pdf data/RAG.pdf --index hybrid_index --semantic_split --hybrid_split

streamlit run frontend/app.py
python backend/app.py


{
    "human_message": "What is RAG?",
    "system_message": "You are helpful personal assistant. While responding use only information from received context. If there is no clear answer from the context give user information about Failure",
    "temperature": 0.7
}


http://127.0.0.1:5000/api/chat