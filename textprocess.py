from langchain_community.vectorstores import FAISS
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# 1) Load both PDFs
loader1 = PyPDFLoader("./static/PRINCIPLES_OF_INTERNAL_MEDICINE.pdf")
loader2 = PyPDFLoader("./static/The_Merck_Manual_of_Diagnosis_and_Therapy_2011 - 19th Edn........pdf")
docs = loader1.load_and_split() + loader2.load_and_split()

# 2) (Optional) re-chunk if you like
# splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
# chunks = splitter.split_documents(docs)

# 3) Embed
emb = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# 4) Build & save FAISS
db = FAISS.from_documents(docs, emb)
db.save_local("faiss_index")
