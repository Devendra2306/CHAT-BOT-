import json
import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize embeddings
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

CHROMA_PATH = "chroma_db"

def ingest_products_and_suppliers():
    print("Ingesting products and suppliers...")
    
    with open("data/products.json", "r") as f:
        products = json.load(f)
        
    with open("data/suppliers.json", "r") as f:
        suppliers = json.load(f)
        
    supplier_map = {s["id"]: s for s in suppliers}
    
    docs = []
    for p in products:
        supplier = supplier_map.get(p["supplier_id"], {})
        
        # Create a rich text representation for embedding
        content = f"Product: {p['name']}\nCategory: {p['category']}\nDescription: {p['description']}\nPrice: {p['price_range']}\nLocation: {p['location']}\nSupplier: {supplier.get('name', 'Unknown')}\nSupplier Specialties: {', '.join(supplier.get('specialties', []))}"
        
        metadata = {
            "type": "product",
            "id": p["id"],
            "name": p["name"],
            "supplier_id": p["supplier_id"],
            "supplier_name": supplier.get("name", "Unknown"),
            "category": p["category"]
        }
        docs.append(Document(page_content=content, metadata=metadata))
        
    # Store in Chroma
    vectorstore = Chroma.from_documents(
        documents=docs, 
        embedding=embeddings, 
        persist_directory=CHROMA_PATH,
        collection_name="catalogue"
    )
    print(f"Ingested {len(docs)} product documents.")

def ingest_faq():
    print("Ingesting FAQ...")
    loader = TextLoader("data/faq.md")
    documents = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["## ", "\n\n", "\n", " ", ""]
    )
    docs = text_splitter.split_documents(documents)
    
    for doc in docs:
        doc.metadata["type"] = "faq"
        
    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=CHROMA_PATH,
        collection_name="faq"
    )
    print(f"Ingested {len(docs)} FAQ chunks.")

if __name__ == "__main__":
    # Create persistent directory if it doesn't exist
    os.makedirs(CHROMA_PATH, exist_ok=True)
    
    # We could optionally clear the DB first
    # import shutil
    # if os.path.exists(CHROMA_PATH):
    #     shutil.rmtree(CHROMA_PATH)
        
    ingest_products_and_suppliers()
    ingest_faq()
    print("Ingestion complete.")
