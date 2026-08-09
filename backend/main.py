from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import json
import os
import asyncio
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

load_dotenv()

app = FastAPI(title="CloudVault Assistant API")

# Enable CORS for the Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In production, restrict this to the frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

# Initialize models and vector store
try:
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    llm = ChatGoogleGenerativeAI(model="gemini-flash-latest", temperature=0.2)
    # Note: Using gemini-flash-latest as it is extremely fast and free-tier friendly
    
    # Load vector store exists
    if os.path.exists("chroma_db"):
        vectorstore = Chroma(persist_directory="chroma_db", embedding_function=embeddings)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    else:
        print("WARNING: ChromaDB not found. Please run ingest.py first.")
        retriever = None
except Exception as e:
    print(f"Error initializing AI components: {e}")
    retriever = None


class ChatRequest(BaseModel):
    message: str
    history: list[dict] = [] # list of {"role": "user"|"assistant", "content": "..."}

SYSTEM_PROMPT = """You are a highly capable AI Assistant for CloudVault, a secure cloud storage and file management platform.
Your goal is to help users understand CloudVault features, troubleshoot issues, and manage their storage efficiently.

Here is some context retrieved from our database (FAQs, documentation, etc) that is relevant to the user's query:
<context>
{context}
</context>

INSTRUCTIONS:
1. Answer the user's questions based ONLY on the context provided above.
2. If the user is asking about how CloudVault works, use the FAQ context to answer.
3. If the user asks about their files, guide them on how to use the CloudVault dashboard (e.g., using the 'Upload' button, navigating 'My Drive', or checking 'Storage' limits).
4. If the user just says "hi", "hello", or asks for your name, respond politely and introduce yourself as the CloudVault AI Assistant.
5. Do NOT invent features that are not in the context or standard for a cloud storage app.

Respond in a helpful, professional, and concise manner.
"""

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    # Don't strictly block if retriever is missing, allow demo mode
    if not retriever:
        print("WARNING: Database not initialized. Will use demo context.")
    
    # 1. Retrieve relevant context
    context = ""
    try:
        if retriever:
            docs = retriever.invoke(request.message)
            context = "\n\n".join([doc.page_content for doc in docs])
        else:
            context = "No database found. Running in demo mode without retrieval."
    except Exception as e:
        print(f"Retrieval error: {e}")
        context = "Database query failed. Running in demo mode."
    
    # 2. Build prompt
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{message}")
    ])
    
    # 3. Format history
    langchain_history = []
    for msg in request.history:
        if msg.get("role") == "user":
            langchain_history.append(HumanMessage(content=msg.get("content")))
        else:
            langchain_history.append(AIMessage(content=msg.get("content")))
            
    # 4. Create the runnable chain
    chain = prompt_template | llm
    
    # 5. Execute and return stream (simulated streaming for now, we can upgrade to true LangChain streaming)
    async def generate():
        try:
            async for chunk in chain.astream({
                "context": context,
                "history": langchain_history,
                "message": request.message
            }):
                content = chunk.content
                if isinstance(content, list):
                    content = "".join(part.get("text", "") if isinstance(part, dict) else str(part) for part in content)
                elif not isinstance(content, str):
                    content = str(content)
                yield f"data: {json.dumps({'content': content})}\n\n"
        except Exception as e:
            print(f"LLM Error: {e}")
            # Mock fallback if API key is invalid or LLM fails
            yield f"data: {json.dumps({'content': 'I am currently running in **Demo Mode** because the Gemini API key is missing or invalid. '})}\n\n"
            yield f"data: {json.dumps({'content': 'However, I can tell you that we have products like *Industrial Gate Valves* and *Solar Panel Mounts* in our mock database! '})}\n\n"
            yield f"data: {json.dumps({'content': '\\n\\nPlease add your `GOOGLE_API_KEY` to `backend/.env` to enable full AI functionality.'})}\n\n"
            
        yield "data: [DONE]\n\n"
        
    return StreamingResponse(generate(), media_type="text/event-stream")

@app.get("/health")
def health_check():
    return {"status": "ok", "database_ready": retriever is not None}
