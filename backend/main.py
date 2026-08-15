from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
import json
import os
import asyncio
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

load_dotenv()

app = FastAPI(title="Marketplace Assistant API")

# Enable CORS for the Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize models and vector store
try:
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    llm = ChatGoogleGenerativeAI(model="gemini-flash-latest", temperature=0.2)
    
    # Load vector store exists
    if os.path.exists("chroma_db"):
        catalogue_vectorstore = Chroma(persist_directory="chroma_db", collection_name="catalogue", embedding_function=embeddings)
        catalogue_retriever = catalogue_vectorstore.as_retriever(search_kwargs={"k": 3})
        
        faq_vectorstore = Chroma(persist_directory="chroma_db", collection_name="faq", embedding_function=embeddings)
        faq_retriever = faq_vectorstore.as_retriever(search_kwargs={"k": 3})
    else:
        print("WARNING: ChromaDB not found. Please run ingest.py first.")
        catalogue_retriever = None
        faq_retriever = None
except Exception as e:
    print(f"Error initializing AI components: {e}")
    catalogue_retriever = None
    faq_retriever = None

class ChatRequest(BaseModel):
    message: str
    history: list[dict] = []

class IntentClassification(BaseModel):
    intent: str = Field(description="The intent of the user's message. Must be exactly one of: 'PRODUCT_SEARCH', 'SUPPORT_QUESTION', 'RFQ_GENERATION', or 'GENERAL_CHAT'.")

class RFQLead(BaseModel):
    product: str = Field(description="The product being requested", default="")
    quantity: str = Field(description="The quantity being requested", default="")
    location: str = Field(description="The delivery location", default="")
    timeline: str = Field(description="The delivery timeline (e.g., 'next week', 'urgent')", default="")
    is_complete: bool = Field(description="True if product, quantity, location, and timeline are ALL explicitly stated in the conversation history.", default=False)

# Structured LLM for intent routing and RFQ extraction
intent_llm = llm.with_structured_output(IntentClassification)
rfq_llm = llm.with_structured_output(RFQLead)

SYSTEM_PROMPT = """You are an advanced AI Marketplace Assistant.
Your goal is to help buyers discover products, find suppliers, draft RFQs (Requests for Quotation), and answer support questions.

[ROUTING CONTEXT]
The user's message was classified with the intent: {intent}.
Relevant information retrieved from the database based on this intent:
<context>
{context}
</context>

INSTRUCTIONS:
1. Answer the user's questions utilizing the retrieved context.
2. If the intent is SUPPORT_QUESTION, the context contains FAQ rules and policies. Use it to answer gracefully.
3. If the intent is PRODUCT_SEARCH, the context contains matching products and suppliers. Summarize them, including supplier names and prices if available.
4. If the intent is RFQ_GENERATION, review the SYSTEM NOTIFICATION in the context. If the RFQ is incomplete, ask the user for the missing details politely. If it is complete, congratulate them that it was submitted.
5. If the intent is GENERAL_CHAT (like "hello" or "how are you"), respond politely and introduce yourself as the Marketplace AI Assistant.
6. Do NOT invent products, suppliers, or policies that are not in the context.

Respond in a helpful, professional, and concise manner.
"""

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    if not catalogue_retriever or not faq_retriever:
        print("WARNING: Database not initialized. Will use demo context.")
    
    # 1. Classify Intent (The "Router Brain")
    try:
        classification = await intent_llm.ainvoke(request.message)
        intent = classification.intent
        print(f"User Message Classified As: {intent}")
    except Exception as e:
        print(f"Intent classification failed: {e}")
        intent = "PRODUCT_SEARCH" # fallback
        
    # 2. Retrieve relevant context based on intent
    context = ""
    try:
        if intent in ["PRODUCT_SEARCH", "RFQ_GENERATION"] and catalogue_retriever:
            docs = await catalogue_retriever.ainvoke(request.message)
            context = "\n\n".join([doc.page_content for doc in docs])
        elif intent == "SUPPORT_QUESTION" and faq_retriever:
            docs = await faq_retriever.ainvoke(request.message)
            context = "\n\n".join([doc.page_content for doc in docs])
        else:
            context = "No database search was required for this query, or database is missing."
    except Exception as e:
        print(f"Retrieval error: {e}")
        context = "Database query failed. Running in demo mode."
        
    # 3. RFQ Conversational Extraction Logic
    if intent == "RFQ_GENERATION":
        try:
            full_convo = "\n".join([msg.get("content", "") for msg in request.history]) + "\nUser: " + request.message
            rfq_data = await rfq_llm.ainvoke(f"Extract RFQ details from this conversation:\n{full_convo}")
            
            if rfq_data.is_complete:
                # Save to mock CRM
                rfq_dict = rfq_data.model_dump()
                crm_path = "data/rfqs.json"
                existing = []
                if os.path.exists(crm_path):
                    with open(crm_path, "r") as f:
                        try:
                            existing = json.load(f)
                        except:
                            pass
                existing.append(rfq_dict)
                with open(crm_path, "w") as f:
                    json.dump(existing, f, indent=4)
                    
                context += f"\n\n[SYSTEM NOTIFICATION: RFQ successfully saved to CRM: {rfq_dict}. Tell the user the RFQ was successfully submitted to our suppliers and they will be contacted shortly.]"
                print(f"--- NEW LEAD CAPTURED --- \n{rfq_dict}")
            else:
                context += f"\n\n[SYSTEM NOTIFICATION: The RFQ is NOT complete yet. Currently extracted: {rfq_data.model_dump()}. Ask the user specifically for the missing fields (product, quantity, location, or timeline).]"
        except Exception as e:
            print(f"RFQ Extraction error: {e}")
    
    # 4. Build prompt
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{message}")
    ])
    
    # 5. Format history
    langchain_history = []
    for msg in request.history:
        if msg.get("role") == "user":
            langchain_history.append(HumanMessage(content=msg.get("content")))
        else:
            langchain_history.append(AIMessage(content=msg.get("content")))
            
    # 6. Create the runnable chain
    chain = prompt_template | llm
    
    # 7. Execute and return stream
    async def generate():
        try:
            async for chunk in chain.astream({
                "intent": intent,
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
            yield f"data: {json.dumps({'content': 'I encountered an error connecting to my AI brain. Please check your API keys.'})}\n\n"
            
        yield "data: [DONE]\n\n"
        
    return StreamingResponse(generate(), media_type="text/event-stream")

@app.get("/health")
def health_check():
    return {"status": "ok", "database_ready": catalogue_retriever is not None}
