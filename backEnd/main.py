from typing import List, Any

from fastapi import FastAPI, Depends, Form, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from models.chat_models import ChatRequest, ChatResponse, ModelInfo, ChatMessage
from models.conversation_entity import ConversationEntity
from services.llmService import LlmService
from fastapi.responses import StreamingResponse
app = FastAPI()

# Configure CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      # Allows all origins
    allow_credentials=True,   # Allows cookies and credentials
    allow_methods=["*"],      # Allows all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],      # Allows all request headers
)

# Resolve service dependencies
def get_llm_service(
) -> LlmService:
    return LlmService()

@app.post("/chat/", response_model=ChatResponse)
async def say_hello(
    userId: int = Form(...),
    chatId: int = Form(...),
    message: str = Form(...),
    modelName: str = Form(...),
    files: List[UploadFile] = File(default=[]),
    llmService: LlmService = Depends(get_llm_service)
):
    chat_id, messages = await llmService.chat_with_llm_2(userId, chatId, modelName, message, files)
    return ChatResponse(userId=userId, chatId=chat_id, messages=messages)

@app.get("/models/local", response_model=List[ModelInfo])
async def get_local_models(llmService: LlmService = Depends(get_llm_service)):
    return llmService.get_local_model_list()

@app.get("/models/cloud", response_model=List[ModelInfo])
async def get_cloud_models(llmService: LlmService = Depends(get_llm_service)):
    return llmService.get_cloud_model_list()

@app.get("/chat/{chat_id}/messages", response_model=List[ChatMessage])
async def get_chat_messages(chat_id: int, llmService: LlmService = Depends(get_llm_service)):
    return llmService.get_messages_for_chat(chat_id)

@app.get("/conversations/all", response_model=List[ConversationEntity])
async def get_chat_messages(userId: int, llmService: LlmService = Depends(get_llm_service)):
    return llmService.get_conversation(userId)

@app.get("/documents/{chat_id}")
async def list_documents(chat_id: int, llmService: LlmService = Depends(get_llm_service)):
    return llmService.vector_repo.list_documents(chat_id)

@app.delete("/documents/{document_id}")
async def delete_document(document_id: int, llmService: LlmService = Depends(get_llm_service)):
    llmService.vector_repo.delete_document(document_id)
    return {"deleted": document_id}

@app.get("clear/all")
async def clear_history(llmService: LlmService = Depends(get_llm_service)):
    result = llmService.clear_all_history()
    return {"result":result}