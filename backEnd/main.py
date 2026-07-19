from typing import List, Any

from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from models.chat_models import ChatRequest, ChatResponse, ModelInfo, ChatMessage
from models.conversation_entity import ConversationEntity
from services.demo_service import DemoService
from services.llmService import LlmService

app = FastAPI()

# Configure CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      # Allows all origins
    allow_credentials=True,   # Allows cookies and credentials
    allow_methods=["*"],      # Allows all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],      # Allows all request headers
)
def getDemoService():
    return DemoService()
def getLlmService():
    return LlmService()
@app.get("/")
async def root(name_of_user:str,demo_service: DemoService = Depends(getDemoService)):
    return {"message": demo_service.say_hello(name_of_user)}

@app.post("/chat/", response_model=ChatResponse)
async def say_hello(request: ChatRequest, llmService: LlmService = Depends(getLlmService)):
    chat_id, messages = llmService.chatWithLlm(request.userId, request.chatId, request.message)
    return ChatResponse(userId=request.userId, chatId = chat_id, messages=messages)

@app.get("/models/local", response_model=List[ModelInfo])
async def get_local_models(llmService: LlmService = Depends(getLlmService)):
    return llmService.getLocalModelList()

@app.get("/models/cloud", response_model=List[ModelInfo])
async def get_cloud_models(llmService: LlmService = Depends(getLlmService)):
    return llmService.getCloudModelList()

@app.get("/chat/{chat_id}/messages", response_model=List[ChatMessage])
async def get_chat_messages(chat_id: int, llmService: LlmService = Depends(getLlmService)):
    return llmService.getMessagesForChat(chat_id)

@app.get("/conversations/all", response_model=List[ConversationEntity])
async def get_chat_messages(userId: int, llmService: LlmService = Depends(getLlmService)):
    return llmService.getConversation(userId)