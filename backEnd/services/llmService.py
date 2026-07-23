# services/llm_service.py
import base64
import json
import traceback
import uuid
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import List, Dict, Any, Optional

from fastapi import UploadFile

from agents.ollama_agent import OllamaAgent
from folder_tools import FolderReadTool, FolderWriteTool
from models import conversation_entity
from models.chat_models import ChatMessage
from repository.repository import ConversationRepository
from web_search_tool import WebSearchTool

IMAGE_UNAVAILABLE_MESSAGE = "Sorry, I can't process images yet."
UPLOAD_ROOT = Path("uploads")

TEXT_EXTENSIONS = {".txt", ".md", ".csv", ".json", ".log", ".py", ".js", ".ts", ".html", ".css", ".yaml", ".yml"}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".webp"}
PDF_EXTENSIONS = {".pdf"}
DOCX_EXTENSIONS = {".docx"}

class LlmService:
    def __init__(self):
        self.repo = ConversationRepository()

        self.local_models = {
            1: ("Gemma 3", "ollama:gemma3"),
            2: ("Gemma 4", "ollama:gemma4"),
            3: ("Qwen 3", "ollama:qwen3"),
            4: ("Llama 3.1", "ollama:llama3.1"),
        }

        self.cloud_models = {
            1: ("GPT-OSS", "ollama:gpt-oss:120b-cloud"),
            2: ("GPT-5 Mini", "openai:gpt-5-mini"),
            3: ("Claude Sonnet 4", "anthropic:claude-sonnet-4"),
            4: ("Gemini 2.5 Pro", "google_genai:gemini-2.5-pro"),
        }

        self.starter_system_prompt = (
            """
            You are a helpful, accurate, and concise AI assistant.
            Your primary goal is to answer the user's questions as directly as possible. Use your own knowledge first whenever it is sufficient. Only use tools when they are necessary to complete the user's request.
            You have access to the following tools:
            1. Read File
               - Use this tool when you need to read the contents of a file requested by the user.
               - Only read files that are explicitly referenced by the user or are required to complete the task.
               - Do not assume file paths. If the path is ambiguous, ask the user for clarification.
               - Read only the files that are necessary.
            2. Write File
               - Use this tool when the user explicitly asks you to create, modify, append, or overwrite a file.
               - Never write or modify files unless the user has requested it.
               - Generate the content first, then use the write tool to save it.
               - Inform the user what file was created or modified.
            General Tool Usage Rules:
            - Never invent tool results.
            - If a tool fails, explain the failure and, if possible, suggest how the user can resolve it.
            - If a request can be answered without tools, do not call any tools.
            - If multiple tools are required, use only the minimum number necessary.
            - Do not repeatedly call the same tool unless new information is required.
            Conversation Guidelines:
            - Be truthful. If you do not know something, say so.
            - Ask clarifying questions whenever the user's request is ambiguous.
            - Explain your reasoning briefly when it helps the user understand your answer.
            - Keep responses concise unless the user requests a detailed explanation.
            - Preserve formatting when reading or writing code, JSON, Markdown, or configuration files.
            - When generating source code, follow best practices and produce clean, maintainable code.
            Your objective is to be helpful while using tools responsibly and only when they provide information or capabilities that you do not already possess.
            You have read/write access to the current project (including "
            "./services and ./repo) and can also look one level outside it. "
            "To browse outside, call read_folder_or_file with path='..' to list "
            "what's there, then drill into whatever folder name you find "
            "(e.g. '../frontEnd').
            "You can also search the web using web_search when you need current "
            "information, facts you're unsure of, or anything outside this codebase."
            """
        )

        self._reader = FolderReadTool()
        self._writer = FolderWriteTool()

        self._fs_tools = [self._reader, self._writer, WebSearchTool]

    async def parse_files(self, chat_id: int, files: List[UploadFile]) -> List[Dict[str, Any]]:
        """
        Saves each uploaded file to disk under uploads/{chat_id}/, then parses
        it by type. Returns per-file dicts carrying both the on-disk path
        (for DB persistence / later re-parsing) and the extracted content
        (for use in *this* turn's agent call only — never persisted).
        """
        parsed: List[Dict[str, Any]] = []
        chat_dir = UPLOAD_ROOT / str(chat_id)
        chat_dir.mkdir(parents=True, exist_ok=True)

        for file in files:
            filename = file.filename or "unnamed_file"
            ext = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
            raw_bytes = await file.read()

            stored_name = f"{uuid.uuid4().hex}_{filename}"
            stored_path = chat_dir / stored_name
            with open(stored_path, "wb") as f:
                f.write(raw_bytes)

            entry: Dict[str, Any] = {
                "filename": filename,
                "path": str(stored_path),
                "content_type": file.content_type,
                "kind": "unsupported",
                "text": None,
                "base64_data": None,
            }

            try:
                if ext in TEXT_EXTENSIONS:
                    entry["kind"] = "text"
                    entry["text"] = raw_bytes.decode("utf-8", errors="replace")

                elif ext in PDF_EXTENSIONS:
                    entry["kind"] = "pdf"
                    try:
                        from pypdf import PdfReader
                        import io
                        reader = PdfReader(io.BytesIO(raw_bytes))
                        entry["text"] = "\n".join(page.extract_text() or "" for page in reader.pages)
                    except Exception:
                        entry["text"] = None

                elif ext in DOCX_EXTENSIONS:
                    entry["kind"] = "docx"
                    try:
                        import docx
                        import io
                        doc = docx.Document(io.BytesIO(raw_bytes))
                        entry["text"] = "\n".join(p.text for p in doc.paragraphs)
                    except Exception:
                        entry["text"] = None

                elif ext in IMAGE_EXTENSIONS:
                    entry["kind"] = "image"
                    entry["base64_data"] = base64.b64encode(raw_bytes).decode("utf-8")

                else:
                    entry["kind"] = "unsupported"

            except Exception:
                entry["kind"] = "error"
                entry["text"] = None

            parsed.append(entry)

        return parsed

    def _build_attachment_records(self, parsedFileData: List[Dict[str, Any]]) -> Optional[List[Dict[str, Any]]]:
        """DB-safe record — path + metadata only, never raw text/base64."""
        if not parsedFileData:
            return None
        return [
            {
                "filename": entry["filename"],
                "path": entry["path"],
                "content_type": entry["content_type"],
                "kind": entry["kind"],
            }
            for entry in parsedFileData
        ]

    def _has_image(self, parsedFileData: List[Dict[str, Any]]) -> bool:
        return any(entry["kind"] == "image" for entry in parsedFileData)

    def _build_enriched_conversation(self, conversation: List[ChatMessage], parsedFileData: List[Dict[str, Any]]) -> \
        List[Dict[str, Any]]:
        """
        Returns a NEW list of plain dicts for the agent call only — `conversation`
        (which gets persisted) is left untouched. Appends extracted text/pdf/docx
        content to the last (new user) message. Assumes no images at this point.
        """
        agent_conversation = [{"role": m.role, "content": m.content} for m in conversation]

        extra_text = ""
        for entry in parsedFileData:
            if entry.get("text"):
                extra_text += f"\n\n---\nFile: {entry['filename']}\n{entry['text']}"

        if extra_text and agent_conversation:
            agent_conversation[-1]["content"] += extra_text

        return agent_conversation

    async def chatWithLlm(self, user_id: int, chat_id: int, model_name:str, message: str, files: Optional[List[UploadFile]] = None) -> tuple[int, List[ChatMessage]]:
        files = files or []
        conversation: List[ChatMessage] = []
        try:
            if not chat_id or chat_id == 0:
                max_chat_id = self.repo.get_max_chat_id()
                chat_id = self.repo.create_conversation(
                    SimpleNamespace(
                        chatId=chat_id,
                        userId=user_id,
                        chatName=f"Conversation #{max_chat_id+1}",
                        created_at=datetime.utcnow(),
                    )
                )

            rows = self.repo.get_messages(chat_id)

            last_sequence_no = 0
            for row in rows:
                _, _, role, msg_text, _, sequence_no, _ = row
                conversation.append(ChatMessage(role=role, content=msg_text))
                last_sequence_no = max(last_sequence_no, sequence_no)

            if not conversation:
                conversation.append(ChatMessage(role="system", content=self.starter_system_prompt))

            conversation.append(ChatMessage(role="user", content=message))

            parsedFileData = await self.parse_files(chat_id, files)
            attachment_records = self._build_attachment_records(parsedFileData)

            now = datetime.utcnow()
            last_sequence_no += 1
            self.repo.create_message(
                SimpleNamespace(
                    id=0,
                    chatId=chat_id,
                    role="user",
                    message=message,
                    sequenceNo=last_sequence_no,
                    created_at=now,
                    attachments=json.dumps(attachment_records) if attachment_records else None,
                )
            )

            if parsedFileData and self._has_image(parsedFileData):
                assistant_msg = ChatMessage(role="assistant", content=IMAGE_UNAVAILABLE_MESSAGE)
                last_sequence_no += 1
                self.repo.create_message(
                    SimpleNamespace(
                        id=0,
                        chatId=chat_id,
                        role="assistant",
                        message=assistant_msg.content,
                        sequenceNo=last_sequence_no,
                        created_at=datetime.utcnow(),
                        attachments=None,
                    )
                )
                return chat_id, [assistant_msg]

            # _, model_name = self.cloud_models[1]
            agent = OllamaAgent(model_name=model_name, tools=self._fs_tools, system_prompt=self.starter_system_prompt)

            agent_conversation = self._build_enriched_conversation(conversation, parsedFileData) if parsedFileData else conversation
            full_response = agent.invoke(agent_conversation)
            new_messages = full_response[len(agent_conversation):]

            for new_msg in new_messages:
                last_sequence_no += 1
                self.repo.create_message(
                    SimpleNamespace(
                        id=0,
                        chatId=chat_id,
                        role=new_msg.role,
                        message=new_msg.content,
                        sequenceNo=last_sequence_no,
                        created_at=datetime.utcnow(),
                        attachments=None,
                    )
                )

            return chat_id, new_messages
        except Exception as e:
            print(f"Error: {e}")
            print(traceback.format_exc())
            return chat_id, []

    async def chatWithLlmStream(self, user_id: int, chat_id: int, model_name:str, message: str, files: Optional[List[UploadFile]] = None):
        files = files or []
        conversation: List[ChatMessage] = []
        try:
            if not chat_id or chat_id == 0:
                max_chat_id = self.repo.get_max_chat_id()
                chat_id = self.repo.create_conversation(
                    SimpleNamespace(
                        chatId=chat_id,
                        userId=user_id,
                        chatName=f"Conversation #{max_chat_id + 1}",
                        created_at=datetime.utcnow(),
                    )
                )

            rows = self.repo.get_messages(chat_id)

            last_sequence_no = 0
            for row in rows:
                _, _, role, msg_text, _, sequence_no, _ = row
                conversation.append(ChatMessage(role=role, content=msg_text))
                last_sequence_no = max(last_sequence_no, sequence_no)

            if not conversation:
                conversation.append(ChatMessage(role="system", content=self.starter_system_prompt))

            conversation.append(ChatMessage(role="user", content=message))

            parsedFileData = await self.parse_files(chat_id, files)
            attachment_records = self._build_attachment_records(parsedFileData)

            now = datetime.utcnow()
            last_sequence_no += 1
            self.repo.create_message(
                SimpleNamespace(
                    id=0,
                    chatId=chat_id,
                    role="user",
                    message=message,
                    sequenceNo=last_sequence_no,
                    created_at=now,
                    attachments=json.dumps(attachment_records) if attachment_records else None,
                )
            )

            if parsedFileData and self._has_image(parsedFileData):
                assistant_msg = ChatMessage(role="assistant", content=IMAGE_UNAVAILABLE_MESSAGE)
                last_sequence_no += 1
                self.repo.create_message(
                    SimpleNamespace(
                        id=0,
                        chatId=chat_id,
                        role="assistant",
                        message=assistant_msg.content,
                        sequenceNo=last_sequence_no,
                        created_at=datetime.utcnow(),
                        attachments=None,
                    )
                )
                yield chat_id, assistant_msg
                return

            # _, model_name = self.cloud_models[1]
            agent = OllamaAgent(model_name=model_name, tools=self._fs_tools, system_prompt=self.starter_system_prompt)

            agent_conversation = self._build_enriched_conversation(conversation,
                                                                   parsedFileData) if parsedFileData else conversation

            last_msg = None
            for new_msg in agent.stream(agent_conversation):
                last_msg = new_msg
                yield chat_id, new_msg

            if last_msg is not None:
                last_sequence_no += 1
                self.repo.create_message(
                    SimpleNamespace(
                        id=0,
                        chatId=chat_id,
                        role=last_msg.role,
                        message=last_msg.content,
                        sequenceNo=last_sequence_no,
                        created_at=datetime.utcnow(),
                        attachments=None,
                    )
                )

        except Exception as e:
            print(f"Error: {e}")
            yield chat_id, None

    def getConversation(self, user_id):
        user_conversations = self.repo.get_user_conversations(user_id)

        return user_conversations

    def getLocalModelList(self) -> List[dict]:
        """Returns the list of available local models as {id, name, modelKey}."""
        try:
            return [
                {"id": model_id, "name": name, "modelKey": model_key}
                for model_id, (name, model_key) in self.local_models.items()
            ]
        except Exception as e:
            print(f"Error fetching local model list: {e}")
            return []

    def getCloudModelList(self) -> List[dict]:
        """Returns the list of available cloud models as {id, name, modelKey}."""
        try:
            return [
                {"id": model_id, "name": name, "modelKey": model_key}
                for model_id, (name, model_key) in self.cloud_models.items()
            ]
        except Exception as e:
            print(f"Error fetching cloud model list: {e}")
            return []

    def getMessagesForChat(self, chat_id: int) -> List[ChatMessage]:
        """Fetches all messages for a given chat_id, ordered by sequence_no."""
        try:
            rows = self.repo.get_messages(chat_id)
            return [
                ChatMessage(role=row[2], content=row[3])
                for row in rows
            ]
        except Exception as e:
            print(f"Error fetching messages for chat {chat_id}: {e}")
            return []
