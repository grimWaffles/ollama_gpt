# services/llm_service.py
import base64
import json
import traceback
import uuid
from datetime import datetime
from types import SimpleNamespace
from typing import List, Dict, Any, Optional
from fastapi import UploadFile
from agents.ollama_agent import OllamaAgent
from repository.vector_repository import VectorRepository
from services.embedding_service import EmbeddingService
from services.env_service import EnvService
from services.llm_model_service import LlmModelService
from services.system_tool_service import SystemToolService
from tools.search_chat_history_tool import ChatHistorySearchTool
from models.chat_models import ChatMessage
from repository.conversation_repository import ConversationRepository
from tools.knowledge_base_tool import KnowledgeBaseSearchTool

def _has_image(parsedFileData: List[Dict[str, Any]]) -> bool:
    return any(entry["kind"] == "image" for entry in parsedFileData)

class LlmService:
    def __init__(self, ) -> None:
        # load app config
        self.app_config = EnvService().get_app_config()

        # load repositories
        self.convo_repo = ConversationRepository()
        self.vector_repo = VectorRepository()  # reuse your existing connection string

        # load services
        self.embedding_service = EmbeddingService(self.vector_repo, self.app_config)

        # load models and starter prompt
        self.llm_model_service = LlmModelService()
        self.local_models = self.llm_model_service.get_local_models()
        self.cloud_models = self.llm_model_service.get_cloud_models()
        self.starter_system_prompt = self.llm_model_service.get_system_prompt()

        # load system tools
        self._fs_tools = SystemToolService().get_system_tools()

    async def _parse_files(self, chat_id: int, files: List[UploadFile]) -> List[Dict[str, Any]]:
        """
        Saves each uploaded file to disk under uploads/{chat_id}/, then parses
        it by type. Returns per-file dicts carrying both the on-disk path
        (for DB persistence / later reparsing) and the extracted content
        (for use in *this* turn's agent call only — never persisted).
        """
        parsed: List[Dict[str, Any]] = []
        chat_dir = self.app_config.upload_root_dir / str(chat_id)
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
                if ext in self.app_config.text_extensions:
                    entry["kind"] = "text"
                    entry["text"] = raw_bytes.decode("utf-8", errors="replace")

                elif ext in self.app_config.pdf_extensions:
                    entry["kind"] = "pdf"
                    try:
                        from pypdf import PdfReader
                        import io
                        reader = PdfReader(io.BytesIO(raw_bytes))
                        entry["text"] = "\n".join(page.extract_text() or "" for page in reader.pages)
                    except Exception:
                        entry["text"] = None

                elif ext in self.app_config.docx_extensions:
                    entry["kind"] = "docx"
                    try:
                        import docx
                        import io
                        doc = docx.Document(io.BytesIO(raw_bytes))
                        entry["text"] = "\n".join(p.text for p in doc.paragraphs)
                    except Exception:
                        entry["text"] = None

                elif ext in self.app_config.image_extensions:
                    entry["kind"] = "image"
                    entry["base64_data"] = base64.b64encode(raw_bytes).decode("utf-8")

                else:
                    entry["kind"] = "unsupported"

            except Exception:
                entry["kind"] = "error"
                entry["text"] = None

            parsed.append(entry)

        return parsed

    def _build_enriched_conversation(self, conversation: List[ChatMessage], parsedFileData: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Returns a NEW list of plain dicts for the agent call only — `conversation`
        (which gets persisted) is left untouched.

        Small files (<= INLINE_TEXT_CHAR_LIMIT): full extracted text is appended
        inline, same as before.

        Large files: only a short notice + filename is appended, since the full
        text was already chunked and embedded by _ingest_documents. The agent is
        expected to call search_knowledge_base for these.
        """
        agent_conversation = [{"role": m.role, "content": m.content} for m in conversation]

        extra_text = ""
        for entry in parsedFileData:
            text = entry.get("text")
            if not text:
                continue

            if len(text) <= self.app_config.inline_text_char_limit:
                extra_text += f"\n\n---\nFile: {entry['filename']}\n{text}"
            else:
                extra_text += (
                    f"\n\n---\nFile: {entry['filename']} "
                    f"({len(text)} characters — too large to inline in full). "
                    f"This file has been indexed; use the search_knowledge_base tool "
                    f"with a specific query to retrieve relevant passages from it."
                )

        if extra_text and agent_conversation:
            agent_conversation[-1]["content"] += extra_text

        return agent_conversation

    async def chat_with_llm(self, user_id: int, chat_id: int, model_name: str, message: str, files: Optional[List[UploadFile]] = None) -> tuple[int, List[ChatMessage]]:
        files = files or []
        conversation: List[ChatMessage] = []
        try:
            if not chat_id or chat_id == 0:
                max_chat_id = self.convo_repo.get_max_chat_id()
                chat_id = self.convo_repo.create_conversation(
                    SimpleNamespace(
                        chatId=chat_id,
                        userId=user_id,
                        chatName=f"Conversation #{max_chat_id + 1}",
                        created_at=datetime.now(),
                    )
                )

            rows = self.convo_repo.get_messages(chat_id)

            if len(rows) > self.app_config.max_inline_messages:
                rows = rows[:1] + rows[-(self.app_config.max_inline_messages - 1):]  # keep system prompt + recent tail

            last_sequence_no = 0

            for row in rows:
                _, _, role, msg_text, _, sequence_no, _ = row
                if role == "tool":  # NEW — skip persisted tool messages on reload
                    last_sequence_no = max(last_sequence_no, sequence_no)
                    continue
                conversation.append(ChatMessage(role=role, content=msg_text))
                last_sequence_no = max(last_sequence_no, sequence_no)

            if not conversation:
                conversation.append(ChatMessage(role="system", content=self.starter_system_prompt))

            conversation.append(ChatMessage(role="user", content=message))

            attachment_records = []

            parsedFileData = await self._parse_files(chat_id, files)
            if parsedFileData:
                self.embedding_service.ingest_documents(chat_id, user_id, parsedFileData)  # NEW
                attachment_records = self._build_enriched_conversation(conversation, parsedFileData)

            now = datetime.now()
            last_sequence_no += 1
            message_id = self.convo_repo.create_message(
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
            if self.app_config.use_embedding: self.embedding_service.ingest_message(message_id, chat_id, user_id, "user", message)  # NEW

            if parsedFileData and _has_image(parsedFileData):
                currentRole = "assistant"
                assistant_msg = ChatMessage(role=currentRole, content=self.app_config.image_unavailable_message)
                last_sequence_no += 1
                message_id = self.convo_repo.create_message(
                    SimpleNamespace(
                        id=0,
                        chatId=chat_id,
                        role=currentRole,
                        message=assistant_msg.content,
                        sequenceNo=last_sequence_no,
                        created_at=datetime.now(),
                        attachments=None,
                    )
                )
                if self.app_config.use_embedding: self.embedding_service.ingest_message(message_id, chat_id, user_id, currentRole, message)  # NEW

                return chat_id, [assistant_msg]

            embedding_tools = []

            if self.app_config.use_embedding:
                kb_tool = KnowledgeBaseSearchTool(
                    vector_repo=self.vector_repo,
                    embedding_service=self.embedding_service,
                    chat_id=chat_id,
                )

                chat_history_tool = ChatHistorySearchTool(
                    vector_repo=self.vector_repo,
                    embedding_service=self.embedding_service,
                    chat_id=chat_id,
                    user_id=user_id,
                )

                embedding_tools = [kb_tool, chat_history_tool]

            mcp_tools = []

            # try:
            #     mcp_tools = await self.mcp_tool_client.get_tools()
            # except Exception as e:
            #     print("Failed to get tools: " + str(e))

            agent = OllamaAgent(
                model_name=model_name,
                tools=self._fs_tools + mcp_tools + embedding_tools if self.app_config.use_embedding else self._fs_tools+mcp_tools,
                system_prompt=self.starter_system_prompt,
            )

            agent_conversation = self._build_enriched_conversation(conversation, parsedFileData) if parsedFileData else conversation

            full_response = await agent.ainvoke(agent_conversation)
            new_messages = full_response[len(agent_conversation):]

            for new_msg in new_messages:
                last_sequence_no += 1
                message_id = self.convo_repo.create_message(
                    SimpleNamespace(
                        id=0,
                        chatId=chat_id,
                        role=new_msg.role,
                        message=new_msg.content,
                        sequenceNo=last_sequence_no,
                        created_at=datetime.now(),
                        attachments=None,
                    )
                )

                if self.app_config.use_embedding: self.embedding_service.ingest_message(message_id, chat_id, user_id, "user", message)  # NEW

            return chat_id, new_messages
        except Exception as e:
            print(f"Error: {e}")
            print(traceback.format_exc())
            return chat_id, []

    def get_conversation(self, user_id):
        user_conversations = self.convo_repo.get_user_conversations(user_id)

        return user_conversations

    def get_local_model_list(self) -> List[dict]:
        """Returns the list of available local models as {id, name, modelKey}."""
        try:
            return [
                {"id": model_id, "name": name, "modelKey": model_key}
                for model_id, (name, model_key) in self.local_models.items()
            ]
        except Exception as e:
            print(f"Error fetching local model list: {e}")
            return []

    def get_cloud_model_list(self) -> List[dict]:
        """Returns the list of available cloud models as {id, name, modelKey}."""
        try:
            return [
                {"id": model_id, "name": name, "modelKey": model_key}
                for model_id, (name, model_key) in self.cloud_models.items()
            ]
        except Exception as e:
            print(f"Error fetching cloud model list: {e}")
            return []

    def get_messages_for_chat(self, chat_id: int) -> List[ChatMessage]:
        """Fetches all messages for a given chat_id, ordered by sequence_no."""
        try:
            rows = self.convo_repo.get_messages(chat_id)
            return [
                ChatMessage(role=row[2], content=row[3])
                for row in rows
                if row[2] != "tool" and row[3] != ""
            ]
        except Exception as e:
            print(f"Error fetching messages for chat {chat_id}: {e}")
            return []
