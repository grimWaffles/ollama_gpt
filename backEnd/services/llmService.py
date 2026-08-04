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

    def _build_enriched_conversation(self, conversation, parsedFileData):
        agent_conversation = [{"role": m.role, "content": m.content} for m in conversation]
        extra_text = ""
        for entry in parsedFileData:
            ...
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

    def _retrieve_context(self, chat_id: int, user_id: int, query: str) -> Dict[str, list]:
        query_embedding = self.embedding_service.embed_query(query)

        memory_k = getattr(self.app_config, "memory_k", 6)
        summary_k = getattr(self.app_config, "summary_k", 3)
        kb_k = getattr(self.app_config, "kb_k", 4)

        memory_hits = self.vector_repo.similarity_search_messages(
            user_id=user_id, query_embedding=query_embedding, chat_id=chat_id, k=memory_k,
        )
        summary_hits = self.vector_repo.similarity_search_summaries(
            chat_id=chat_id, query_embedding=query_embedding, k=summary_k,
        )
        kb_hits = self.vector_repo.similarity_search(chat_id, query_embedding, k=kb_k)

        # Summaries are curated to already hold the categories we want to
        # prioritize (preferences, decisions, unresolved tasks), so give them a
        # small edge over raw individual turns when ranking.
        summary_boost = getattr(self.app_config, "summary_relevance_boost", 0.85)
        for hit in summary_hits:
            hit["distance"] = hit.get("distance", 1.0) * summary_boost
            hit["is_summary"] = True

        combined_memory = self._rank_and_filter(
            memory_hits + summary_hits,
            max_distance=getattr(self.app_config, "memory_relevance_threshold", None),
            max_age_days=getattr(self.app_config, "memory_recency_days", None),
            limit=getattr(self.app_config, "memory_result_limit", 8),
        )
        filtered_kb = self._rank_and_filter(
            kb_hits,
            max_distance=getattr(self.app_config, "kb_relevance_threshold", None),
            limit=getattr(self.app_config, "kb_result_limit", 4),
        )

        return {"memories": combined_memory, "knowledge": filtered_kb}

    def _rank_and_filter(self, results: list, max_distance: Optional[float] = None,
                         max_age_days: Optional[float] = None, limit: int = 8) -> list:
        seen_content = set()
        deduped = []

        now = datetime.now()

        for r in sorted(results, key=lambda r: r.get("distance", 1.0)):
            content_key = (r.get("content") or "").strip()
            if not content_key or content_key in seen_content:
                continue

            if max_distance is not None and r.get("distance", 1.0) > max_distance:
                continue

            # Recency threshold only applies to raw memories with a created_at;
            # summaries represent long-term memory and are exempt by design —
            # they shouldn't age out just because the exchange they cover is old.
            if max_age_days is not None and not r.get("is_summary") and r.get("created_at"):
                age_days = (now - r["created_at"]).total_seconds() / 86400
                if age_days > max_age_days:
                    continue

            seen_content.add(content_key)
            deduped.append(r)
            if len(deduped) >= limit:
                break

        return deduped

    def _compose_user_content(self, message: str, parsedFileData: List[Dict[str, Any]]) -> str:
        if not parsedFileData:
            return message

        inline_limit = getattr(self.app_config, "inline_text_char_limit", 6000)
        extra_text = ""
        for entry in parsedFileData:
            text = entry.get("text")
            if not text:
                continue
            if len(text) <= inline_limit:
                extra_text += f"\n\n---\nFile: {entry['filename']}\n{text}"
            else:
                extra_text += (
                    f"\n\n---\nFile: {entry['filename']} ({len(text)} characters — "
                    f"too large to inline). This file has been indexed; use "
                    f"search_knowledge_base with a specific query to retrieve from it."
                )
        return message + extra_text

    def _build_compact_prompt(self, message: str, retrieved: Dict[str, list],
                              parsedFileData: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        system_content = self.starter_system_prompt
        context_sections = []

        if retrieved["memories"]:
            memory_block = "\n\n".join(
                f"[{'summary' if r.get('is_summary') else r.get('role', 'memory')}]\n{r['content']}"
                for r in retrieved["memories"]
            )
            context_sections.append(f"### Relevant memory from this conversation\n{memory_block}")

        if retrieved["knowledge"]:
            knowledge_block = "\n\n".join(
                f"[Source: {r['filename']}]\n{r['content']}"
                for r in retrieved["knowledge"]
            )
            context_sections.append(f"### Relevant knowledge from uploaded documents\n{knowledge_block}")

        if context_sections:
            system_content += "\n\n" + "\n\n".join(context_sections)

        return [
            {"role": "system", "content": system_content},
            {"role": "user", "content": self._compose_user_content(message, parsedFileData)},
        ]

    async def _maybe_summarize_old_messages(self, chat_id: int, user_id: int) -> None:
        cluster_size = getattr(self.app_config, "summary_cluster_size", 20)
        keep_recent = getattr(self.app_config, "summary_keep_recent", 6)

        unarchived_count = self.convo_repo.get_unarchived_message_count(chat_id)
        if unarchived_count < cluster_size + keep_recent:
            return

        to_summarize = self.convo_repo.get_oldest_unarchived_messages(chat_id, cluster_size)
        if not to_summarize:
            return

        transcript = "\n".join(f"{role}: {text}" for (_id, role, text, _seq) in to_summarize)
        summary_text = await self._generate_summary(transcript)

        start_seq = to_summarize[0][3]
        end_seq = to_summarize[-1][3]
        self.embedding_service.ingest_summary(chat_id, user_id, summary_text, start_seq, end_seq)

        self.convo_repo.archive_messages([row[0] for row in to_summarize])

    async def _generate_summary(self, transcript: str) -> str:
        summarizer_prompt = (
            "Compress the following conversation excerpt into dense, durable factual "
            "notes: user preferences, decisions made, unresolved tasks, and any key "
            "facts worth remembering. Bullet points, no filler, no meta-commentary.\n\n"
            f"{transcript}"
        )
        summarizer_model = getattr(self.app_config, "summary_model_name", None) or self.cloud_models[1][1]
        summarizer = OllamaAgent(
            model_name=summarizer_model,
            tools=[],
            system_prompt="You compress conversation excerpts into compact factual memory notes.",
        )
        result = await summarizer.ainvoke([{"role": "user", "content": summarizer_prompt}])
        return result[-1].content if result else transcript[:800]

    def _build_attachment_records(self, parsedFileData: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return [
            {
                "filename": entry["filename"],
                "path": entry["path"],
                "content_type": entry["content_type"],
                "kind": entry["kind"],
                "document_id": entry.get("document_id"),
            }
            for entry in parsedFileData
        ]

    async def chat_with_llm_2(self, user_id: int, chat_id: int, model_name: str,
                              message: str, files: Optional[List[UploadFile]] = None) -> tuple[int, List[ChatMessage]]:
        files = files or []
        try:
            is_new_conversation = not chat_id or chat_id == 0

            if is_new_conversation:
                max_chat_id = self.convo_repo.get_max_chat_id()
                chat_id = self.convo_repo.create_conversation(
                    SimpleNamespace(
                        chatId=chat_id,
                        userId=user_id,
                        chatName=f"Conversation #{max_chat_id + 1}",
                        created_at=datetime.now(),
                    )
                )

            last_sequence_no = self.convo_repo.get_last_sequence_no(chat_id)

            # --- Files: ingest into the knowledge base regardless of turn number ---
            parsedFileData = await self._parse_files(chat_id, files)
            attachment_records = None
            if parsedFileData:
                self.embedding_service.ingest_documents(chat_id, user_id, parsedFileData)
                attachment_records = self._build_attachment_records(parsedFileData)

            # --- 1. Persist the user's message first, then embed it ---
            now = datetime.now()
            last_sequence_no += 1
            user_message_id = self.convo_repo.create_message(
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


            if self.app_config.use_embedding:
                self.embedding_service.ingest_message(user_message_id, chat_id, user_id, "user", message)

            # --- Image-attachment short-circuit (unchanged behavior) ---
            if parsedFileData and _has_image(parsedFileData):
                assistant_msg = ChatMessage(role="assistant", content=self.app_config.image_unavailable_message)
                last_sequence_no += 1
                self.convo_repo.create_message(
                    SimpleNamespace(
                        id=0, chatId=chat_id, role="assistant", message=assistant_msg.content,
                        sequenceNo=last_sequence_no, created_at=datetime.now(), attachments=None,
                    )
                )
                return chat_id, [assistant_msg]

            # --- 2. Build the prompt: full context only on turn 1, retrieved context after ---
            if is_new_conversation:
                agent_conversation = [
                    {"role": "system", "content": self.starter_system_prompt},
                    {"role": "user", "content": self._compose_user_content(message, parsedFileData)},
                ]
            else:
                retrieved = self._retrieve_context(chat_id, user_id, message)
                agent_conversation = self._build_compact_prompt(message, retrieved, parsedFileData)

            # --- 3. Tools (unchanged) ---
            embedding_tools = []
            if self.app_config.use_embedding:
                kb_tool = KnowledgeBaseSearchTool(
                    vector_repo=self.vector_repo, embedding_service=self.embedding_service, chat_id=chat_id,
                )
                chat_history_tool = ChatHistorySearchTool(
                    vector_repo=self.vector_repo, embedding_service=self.embedding_service,
                    chat_id=chat_id, user_id=user_id,
                )
                embedding_tools = [kb_tool, chat_history_tool]

            mcp_tools = []

            agent = OllamaAgent(
                model_name=model_name,
                tools=self._fs_tools + mcp_tools + embedding_tools,
                system_prompt=self.starter_system_prompt,
            )

            # --- 4. Invoke with the compact prompt, not raw history ---
            full_response = await agent.ainvoke(agent_conversation)
            new_messages = full_response[len(agent_conversation):]

            # --- 5. Persist + embed assistant output ---
            for new_msg in new_messages:
                last_sequence_no += 1
                new_msg_id = self.convo_repo.create_message(
                    SimpleNamespace(
                        id=0, chatId=chat_id, role=new_msg.role, message=new_msg.content,
                        sequenceNo=last_sequence_no, created_at=datetime.now(), attachments=None,
                    )
                )
                if self.app_config.use_embedding and new_msg.role not in ("tool",):
                    self.embedding_service.ingest_message(new_msg_id, chat_id, user_id, new_msg.role, new_msg.content)

            # --- 6. Periodic summarization/archival so prompt size stays bounded ---
            if self.app_config.use_embedding:
                await self._maybe_summarize_old_messages(chat_id, user_id)

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

    def clear_all_history(self)-> bool:
        return self.convo_repo.clear_history()
