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

import asyncio
import re

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

    async def _get_verified_response(
            self,
            agent_conversation: List[Dict[str, str]],
            message: str,
            embedding_tools: list,
    ) -> tuple[ChatMessage, Dict[str, Any]]:
        """
        Multi-model verification pipeline:
          1. Two independent models answer the same prompt in parallel, each
             required to give sources/confidence/reasoning alongside the answer.
          2. A third model receives both responses and is instructed to verify
             them, surface disagreements, and produce a single resolved answer.
          3. If the query looks high-stakes/factual, external search results are
             gathered first and handed to the verifier as ground truth — because
             two LLMs agreeing is still just LLM opinion, not evidence.

        Returns (final_message, debate_metadata) — metadata is for logging/
        debugging only and is not sent to the user.
        """
        cloud_models = self.llm_model_service.get_cloud_models()
        _, model_a_name = cloud_models[1]  # GPT-OSS
        _, model_b_name = cloud_models[5]  # Gemma 4 31B
        _, judge_name = cloud_models[6]  # Minimax-m3 as the verifier

        debate_instructions = (
            "Answer the user's question fully. Then, in addition to the answer, "
            "give:\n"
            "SOURCES: what sources, documents, or general knowledge you are relying on.\n"
            "CONFIDENCE: low / medium / high, and why.\n"
            "REASONING: a short explanation of how you reached the answer.\n"
            "Structure your reply with these labeled sections: ANSWER, SOURCES, "
            "CONFIDENCE, REASONING."
        )

        def _with_debate_instructions(base: List[Dict[str, str]]) -> List[Dict[str, str]]:
            variant = [dict(m) for m in base]
            if variant and variant[0]["role"] == "system":
                variant[0] = {**variant[0], "content": variant[0]["content"] + "\n\n" + debate_instructions}
            else:
                variant.insert(0, {"role": "system", "content": debate_instructions})
            return variant

        conversation_a = _with_debate_instructions(agent_conversation)
        conversation_b = _with_debate_instructions(agent_conversation)

        agent_a = OllamaAgent(model_name=model_a_name, tools=embedding_tools, system_prompt=self.starter_system_prompt)
        agent_b = OllamaAgent(model_name=model_b_name, tools=embedding_tools, system_prompt=self.starter_system_prompt)

        # --- Step 1: independent, parallel first-pass answers ---
        response_a, response_b = await asyncio.gather(
            agent_a.ainvoke(conversation_a),
            agent_b.ainvoke(conversation_b),
        )
        answer_a = response_a[-1].content if response_a else "(model A produced no response)"
        answer_b = response_b[-1].content if response_b else "(model B produced no response)"

        # --- Step 2: external evidence for high-stakes factual queries (optional) ---
        external_context = ""
        if self._is_high_stakes_query(message):
            external_context = await self._gather_external_evidence(message)

        # --- Step 3: third model verifies, resolves conflicts, gives final answer ---
        if external_context:
            evidence_instruction = (
                "3. Where external evidence is provided below, treat it as ground truth "
                "and prefer claims it supports over either model's unsupported claims.\n"
                "4. Where no external evidence is available, resolve disagreements by "
                "weighing stated confidence and reasoning quality — do not average or "
                "split the difference between conflicting facts.\n"
            )
        else:
            evidence_instruction = (
                "3. No external evidence was retrieved for this query — resolve any "
                "disagreements by weighing each model's stated confidence and reasoning "
                "quality. Do not average or split the difference between conflicting "
                "facts, and do not present unverified claims as more certain than they are.\n"
            )

        verifier_prompt = (
                "You are a verification judge reviewing two independent AI answers to "
                "the same question. Each answer includes its own sources, confidence, "
                "and reasoning.\n\n"
                "Your job:\n"
                "1. Identify where the two answers agree and where they disagree.\n"
                "2. Cross-check specific facts, numbers, and claims between the two.\n"
                + evidence_instruction +
                "5. Output the FINAL ANSWER ONLY, after conflicts are resolved. Do not "
                "mention the models, the debate process, or unresolved disagreements "
                "in your output — write directly to the end user as a normal answer.\n\n"
                f"User's question:\n{message}\n\n"
                f"--- Response A ---\n{answer_a}\n\n"
                f"--- Response B ---\n{answer_b}\n\n"
                + (
                    f"--- External evidence (treat as ground truth) ---\n{external_context}\n\n" if external_context else "")
        )

        judge_agent = OllamaAgent(
            model_name=judge_name,
            tools=[],
            system_prompt="You are a careful, evidence-driven fact-checking judge. You resolve disagreements between two AI answers and output only the final, correct answer.",
        )
        judge_response = await judge_agent.ainvoke([{"role": "user", "content": verifier_prompt}])
        final_content = judge_response[-1].content if judge_response else (answer_a or answer_b)

        final_message = ChatMessage(role="assistant", content=final_content)
        debate_metadata = {
            "model_a": model_a_name,
            "model_b": model_b_name,
            "judge_model": judge_name,
            "answer_a": answer_a,
            "answer_b": answer_b,
            "used_external_evidence": bool(external_context),
        }
        return final_message, debate_metadata

    def _is_high_stakes_query(self, message: str) -> bool:
        """
        Cheap heuristic gate for "is this worth a real search, not just LLM
        cross-checking". Errs toward searching when unsure — a wasted search is
        far cheaper than a confidently-wrong fact getting through two LLMs that
        happen to agree.
        """
        triggers = [
            r"\bhow many\b", r"\bhow much\b", r"\bwhen (did|was|is|will)\b",
            r"\bwho is\b", r"\bwho was\b", r"\bcurrent(ly)?\b", r"\blatest\b",
            r"\btoday'?s\b", r"\bthis year\b", r"\bstatistic", r"\bpercent",
            r"\bprice\b", r"\bdate\b", r"\bversion\b", r"\brelease", r"\bnews\b",
            r"\d{4}",  # a bare year
        ]
        text = message.lower()
        return any(re.search(t, text) for t in triggers)

    async def _gather_external_evidence(self, message: str) -> str:
        """
        Runs an external web search / retrieval pass so verification has
        something outside the two models to check claims against. Optional:
        controlled by app_config.use_external_search, and safely no-ops if the
        search tool isn't configured. Never blocks the response on failure.
        """
        if not getattr(self.app_config, "USE_EXTERNAL_WEBSEARCH", False):
            return ""

        web_search_tool = getattr(self, "web_search_tool", None)
        if web_search_tool is None:
            print("External search enabled in config but no web_search_tool is configured — skipping.")
            return ""

        try:
            results = await web_search_tool.asearch(message, max_results=5)
        except Exception as e:
            print(f"External evidence retrieval failed: {e}")
            return ""

        if not results:
            return ""

        return "\n\n".join(
            f"[{r.get('title', 'source')}] {r.get('snippet', '')} ({r.get('url', '')})"
            for r in results
        )

    async def chat_with_llm_2(self, user_id: int, chat_id: int, model_name: str,
                              message: str, extended_thinking: bool,files: Optional[List[UploadFile]] = None) -> tuple[int, List[ChatMessage]]:
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
            new_messages = ""

            if not extended_thinking:
                # --- 4.a Invoke with the compact prompt, not raw history ---
                agent = OllamaAgent(
                    model_name=model_name,
                    tools=self._fs_tools + mcp_tools + embedding_tools,
                    system_prompt=self.starter_system_prompt,
                )

                full_response = await agent.ainvoke(agent_conversation)
                new_messages = full_response[len(agent_conversation):]

            else:
                # --- 4.b Dual-model debate + third-model verification, instead of a single agent.ainvoke ---
                final_message, debate_metadata = await self._get_verified_response(
                    agent_conversation=agent_conversation,
                    message=message,
                    embedding_tools=self._fs_tools + embedding_tools,
                )
                new_messages = [final_message]

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
