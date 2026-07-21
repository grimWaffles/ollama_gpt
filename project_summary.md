# Streaming Chat Feature — Summary

## Backend (Python / FastAPI / LangChain)

### `agents/ollama_agent.py`
- Fixed `extract_text` — was missing `self`, causing a `TypeError` at runtime.
- Added `stream()` method: streams the LangChain agent via `stream_mode="values"`, tracks `seen` count (initialized to `len(messages)` to avoid re-emitting the input conversation) and yields only new `ChatMessage` objects as they appear.

### `LlmService`
- Added `chatWithLlmStream(user_id, chat_id, message)`: mirrors `chatWithLlm` (creates conversation if needed, loads history, saves user message) but is a generator that yields `(chat_id, ChatMessage)` tuples from `agent.stream(...)`, and persists the final assistant message to the DB after the loop completes.

### API
- Added `POST /chat/stream/` endpoint using `fastapi.responses.StreamingResponse`, emitting Server-Sent Events (`data: <json>\n\n`) built from `ChatResponse` payloads, one message at a time.
- Import needed: `from fastapi.responses import StreamingResponse`.

## Frontend (Angular)

### `chat.service.ts`
- Added `sendMessageStream(messageContent)`: uses `fetch` + `ReadableStream` to consume the SSE stream from `/chat/stream/`, parses each `data:` line, updates `messages` signal incrementally, and toggles `isThinking` off on the **first** received chunk (thinking loader now correctly turns off when streaming starts, not just on full completion).
- `ChatMessage` interface extended with optional `animate?: boolean` flag — set to `true` only on messages that arrive via the streaming path, so downstream components know which messages should be typed out character-by-character vs. rendered instantly.

### `home-component.ts` / `.html`
- Added `streamingEnabled` signal + `onToggleStreaming()`; added a Stream toggle button in the UI beside the model selector/send button.
- `submitMessage()` and `retryMessage()` branch on `streamingEnabled()` to call either `sendMessage()` or `sendMessageStream()`.
- Added `copyMessage()` (clipboard) and `retryMessage(index)` (finds the preceding user message and resubmits it).
- Refactored chat bubble markup into a standalone `ChatMessageComponent`; parent now loops with `<app-chat-message [msg]="msg" (copy)="copyMessage($event)" (retry)="retryMessage($index)">`.

### `chat-message.component.ts` (new)
- Renders user messages as plain text, assistant messages via `ngx-markdown`.
- Includes Copy and Retry action buttons under assistant messages.
- Added typewriter effect: an `effect()` watches `msg().content`, and if `msg().animate` is true (and role isn't user), types out only the new tail of text via `setInterval` (12ms/char) into a `displayedContent` signal, which is what's actually bound to the markdown view. Messages loaded from history (`loadChatMessages`) or from the non-streaming endpoint do **not** animate — only messages flagged `animate: true` from `sendMessageStream` type out character-by-character.
- Cleans up the interval on `ngOnDestroy`.

## Known open items / not yet implemented
- No typing-speed toggle exists yet — speed is hardcoded (`12` ms/char) in `chat-message.component.ts`. Was discussed but not built.

---

## Full File Contents

### `agents/ollama_agent.py`

```python
import traceback
from pathlib import Path
from typing import Any, List, Optional

from dotenv import load_dotenv
from langchain.agents import create_agent

from models.chat_models import ChatMessage

# Load ../.env relative to the Agents folder
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

# LangChain's internal message ".type" values -> our app's role strings
_ROLE_MAP = {
    "human": "user",
    "ai": "assistant",
    "system": "system",
    "tool": "tool",
}


class OllamaAgent:
    def __init__(
            self,
            model_name: str,
            tools: Optional[List[Any]] = None,
            system_prompt: str = "",
    ):
        self.agent = create_agent(
            model=model_name,
            tools=tools or [],
            system_prompt=system_prompt,
        )

    def extract_text(self, content: Any) -> str:
        """
        LangChain message .content can be a plain string OR a list of content
        blocks (e.g. when tool calls / tool results are involved). Normalize
        to a single string so it fits our ChatMessage(content: str) model.
        """
        if isinstance(content, str):
            return content

        if isinstance(content, list):
            parts = []
            for block in content:
                if isinstance(block, dict):
                    if "text" in block:
                        parts.append(block["text"])
                    else:
                        parts.append(str(block))
                else:
                    parts.append(str(block))
            return "".join(parts)

        return str(content)

    def invoke(self, conversation: List[ChatMessage]) -> List[ChatMessage]:
        """
        Runs the agent on the full conversation and returns the FULL set of
        messages produced by the graph (including the messages that were
        passed in). Callers that only want the newest reply should slice
        the result themselves (see LlmService.chatWithLlm).
        """
        messages = [
            {
                "role": msg.role,
                "content": msg.content,
            }
            for msg in conversation
        ]

        try:
            result = self.agent.invoke({"messages": messages})
            returned_messages: List[ChatMessage] = []

            for message in result["messages"]:
                content = self.extract_text(getattr(message, "content", None))

                msg_type = getattr(message, "type", None)
                if msg_type in _ROLE_MAP:
                    role = _ROLE_MAP[msg_type]
                else:
                    role = getattr(message, "role", None) or type(message).__name__.replace("Message", "").lower()

                returned_messages.append(
                    ChatMessage(
                        role=role,
                        content=content,
                    )
                )

            return returned_messages

        except Exception:
            print(f"Error: {traceback.format_exc()}")
            return []

    def stream(self, conversation: List[ChatMessage]):
        """
        Streams agent output. Yields ChatMessage objects as they are produced.
        """
        messages = [
            {
                "role": msg.role,
                "content": msg.content,
            }
            for msg in conversation
        ]

        seen = len(messages)

        try:
            for chunk in self.agent.stream({"messages": messages}, stream_mode="values"):
                msgs = chunk["messages"]
                for message in msgs[seen:]:
                    content = self.extract_text(getattr(message, "content", None))

                    msg_type = getattr(message, "type", None)
                    if msg_type in _ROLE_MAP:
                        role = _ROLE_MAP[msg_type]
                    else:
                        role = getattr(message, "role", None) or type(message).__name__.replace("Message", "").lower()

                    yield ChatMessage(role=role, content=content)
                seen = len(msgs)

        except Exception:
            print(f"Error: {traceback.format_exc()}")
            return
```

### `LlmService` (relevant methods)

```python
def chatWithLlm(self, user_id: int, chat_id: int, message: str) -> tuple[int, List[ChatMessage]]:
    conversation: List[ChatMessage] = []
    try:
        # --- New chat: assign the next available chat_id ---
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

        # --- Fetch existing conversation from DB ---
        rows = self.repo.get_messages(chat_id)

        last_sequence_no = 0
        for row in rows:
            _, _, role, msg_text, sequence_no, _ = row
            conversation.append(ChatMessage(role=role, content=msg_text))
            last_sequence_no = max(last_sequence_no, sequence_no)

        if not conversation:
            conversation.append(
                ChatMessage(
                    role="system",
                    content=self.starter_system_prompt,
                )
            )

        # --- Add the user's current message ---
        conversation.append(
            ChatMessage(
                role="user",
                content=message,
            )
        )

        # Select the desired model (example: GPT-5 Mini)
        _, model_name = self.cloud_models[1]

        agent = OllamaAgent(
            model_name=model_name,
            tools=[],
            system_prompt=self.starter_system_prompt,
        )

        full_response = agent.invoke(conversation)
        new_messages = full_response[len(conversation):]

        # --- Save user message ---
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
            )
        )

        # --- Save assistant response message(s) ---
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
                )
            )

        return chat_id, new_messages
    except Exception as e:
        print(f"Error: {e}")
        return chat_id, []


def chatWithLlmStream(self, user_id: int, chat_id: int, message: str):
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
            _, _, role, msg_text, sequence_no, _ = row
            conversation.append(ChatMessage(role=role, content=msg_text))
            last_sequence_no = max(last_sequence_no, sequence_no)

        if not conversation:
            conversation.append(
                ChatMessage(
                    role="system",
                    content=self.starter_system_prompt,
                )
            )

        conversation.append(
            ChatMessage(
                role="user",
                content=message,
            )
        )

        _, model_name = self.cloud_models[1]

        agent = OllamaAgent(
            model_name=model_name,
            tools=[],
            system_prompt=self.starter_system_prompt,
        )

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
            )
        )

        last_msg = None

        for new_msg in agent.stream(conversation):
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
                )
            )

    except Exception as e:
        print(f"Error: {e}")
        yield chat_id, None
```

### `main.py` (API routes, relevant excerpt)

```python
from fastapi.responses import StreamingResponse

@app.post("/chat/", response_model=ChatResponse)
async def say_hello(request: ChatRequest, llmService: LlmService = Depends(getLlmService)):
    chat_id, messages = llmService.chatWithLlm(request.userId, request.chatId, request.message)
    return ChatResponse(userId=request.userId, chatId=chat_id, messages=messages)


@app.post("/chat/stream/")
async def say_hello_stream(request: ChatRequest, llmService: LlmService = Depends(getLlmService)):
    def event_generator():
        for chat_id, msg in llmService.chatWithLlmStream(request.userId, request.chatId, request.message):
            if msg is None:
                continue
            payload = ChatResponse(
                userId=request.userId,
                chatId=chat_id,
                messages=[msg],
            )
            yield f"data: {payload.model_dump_json()}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")
```

### `chat.service.ts`

```typescript
import { inject, Injectable, signal } from '@angular/core';
import { HttpClient } from '@angular/common/http';

export interface ModelInfo { id: number; name: string; modelKey: string; badge?: string}
export interface ChatMessage { role: 'user' | 'assistant' | string; content: string; animate?: boolean; }
export interface ChatRequest { userId: number; chatId: number; message: string; }
export interface ChatResponse { userId: number; chatId: number; messages: ChatMessage[]; }
export interface ConversationEntity { chatId: number; chatName: string; }

@Injectable({ providedIn: 'root' })
export class ChatService {
  private http = inject(HttpClient);
  private apiUrl = 'http://localhost:8000';

  currentChatId = signal<number>(0);
  userId = signal<number>(1);
  messages = signal<ChatMessage[]>([]);
  conversations = signal<ConversationEntity[]>([]);
  localModels = signal<ModelInfo[]>([]);
  cloudModels = signal<ModelInfo[]>([]);
  selectedModelKey = signal<string>('');

  isThinking = signal<boolean>(false);
  thinkingStatus = signal<string>('Thinking...');

  private thinkingPhrases = [
    'Thinking...', 'Contemplating...', 'Analyzing variables...',
    'Formulating response...', 'Synthesizing knowledge...', 'Consulting neural map...'
  ];

  private startThinkingAnimation(): any {
    this.isThinking.set(true);
    this.thinkingStatus.set(this.thinkingPhrases[0]);
    let index = 1;
    return setInterval(() => {
      this.thinkingStatus.set(this.thinkingPhrases[index % this.thinkingPhrases.length]);
      index++;
    }, 1800);
  }

  loadConversations(): void {
    this.http.get<ConversationEntity[]>(`${this.apiUrl}/conversations/all?userId=${this.userId()}`)
      .subscribe(data => this.conversations.set(data));
  }

  loadModels(): void {
    this.http.get<ModelInfo[]>(`${this.apiUrl}/models/local`).subscribe(data => {
      this.localModels.set(data);
      if (data.length > 0 && !this.selectedModelKey()) this.selectedModelKey.set(data[0].modelKey);
    });
    this.http.get<ModelInfo[]>(`${this.apiUrl}/models/cloud`).subscribe(data => this.cloudModels.set(data));
  }

  loadChatMessages(chatId: number): void {
    this.currentChatId.set(chatId);
    this.http.get<ChatMessage[]>(`${this.apiUrl}/chat/${chatId}/messages`)
      .subscribe(data => this.messages.set(data));
  }

  /*
   * NON-STREAMING STANDARD HTTP POST PIPELINE
   */
  sendMessage(messageContent: string): void {
    const userMsg: ChatMessage = { role: 'user', content: messageContent };
    this.messages.update(prev => [...prev, userMsg]);

    const intervalId = this.startThinkingAnimation();

    const body: ChatRequest = {
      userId: this.userId(),
      chatId: this.currentChatId(),
      message: messageContent
    };

    this.http.post<ChatResponse>(`${this.apiUrl}/chat/`, body).subscribe({
      next: (res) => {
        clearInterval(intervalId);
        this.isThinking.set(false);

        this.currentChatId.set(res.chatId);
        this.messages.update(prev => [...prev, ...res.messages]);

        this.loadConversations();
      },
      error: (err) => {
        clearInterval(intervalId);
        this.isThinking.set(false);
        console.error('Failed to post message context:', err);
      }
    });
  }

  /*
   * STREAMING PIPELINE
   * Uses fetch + ReadableStream to consume Server-Sent Events from /chat/stream/
   */
  async sendMessageStream(messageContent: string): Promise<void> {
    const userMsg: ChatMessage = { role: 'user', content: messageContent };
    this.messages.update(prev => [...prev, userMsg]);

    const intervalId = this.startThinkingAnimation();

    const body: ChatRequest = {
      userId: this.userId(),
      chatId: this.currentChatId(),
      message: messageContent
    };

    try {
      const response = await fetch(`${this.apiUrl}/chat/stream/`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body)
      });

      if (!response.ok || !response.body) {
        throw new Error(`Request failed with status ${response.status}`);
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';
      let firstChunkReceived = false;

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n\n');
        buffer = lines.pop() ?? '';

        for (const line of lines) {
          if (!line.startsWith('data: ')) continue;
          const jsonStr = line.slice(6).trim();
          if (!jsonStr) continue;

          if (!firstChunkReceived) {
            firstChunkReceived = true;
            clearInterval(intervalId);
            this.isThinking.set(false);
          }

          const res: ChatResponse = JSON.parse(jsonStr);
          this.currentChatId.set(res.chatId);
          const animated = res.messages.map(m => ({ ...m, animate: true }));
          this.messages.update(prev => [...prev, ...animated]);
        }
      }

      clearInterval(intervalId);
      this.isThinking.set(false);
      this.loadConversations();
    } catch (err) {
      clearInterval(intervalId);
      this.isThinking.set(false);
      console.error('Failed to stream message context:', err);
    }
  }

  startNewChat(): void {
    this.currentChatId.set(0);
    this.messages.set([]);
  }
}
```

### `chat-message.component.ts`

```typescript
import {Component, input, output, ChangeDetectionStrategy, computed, signal, effect, OnDestroy} from '@angular/core';
import { CommonModule } from '@angular/common';
import {MarkdownComponent} from 'ngx-markdown';

export interface ChatMessage {
  role: 'user' | string;
  content: string;
  animate?: boolean;
}

@Component({
  selector: 'app-chat-message',
  standalone: true,
  imports: [CommonModule, MarkdownComponent],
  changeDetection: ChangeDetectionStrategy.OnPush,
  template: `
    <div
      class="flex w-full"
      [class.justify-end]="isUser()"
      [class.justify-start]="!isUser()"
    >
      <div
        class="max-w-[75%] rounded-3xl px-4 py-3 shadow-sm"
        [class.bg-indigo-600]="isUser()"
        [class.text-white]="isUser()"
        [class.rounded-br-md]="isUser()"
        [class.bg-white-80]="!isUser()"
        [class.text-neutral-800]="!isUser()"
        [class.rounded-bl-md]="!isUser()"
        [class.border]="!isUser()"
        [class.border-black-5]="!isUser()"
      >
        <div
          class="text-[10px] font-bold uppercase tracking-wider mb-1.5"
          [class.text-indigo-100]="isUser()"
          [class.text-neutral-400]="!isUser()"
        >
          {{ isUser() ? 'User Identity' : 'Sage Engine' }}
        </div>

        @if (msg().role === 'user') {
          <div
            class="whitespace-pre-wrap break-words text-base leading-relaxed">
            {{ msg().content }}
          </div>
        } @else {
          <markdown
            class="prose prose-neutral max-w-none"
            [data]="displayedContent()">
          </markdown>
        }

        @if (!isUser()) {
          <div class="flex items-center gap-2 mt-2 -mb-1">
            <button
              type="button"
              (click)="onCopy()"
              title="Copy"
              class="p-1.5 rounded-lg text-neutral-400 hover:text-indigo-600 hover:bg-black/5 transition-colors cursor-pointer"
            >
              <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <rect x="9" y="9" width="13" height="13" rx="2"></rect>
                <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path>
              </svg>
            </button>
            <button
              type="button"
              (click)="onRetry()"
              title="Retry"
              class="p-1.5 rounded-lg text-neutral-400 hover:text-indigo-600 hover:bg-black/5 transition-colors cursor-pointer"
            >
              <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <path d="M3 12a9 9 0 1 0 3-6.7"></path>
                <path d="M3 4v5h5"></path>
              </svg>
            </button>
          </div>
        }
      </div>
    </div>
  `,
})
export class ChatMessageComponent implements OnDestroy {
  msg = input.required<ChatMessage>();

  isUser = computed(() => this.msg().role === 'user');

  copy = output<string>();
  retry = output<void>();

  displayedContent = signal<string>('');
  private timerId: ReturnType<typeof setInterval> | null = null;

  constructor() {
    effect(() => {
      const fullText = this.msg().content;

      if (this.isUser() || !this.msg().animate) {
        this.displayedContent.set(fullText);
        return;
      }

      const current = this.displayedContent();
      const startFrom = fullText.startsWith(current) ? current.length : 0;
      if (startFrom === 0) this.displayedContent.set('');

      if (this.timerId) clearInterval(this.timerId);

      let i = startFrom;
      this.timerId = setInterval(() => {
        i++;
        this.displayedContent.set(fullText.slice(0, i));
        if (i >= fullText.length && this.timerId) {
          clearInterval(this.timerId);
          this.timerId = null;
        }
      }, 12);
    });
  }

  ngOnDestroy(): void {
    if (this.timerId) clearInterval(this.timerId);
  }

  onCopy(): void {
    this.copy.emit(this.msg().content);
  }

  onRetry(): void {
    this.retry.emit();
  }
}
```

### `home-component.ts`

```typescript
import {Component, inject, OnInit, signal, effect, viewChild, ElementRef, ViewChild, computed} from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { ChatService } from '../../chat.service';
import { ChatMessageComponent } from '../chat-message/chat-message.component';

@Component({
  selector: 'app-chat-interface',
  standalone: true,
  imports: [CommonModule, FormsModule, ChatMessageComponent],
  templateUrl: './home-component.html'
})
export class HomeComponent implements OnInit {
  protected chatService = inject(ChatService);
  userInput = signal<string>('');

  private scrollContainer = viewChild<ElementRef<HTMLDivElement>>('scrollFrame');

  constructor() {
    effect(() => {
      this.chatService.messages();
      this.chatService.isThinking();
      this.scrollToBottom();
    });
  }

  ngOnInit(): void {
    this.chatService.loadConversations();
    this.chatService.loadModels();
  }

  @ViewChild('scrollFrame') scrollFrame!: ElementRef;
  @ViewChild('chatListFrame') chatListFrame!: ElementRef;

  ngAfterViewChecked() {
    this.scrollFrame?.nativeElement.scrollTo({ top: this.scrollFrame.nativeElement.scrollHeight });
  }

  private scrollToBottom(): void {
    setTimeout(() => {
      const container = this.scrollContainer()?.nativeElement;
      if (container) {
        container.scrollTo({
          top: container.scrollHeight,
          behavior: 'smooth'
        });
      }
    }, 60);
  }

  onSelectConversation(id: number): void {
    this.chatService.loadChatMessages(id);
    this.chatListFrame?.nativeElement.scrollTo({ top: this.chatListFrame.nativeElement.scrollHeight });
  }

  onNewChat(): void {
    this.chatService.startNewChat();
  }

  onModelChange(model: { modelKey: string; name: string; id: string | number; badge?: string }) {
    this.chatService.selectedModelKey.set(model.modelKey);
  }

  streamingEnabled = signal<boolean>(false);

  onToggleStreaming(): void {
    this.streamingEnabled.update(v => !v);
  }

  submitMessage(): void {
    const text = this.userInput().trim();
    if (!text || this.chatService.isThinking()) return;

    if (this.streamingEnabled()) {
      this.chatService.sendMessageStream(text);
    } else {
      this.chatService.sendMessage(text);
    }
    this.userInput.set('');
  }

  handleEnterKey(event: Event): void {
    const keyboardEvent = event as KeyboardEvent;
    if (keyboardEvent.shiftKey) {
      return;
    }
    keyboardEvent.preventDefault();
    this.submitMessage();
  }

  isModelMenuOpen = signal(false);

  onToggleModelMenu() {
    this.isModelMenuOpen.update(v => !v);
  }

  selectedModelName = computed(() => {
    const key = this.chatService.selectedModelKey();
    const all = [...this.chatService.localModels(), ...this.chatService.cloudModels()];
    return all.find(m => m.modelKey === key)?.name ?? '';
  });

  onTriggerSettings() { console.log('Settings triggered'); }
  onTriggerProjects() { console.log('Projects triggered'); }
  onTriggerReset() { this.chatService.startNewChat(); }
  onTriggerAbout() { console.log('About context window opened'); }

  copyMessage(content: string): void {
    navigator.clipboard.writeText(content);
  }

  retryMessage(index: number): void {
    if (this.chatService.isThinking()) return;
    const messages = this.chatService.messages();
    let userContent = '';
    for (let i = index - 1; i >= 0; i--) {
      if (messages[i].role === 'user') {
        userContent = messages[i].content;
        break;
      }
    }
    if (!userContent) return;

    if (this.streamingEnabled()) {
      this.chatService.sendMessageStream(userContent);
    } else {
      this.chatService.sendMessage(userContent);
    }
  }
}
```

### `home-component.html` (message loop + footer excerpt reflecting current state)

```html
<!-- Message Scrolling Viewport Area -->
<div #scrollFrame class="flex-1 overflow-y-auto p-6 space-y-4 scrollbar-thin bg-black/[0.015]">
  @for (msg of chatService.messages(); track $index) {
    <app-chat-message
      [msg]="msg"
      (copy)="copyMessage($event)"
      (retry)="retryMessage($index)">
    </app-chat-message>
  }

  @if (chatService.isThinking()) {
  <div class="flex w-full justify-start transition-all duration-200">
    <div
      class="max-w-xs rounded-3xl rounded-bl-md p-4 border border-black/5 bg-white/80 text-neutral-400 text-xs flex items-center gap-3 shadow-sm">
      <span class="flex h-2 w-2 relative">
        <span
          class="animate-ping absolute inline-flex h-full w-full rounded-full bg-indigo-400 opacity-75"></span>
        <span class="relative inline-flex rounded-full h-2 w-2 bg-indigo-500"></span>
      </span>
      <span class="italic font-mono tracking-wide">{{ chatService.thinkingStatus() }}</span>
    </div>
  </div>
  }
</div>

<!-- Footer form controls: streaming toggle placed beside send button -->
<div class="flex items-center gap-2.5 pl-2.5 border-l border-black/5 h-10 mb-0.5">

  <!-- Model selector dropdown (unchanged, omitted here for brevity — see model menu block) -->

  <!-- Streaming Toggle -->
  <button type="button" (click)="onToggleStreaming()"
    [class.bg-indigo-100]="streamingEnabled()"
    [class.text-indigo-600]="streamingEnabled()"
    [class.bg-neutral-100]="!streamingEnabled()"
    [class.text-neutral-400]="!streamingEnabled()"
    class="flex items-center gap-1.5 h-9 px-3 rounded-full border border-black/5 text-xs font-bold transition-colors cursor-pointer">
    <span class="flex h-2 w-2 rounded-full" [class.bg-indigo-500]="streamingEnabled()" [class.bg-neutral-300]="!streamingEnabled()"></span>
    Stream
  </button>

  <!-- Send Button Anchor -->
  <button type="submit" [disabled]="!userInput().trim() || chatService.isThinking()"
    class="h-9 w-9 flex items-center justify-center bg-indigo-600 hover:bg-indigo-500 text-white disabled:opacity-20 disabled:hover:bg-indigo-600 rounded-full transition-all active:scale-95 cursor-pointer shadow-sm">
    <span class="font-bold text-sm">↑</span>
  </button>
</div>
```

> Note: the full sidebar/header/model-dropdown markup from the original `home-component.html` is unchanged except for the message loop (now uses `<app-chat-message>`) and the addition of the Stream toggle button above.

