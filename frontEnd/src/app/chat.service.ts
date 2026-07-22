import {inject, Injectable, signal} from '@angular/core';
import {HttpClient} from '@angular/common/http';

export interface ModelInfo {
  id: number;
  name: string;
  modelKey: string;
  badge?: string
}
//** A file staged in the composer, not yet sent. */
export interface PendingAttachment {
  file: File;
  previewUrl?: string;
}

export interface ChatAttachment {
  name: string;
  size: number;
  type: string;
  /** Local object URL for image previews. Only ever set client-side; never sent to / returned from the API. */
  previewUrl?: string;
}

export interface ChatMessage {
  role: 'user' | 'assistant' | string;
  content: string;
  animate?: boolean;
  attachments?: ChatAttachment[];
}

export interface ChatRequest {
  userId: number;
  chatId: number;
  message: string;
  modelName: string;
}

export interface ChatResponse {
  userId: number;
  chatId: number;
  messages: ChatMessage[];
}

export interface ConversationEntity {
  chatId: number;
  chatName: string;
}

@Injectable({providedIn: 'root'})
export class ChatService {
  private http = inject(HttpClient);
  private apiUrl = 'http://localhost:8000';

  currentChatId = signal<number>(0);
  userId = signal<number>(1);
  messages = signal<ChatMessage[]>([]);
  conversations = signal<ConversationEntity[]>([]);
  localModels = signal<ModelInfo[]>([]);
  cloudModels = signal<ModelInfo[]>([]);
  selectedModel = signal<ModelInfo>({modelKey: "", name: "", id: 0, badge: ""});

  // Status tracking states
  isThinking = signal<boolean>(false);
  thinkingStatus = signal<string>('Thinking...');

  private thinkingPhrases = [
    'Thinking...', 'Contemplating...', 'Analyzing variables...',
    'Formulating response...', 'Synthesizing knowledge...', 'Consulting neural map...'
  ];

   /**
   * Builds the metadata shown in the chat bubble for files the user attached.
   * Creates local object URLs for images so they can be previewed inline;
   * these are never uploaded — only the underlying File objects are sent to the API.
   */
  private toAttachments(files: File[]): ChatAttachment[] | undefined {
    if (!files.length) return undefined;
    return files.map(file => ({
      name: file.name,
      size: file.size,
      type: file.type,
      previewUrl: file.type.startsWith('image/') ? URL.createObjectURL(file) : undefined
    }));
  }

  /**
   * Builds a multipart/form-data payload for /chat/ and /chat/stream/.
   * IMPORTANT: never set a Content-Type header manually alongside a FormData body —
   * both HttpClient and fetch() will generate the correct multipart boundary themselves.
   */
  private buildFormData(message: string, files: File[]): FormData {
    const formData = new FormData();
    formData.append('userId', String(this.userId()));
    formData.append('chatId', String(this.currentChatId()));
    formData.append("modelName",String(this.selectedModel().modelKey))
    formData.append('message', message);
    for (const file of files) {
      formData.append('files', file, file.name);
    }
    return formData;
  }

  private startThinkingAnimation(): any {
    this.isThinking.set(true);
    this.thinkingStatus.set(this.thinkingPhrases[0]);
    let index = 1;
    return setInterval(() => {
      this.thinkingStatus.set(this.thinkingPhrases[index % this.thinkingPhrases.length]);
      index++;
    }, 2500);
  }

  loadConversations(): void {
    this.http.get<ConversationEntity[]>(`${this.apiUrl}/conversations/all?userId=${this.userId()}`)
      .subscribe(data => this.conversations.set(data));
  }

  loadModels(): void {
    this.http.get<ModelInfo[]>(`${this.apiUrl}/models/cloud`).subscribe(data => {
      if (data.length > 0) {
        this.cloudModels.set(data)
        if (this.selectedModel().id == 0) {
          this.selectedModel.set({
            id: data[0].id,
            modelKey: data[0].modelKey,
            name: data[0].name,
            badge: ""
          });
        }
      }
    });

    this.http.get<ModelInfo[]>(`${this.apiUrl}/models/local`).subscribe(data => {
      this.localModels.set(data);
      if (data.length > 0) {
        if (this.selectedModel().id == 0) {
          this.selectedModel.set({
            id: data[0].id,
            modelKey: data[0].modelKey,
            name: data[0].name,
            badge: ""
          });
        }
      }
    });
  }

  loadChatMessages(chatId: number): void {
    this.currentChatId.set(chatId);
    this.http.get<ChatMessage[]>(`${this.apiUrl}/chat/${chatId}/messages`)
      .subscribe(data => this.messages.set(data));
  }

  /*
   * NON-STREAMING STANDARD HTTP POST PIPELINE
   * `files` is optional so existing callers that only pass a string keep working.
   */
  sendMessage(messageContent: string, files: File[] = []): void {
    const userMsg: ChatMessage = {
      role: 'user',
      content: messageContent,
      attachments: this.toAttachments(files)
    };
    this.messages.update(prev => [...prev, userMsg]);

    const intervalId = this.startThinkingAnimation();

    const formData = this.buildFormData(messageContent, files);

    this.http.post<ChatResponse>(`${this.apiUrl}/chat/`, formData).subscribe({
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
   * Uses fetch + ReadableStream to consume Server-Sent Events from /chat/stream/.
   * `files` is optional so existing callers that only pass a string keep working.
   */
  async sendMessageStream(messageContent: string, files: File[] = []): Promise<void> {
    const userMsg: ChatMessage = {
      role: 'user',
      content: messageContent,
      attachments: this.toAttachments(files)
    };
    this.messages.update(prev => [...prev, userMsg]);

    const intervalId = this.startThinkingAnimation();

    const formData = this.buildFormData(messageContent, files);

    try {
      const response = await fetch(`${this.apiUrl}/chat/stream/`, {
        method: 'POST',
        // No headers set here on purpose — the browser derives the correct
        // `multipart/form-data; boundary=...` Content-Type from the FormData body.
        body: formData
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
