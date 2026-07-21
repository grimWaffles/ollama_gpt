import {inject, Injectable, signal} from '@angular/core';
import {HttpClient} from '@angular/common/http';

export interface ModelInfo {
  id: number;
  name: string;
  modelKey: string;
  badge?: string
}

export interface ChatMessage {
  role: 'user' | 'assistant' | string;
  content: string;
}

export interface ChatRequest {
  userId: number;
  chatId: number;
  message: string;
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
  selectedModelKey = signal<string>('');

  // Status tracking states
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
    }, 2500);
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
   * Directly posts to your FastAPI endpoint and sets the full message list on return.
   */
  sendMessage(messageContent: string): void {
    // Optimistic UI Update: immediately render what the user wrote
    const userMsg: ChatMessage = {role: 'user', content: messageContent};
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

        // Update application contexts with full backend payloads
        this.currentChatId.set(res.chatId);
        this.messages.update(prev => [...prev, ...res.messages]);

        // Refresh sidebar conversation logs
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
    const userMsg: ChatMessage = {role: 'user', content: messageContent};
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
        headers: {'Content-Type': 'application/json'},
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
        const {done, value} = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, {stream: true});
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
          const animated = res.messages.map(m => ({...m, animate: true}));
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
