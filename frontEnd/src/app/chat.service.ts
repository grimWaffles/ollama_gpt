import { inject, Injectable, signal } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable, tap } from 'rxjs';

export interface ModelInfo {
    id: number;
    name: string;
    modelKey: string;
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

@Injectable({
    providedIn: 'root'
})
export class ChatService {
    private http = inject(HttpClient);
    private apiUrl = 'http://localhost:8000'; // Update to your FastAPI server URL

    // Global State via Signals
    currentChatId = signal<number>(0);
    userId = signal<number>(1); // Mocked user ID
    messages = signal<ChatMessage[]>([]);
    conversations = signal<ConversationEntity[]>([]);
    localModels = signal<ModelInfo[]>([]);
    cloudModels = signal<ModelInfo[]>([]);
    selectedModelKey = signal<string>('');

    loadConversations(): void {
        this.http.get<ConversationEntity[]>(`${this.apiUrl}/conversations/all?userId=${this.userId()}`)
            .subscribe(data => this.conversations.set(data));
    }

    loadModels(): void {
        this.http.get<ModelInfo[]>(`${`${this.apiUrl}/models/local`}`).subscribe(data => {
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

    sendMessage(messageContent: string): void {
        // Optimistic UI Update
        const userMsg: ChatMessage = { role: 'user', content: messageContent };
        this.messages.update(prev => [...prev, userMsg]);

        const body: ChatRequest = {
            userId: this.userId(),
            chatId: this.currentChatId(),
            message: messageContent
        };

        this.http.post<ChatResponse>(`${this.apiUrl}/chat/`, body).subscribe(res => {
            this.currentChatId.set(res.chatId);
            this.messages.update(prev => [...prev, ...res.messages]);
            // Refresh sidebar if it's a completely new conversation thread
            this.loadConversations();
        });
    }

    startNewChat(): void {
        this.currentChatId.set(0);
        this.messages.set([]);
    }
}