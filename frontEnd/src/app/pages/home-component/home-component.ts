import { Component, inject, OnInit, signal } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { ChatService } from '../../chat.service';

@Component({
  selector: 'app-chat-interface',
  standalone: true,
  imports: [CommonModule, FormsModule],
  templateUrl: './home-component.html',
  styleUrls: [] // Tailwind layout handles all configurations directly
})
export class HomeComponent implements OnInit {
  protected chatService = inject(ChatService);
  
  // Local binding UI state
  userInput = signal<string>('');

  ngOnInit(): void {
    this.chatService.loadConversations();
    this.chatService.loadModels();
  }

  onSelectConversation(id: number): void {
    this.chatService.loadChatMessages(id);
  }

  onNewChat(): void {
    this.chatService.startNewChat();
  }

  onModelChange(event: Event): void {
    const target = event.target as HTMLSelectElement;
    this.chatService.selectedModelKey.set(target.value);
  }

  submitMessage(): void {
    const text = this.userInput().trim();
    if (!text) return;
    
    this.chatService.sendMessage(text);
    this.userInput.set(''); // Clear form field
  }
}