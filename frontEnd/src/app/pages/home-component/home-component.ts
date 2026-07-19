import { Component, inject, OnInit, signal, effect, viewChild, ElementRef, ViewChild } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { ChatService } from '../../chat.service';

@Component({
  selector: 'app-chat-interface',
  standalone: true,
  imports: [CommonModule, FormsModule],
  templateUrl: './home-component.html'
})
export class HomeComponent implements OnInit {
  protected chatService = inject(ChatService);
  userInput = signal<string>('');

  // Target viewport selector anchor reference
  private scrollContainer = viewChild<ElementRef<HTMLDivElement>>('scrollFrame');

  constructor() {
    // Automatically triggers execution thread anytime messages or thinking states change
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

  onModelChange(event: Event): void {
    const target = event.target as HTMLSelectElement;
    this.chatService.selectedModelKey.set(target.value);
  }

  submitMessage(): void {
    const text = this.userInput().trim();
    if (!text || this.chatService.isThinking()) return;

    // Directly fires the standard complete HTTP call transaction
    this.chatService.sendMessage(text);
    this.userInput.set('');
  }

  handleEnterKey(event: Event): void {
    const keyboardEvent = event as KeyboardEvent;

    // If Shift + Enter is pressed, do nothing and let the newline wrap naturally
    if (keyboardEvent.shiftKey) {
      return;
    }

    // Prevent default carriage return and submit the message string
    keyboardEvent.preventDefault();
    this.submitMessage();
  }

  // Bottom action triggers
  onTriggerSettings() { console.log('Settings triggered'); }
  onTriggerProjects() { console.log('Projects triggered'); }
  onTriggerReset() { this.chatService.startNewChat(); }
  onTriggerAbout() { console.log('About context window opened'); }
}