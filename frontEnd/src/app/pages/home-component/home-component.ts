import {Component, inject, OnInit, signal, effect, viewChild, ElementRef, ViewChild, computed} from '@angular/core';
import {CommonModule} from '@angular/common';
import {FormsModule} from '@angular/forms';
import {ChatService, ModelInfo, PendingAttachment} from '../../chat.service';
import {ChatMessageComponent} from '../../components/chat-message';

@Component({
  selector: 'app-chat-interface',
  standalone: true,
  imports: [CommonModule, FormsModule, ChatMessageComponent],
  templateUrl: './home-component.html'
})
export class HomeComponent implements OnInit {
  protected chatService = inject(ChatService);
  userInput = signal<string>('');
  streamingEnabled = signal<boolean>(false);

   /** Files the user has attached via the paperclip button, staged until send. */
  selectedFiles = signal<PendingAttachment[]>([]);

  private scrollContainer = viewChild<ElementRef<HTMLDivElement>>('scrollFrame');
  private fileInputRef = viewChild<ElementRef<HTMLInputElement>>('fileInput');

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
    this.scrollFrame?.nativeElement.scrollTo({top: this.scrollFrame.nativeElement.scrollHeight});
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
    this.chatListFrame?.nativeElement.scrollTo({top: this.chatListFrame.nativeElement.scrollHeight});
  }

  onNewChat(): void {
    this.chatService.startNewChat();
  }

  onModelChange(model: ModelInfo) {
    this.chatService.selectedModel.set(model);
  }

  onToggleStreaming(): void {
    this.streamingEnabled.update(v => !v);
  }

  /*
   * FILE ATTACHMENT HANDLING
   */

  onAttachClick(): void {
    this.fileInputRef()?.nativeElement.click();
  }

  onFilesSelected(event: Event): void {
    const input = event.target as HTMLInputElement;
    const files = input.files ? Array.from(input.files) : [];

    const pending: PendingAttachment[] = files.map(file => ({
      file,
      previewUrl: file.type.startsWith('image/') ? URL.createObjectURL(file) : undefined
    }));

    this.selectedFiles.update(prev => [...prev, ...pending]);

    // Reset so selecting the same file again still fires a change event.
    input.value = '';
  }

  removeAttachment(index: number): void {
    this.selectedFiles.update(prev => {
      const target = prev[index];
      if (target?.previewUrl) URL.revokeObjectURL(target.previewUrl);
      return prev.filter((_, i) => i !== index);
    });
  }

  private clearAttachments(): void {
    this.selectedFiles().forEach(a => {
      if (a.previewUrl) URL.revokeObjectURL(a.previewUrl);
    });
    this.selectedFiles.set([]);
  }

  formatSize(bytes: number): string {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  }

  /*
   * SUBMIT / RETRY — now files-aware
   */

  submitMessage(): void {
    const text = this.userInput().trim();
    const files = this.selectedFiles().map(a => a.file);

    if ((!text && files.length === 0) || this.chatService.isThinking()) return;

    if (this.streamingEnabled()) {
      this.chatService.sendMessageStream(text, files);
    } else {
      this.chatService.sendMessage(text, files);
    }

    this.userInput.set('');
    this.clearAttachments();
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

  isModelMenuOpen = signal(false);

  onToggleModelMenu() {
    this.isModelMenuOpen.update(v => !v);
  }

  selectedModelName = computed(() => {
    const key = this.chatService.selectedModel().modelKey;
    const all = [...this.chatService.localModels(), ...this.chatService.cloudModels()];
    return all.find(m => m.modelKey === key)?.name ?? '';
  });

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

  // Bottom action triggers
  onTriggerSettings() {
    console.log('Settings triggered');
  }

  onTriggerProjects() {
    console.log('Projects triggered');
  }

  onTriggerReset() {
    this.chatService.startNewChat();
  }

  onTriggerAbout() {
    console.log('About context window opened');
  }
}
