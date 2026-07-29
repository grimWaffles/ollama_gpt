import {Component, inject, OnInit, OnDestroy, AfterViewInit, signal, effect, viewChild, ElementRef, ViewChild, computed} from '@angular/core';
import {CommonModule} from '@angular/common';
import {FormsModule} from '@angular/forms';
import {ChatService, ModelInfo, PendingAttachment} from '../../chat.service';
import {ChatMessageComponent, MessageCopyPayload} from '../../components/chat-message';

const DARK_MODE_KEY = 'sagegpt-dark-mode';
const SIDEBAR_OPEN_KEY = 'sagegpt-sidebar-open';

/** Max height (px) the composer textarea is allowed to grow to before it scrolls internally. */
const TEXTAREA_MAX_HEIGHT = 200;

@Component({
  selector: 'app-chat-interface',
  standalone: true,
  imports: [CommonModule, FormsModule, ChatMessageComponent],
  templateUrl: './home-component.html'
})
export class HomeComponent implements OnInit, AfterViewInit, OnDestroy {
  protected chatService = inject(ChatService);
  userInput = signal<string>('');
  streamingEnabled = signal<boolean>(false);

   /** Files the user has attached via the paperclip button, staged until send. */
  selectedFiles = signal<PendingAttachment[]>([]);

  /** Sidebar open/collapsed state — persisted so it survives reloads. */
  sidebarOpen = signal<boolean>(true);

  /** Dark mode state — persisted, and initialized from OS preference if unset. */
  isDarkMode = signal<boolean>(false);

  private scrollContainer = viewChild<ElementRef<HTMLDivElement>>('scrollFrame');
  private fileInputRef = viewChild<ElementRef<HTMLInputElement>>('fileInput');
  private textareaRef = viewChild<ElementRef<HTMLTextAreaElement>>('messageTextarea');
  private composerRef = viewChild<ElementRef<HTMLElement>>('composerFooter');

  /** Live footer height (px), fed to the message list as scroll padding so the last
   *  message is never hidden behind the sticky/floating composer. */
  composerHeight = signal<number>(140);
  private composerResizeObserver?: ResizeObserver;

  constructor() {
    // Automatically triggers execution thread anytime messages or thinking states change
    effect(() => {
      this.chatService.messages();
      this.chatService.isThinking();
      this.scrollToBottom();
    });

    // Keep the <html> root in sync with dark mode state and persist the choice.
    effect(() => {
      const dark = this.isDarkMode();
      if (typeof document !== 'undefined') {
        document.documentElement.classList.toggle('dark', dark);
      }
      this.safeSetItem(DARK_MODE_KEY, dark ? '1' : '0');
    });

    effect(() => {
      this.safeSetItem(SIDEBAR_OPEN_KEY, this.sidebarOpen() ? '1' : '0');
    });
  }

  ngOnInit(): void {
    this.chatService.loadConversations();
    this.chatService.loadModels();
    this.restorePreferences();
  }

  ngAfterViewInit(): void {
    // Track the composer's real rendered height so the scrollable message
    // list can reserve exactly enough bottom padding to clear it.
    const el = this.composerRef()?.nativeElement;
    if (el && typeof ResizeObserver !== 'undefined') {
      this.composerResizeObserver = new ResizeObserver(entries => {
        for (const entry of entries) {
          this.composerHeight.set(Math.ceil(entry.contentRect.height) + 24);
        }
      });
      this.composerResizeObserver.observe(el);
    }
  }

  ngOnDestroy(): void {
    this.composerResizeObserver?.disconnect();
  }

  private restorePreferences(): void {
    const storedDark = this.safeGetItem(DARK_MODE_KEY);
    if (storedDark !== null) {
      this.isDarkMode.set(storedDark === '1');
    } else if (typeof window !== 'undefined' && window.matchMedia) {
      this.isDarkMode.set(window.matchMedia('(prefers-color-scheme: dark)').matches);
    }

    const storedSidebar = this.safeGetItem(SIDEBAR_OPEN_KEY);
    if (storedSidebar !== null) {
      this.sidebarOpen.set(storedSidebar === '1');
    } else if (typeof window !== 'undefined' && window.innerWidth < 768) {
      // Default to collapsed on small screens so it doesn't eat the viewport.
      this.sidebarOpen.set(false);
    }
  }

  private safeGetItem(key: string): string | null {
    try {
      return localStorage.getItem(key);
    } catch {
      return null;
    }
  }

  private safeSetItem(key: string, value: string): void {
    try {
      localStorage.setItem(key, value);
    } catch {
      /* storage unavailable (private mode, SSR, etc.) — non-fatal */
    }
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
   * SIDEBAR
   */

  toggleSidebar(): void {
    this.sidebarOpen.update(v => !v);
  }

  /*
   * DARK MODE
   */

  toggleDarkMode(): void {
    this.isDarkMode.update(v => !v);
  }

  /*
   * COMPOSER — auto-growing textarea
   */

  autoResizeTextarea(): void {
    const el = this.textareaRef()?.nativeElement;
    if (!el) return;
    // Reset first so shrinking (e.g. deleting text) is measured correctly.
    el.style.height = 'auto';
    el.style.height = Math.min(el.scrollHeight, TEXTAREA_MAX_HEIGHT) + 'px';
    el.style.overflowY = el.scrollHeight > TEXTAREA_MAX_HEIGHT ? 'auto' : 'hidden';
  }

  private resetTextareaHeight(): void {
    // Run after the DOM updates from clearing userInput.
    setTimeout(() => this.autoResizeTextarea(), 0);
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
    this.resetTextareaHeight();
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

  /**
   * Writes a message to the clipboard. When rendered HTML is available
   * (assistant messages), writes both `text/html` and `text/plain` via the
   * async Clipboard API so pasting into rich-text targets (docs, email,
   * Slack) keeps real formatting instead of raw markdown or a flattened
   * text chunk. Falls back to plain-text copy if the rich API is
   * unavailable or rejected (e.g. insecure context, older browsers).
   */
  async copyMessage(payload: MessageCopyPayload): Promise<void> {
    const { text, html } = payload;

    if (html && typeof ClipboardItem !== 'undefined' && navigator.clipboard?.write) {
      try {
        const item = new ClipboardItem({
          'text/plain': new Blob([text], { type: 'text/plain' }),
          'text/html': new Blob([html], { type: 'text/html' }),
        });
        await navigator.clipboard.write([item]);
        return;
      } catch {
        // fall through to plain text
      }
    }

    try {
      await navigator.clipboard.writeText(text);
    } catch {
      /* clipboard unavailable — nothing more we can do silently */
    }
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
