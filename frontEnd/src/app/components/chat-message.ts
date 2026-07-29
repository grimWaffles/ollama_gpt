import {Component, input, output, ChangeDetectionStrategy, computed, signal, effect, OnDestroy, ElementRef, viewChild} from '@angular/core';
import { CommonModule } from '@angular/common';
import {MarkdownComponent} from 'ngx-markdown';
import {ChatMessage} from '../chat.service';

/**
 * Payload emitted when the user copies a message. `html` (when present)
 * carries the fully-rendered markup so pasting into rich-text targets
 * (docs, email, Slack, etc.) preserves bold/italic/lists/code formatting
 * instead of dumping raw markdown syntax or a flattened text blob.
 */
export interface MessageCopyPayload {
  text: string;
  html?: string;
}

@Component({
  selector: 'app-chat-message',
  standalone: true,
  imports: [CommonModule, MarkdownComponent],
  changeDetection: ChangeDetectionStrategy.OnPush,
  styles:`
    :host {
      display: block;
    }
  `,
  template: `
    <div class="flex w-full gap-2.5" [class.justify-end]="isUser()" [class.justify-start]="!isUser()">

      @if (!isUser()) {
        <div class="shrink-0 h-8 w-8 rounded-full bg-gradient-to-br from-indigo-400 to-fuchsia-400 flex items-center justify-center text-white text-xs font-bold shadow-[0_2px_8px_rgba(129,140,248,0.5)] mt-0.5">
          ✦
        </div>
      }

      <div class="flex flex-col max-w-[85%] sm:max-w-[75%] md:max-w-[72%]" [class.items-end]="isUser()">

        <div
          class="rounded-3xl px-4 py-3"
          [style.boxShadow]="isUser() ? '0 6px 20px rgba(99,102,241,0.35)' : '0 4px 16px rgba(15,23,42,0.06)'"
          [class.bg-gradient-to-br]="isUser()"
          [class.from-indigo-500]="isUser()"
          [class.to-violet-500]="isUser()"
          [class.text-white]="isUser()"
          [class.rounded-br-md]="isUser()"
          [class.bg-white-80]="!isUser()"
          [class.dark:bg-slate-800/70]="!isUser()"
          [class.backdrop-blur-xl]="!isUser()"
          [class.text-slate-700]="!isUser()"
          [class.dark:text-white]="!isUser()"
          [class.rounded-bl-md]="!isUser()"
          [class.border]="!isUser()"
          [class.border-white-80]="!isUser()"
          [class.dark:border-white/10]="!isUser()"
        >
          <div
            class="text-[10px] font-bold uppercase tracking-wider mb-1.5"
            [class.text-indigo-100]="isUser()"
            [class.text-indigo-400]="!isUser()"
            [class.dark:text-indigo-300]="!isUser()"
          >
            {{ isUser() ? 'You' : 'Sage Engine' }}
          </div>
            @if (attachments().length > 0) {
          <div class="flex flex-wrap gap-2 mb-2">
            @for (attachment of attachments(); track attachment.name) {
              @if (attachment.previewUrl) {
                <img
                  [src]="attachment.previewUrl"
                  [alt]="attachment.name"
                  class="h-20 w-20 object-cover rounded-xl border"
                  [class.border-white-30]="isUser()"
                  [class.border-black-5]="!isUser()"
                />
              } @else {
                <div
                  class="flex items-center gap-1.5 rounded-xl px-2.5 py-1.5 text-xs border"
                  [class.bg-white-10]="isUser()"
                  [class.border-white-30]="isUser()"
                  [class.bg-black-5]="!isUser()"
                  [class.border-black-5]="!isUser()"
                >
                  <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                    <path d="M21.44 11.05l-9.19 9.19a6 6 0 0 1-8.49-8.49l9.19-9.19a4 4 0 0 1 5.66 5.66l-9.2 9.19a2 2 0 0 1-2.83-2.83l8.49-8.48"></path>
                  </svg>
                  <span class="font-medium truncate max-w-[10rem]">{{ attachment.name }}</span>
                  <span class="opacity-60">{{ formatSize(attachment.size) }}</span>
                </div>
              }
            }
          </div>
        }
          @if (msg().role === 'user') {
            <div class="user-message-text whitespace-pre-wrap break-words text-base leading-relaxed">
              {{ msg().content }}
            </div>
          } @else {
            <div class="assistant-markdown" #markdownContent>
              <markdown
                class="max-w-none"
                [data]="displayedContent()">
              </markdown>
            </div>
          }
        </div>

        <div
          class="flex items-center gap-1 mt-1.5"
          [class.pl-1]="!isUser()"
          [class.pr-1]="isUser()"
          [class.justify-end]="isUser()"
        >
          <button
            type="button"
            (click)="onCopy()"
            title="Copy"
            class="p-1.5 rounded-lg text-slate-400 hover:text-indigo-500 hover:bg-white/70 dark:text-white/50 dark:hover:text-white dark:hover:bg-white/10 transition-colors cursor-pointer"
          >
            @if (justCopied()) {
              <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <polyline points="20 6 9 17 4 12"></polyline>
              </svg>
            } @else {
              <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <rect x="9" y="9" width="13" height="13" rx="2"></rect>
                <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path>
              </svg>
            }
          </button>
          @if (!isUser()) {
            <button
              type="button"
              (click)="onRetry()"
              title="Retry"
              class="p-1.5 rounded-lg text-slate-400 hover:text-indigo-500 hover:bg-white/70 dark:text-white/50 dark:hover:text-white dark:hover:bg-white/10 transition-colors cursor-pointer"
            >
              <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <path d="M3 12a9 9 0 1 0 3-6.7"></path>
                <path d="M3 4v5h5"></path>
              </svg>
            </button>
          }
        </div>
      </div>

      @if (isUser()) {
        <div class="shrink-0 h-8 w-8 rounded-full bg-slate-200 dark:bg-white/15 flex items-center justify-center text-slate-500 dark:text-white text-xs font-bold mt-0.5">
          U
        </div>
      }
    </div>
  `,
})
export class ChatMessageComponent implements OnDestroy {
  msg = input.required<ChatMessage>();

  isUser = computed(() => this.msg().role === 'user');
  attachments = computed(() => this.msg().attachments ?? []);
  copy = output<MessageCopyPayload>();
  retry = output<void>();

  displayedContent = signal<string>('');
  justCopied = signal<boolean>(false);
  private copiedTimerId: ReturnType<typeof setTimeout> | null = null;

  private markdownContentRef = viewChild<ElementRef<HTMLElement>>('markdownContent');

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
    if (this.copiedTimerId) clearTimeout(this.copiedTimerId);
  }

  onCopy(): void {
    // User messages are plain text already — nothing to render.
    if (this.isUser()) {
      this.copy.emit({ text: this.msg().content });
      this.flashCopied();
      return;
    }

    // Assistant messages: pull the *rendered* markup out of the DOM so the
    // consumer can copy real formatting (bold, lists, code blocks, links)
    // rather than raw markdown syntax or an unformatted text dump.
    const el = this.markdownContentRef()?.nativeElement;
    if (el) {
      this.copy.emit({ text: el.innerText, html: el.innerHTML });
    } else {
      this.copy.emit({ text: this.msg().content });
    }
    this.flashCopied();
  }

  private flashCopied(): void {
    this.justCopied.set(true);
    if (this.copiedTimerId) clearTimeout(this.copiedTimerId);
    this.copiedTimerId = setTimeout(() => this.justCopied.set(false), 1400);
  }

  onRetry(): void {
    this.retry.emit();
  }
  formatSize(bytes: number): string {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  }
}
