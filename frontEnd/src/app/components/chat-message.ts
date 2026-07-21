import {Component, input, output, ChangeDetectionStrategy, computed, signal, effect, OnDestroy} from '@angular/core';
import { CommonModule } from '@angular/common';
import {MarkdownComponent} from 'ngx-markdown';

export interface ChatMessage { role: 'user' | 'assistant' | string; content: string; animate?: boolean; }

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
        <!-- Header -->
        <div
          class="text-[10px] font-bold uppercase tracking-wider mb-1.5"
          [class.text-indigo-100]="isUser()"
          [class.text-neutral-400]="!isUser()"
        >
          {{ isUser() ? 'User Identity' : 'Sage Engine' }}
        </div>

        <!-- Body -->
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

        <!-- Actions (Assistant/Engine only) -->
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
  speedInMs = signal<number>(8)
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
      }, this.speedInMs());
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
