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
    <div class="flex w-full gap-2.5" [class.justify-end]="isUser()" [class.justify-start]="!isUser()">

      @if (!isUser()) {
        <div class="shrink-0 h-8 w-8 rounded-full bg-gradient-to-br from-indigo-400 to-fuchsia-400 flex items-center justify-center text-white text-xs font-bold shadow-[0_2px_8px_rgba(129,140,248,0.5)] mt-0.5">
          ✦
        </div>
      }

      <div class="flex flex-col max-w-[72%]" [class.items-end]="isUser()">

        <div
          class="rounded-3xl px-4 py-3 transition-transform hover:-translate-y-0.5"
          [class.bg-gradient-to-br]="isUser()"
          [class.from-indigo-500]="isUser()"
          [class.to-violet-500]="isUser()"
          [class.text-white]="isUser()"
          [class.rounded-br-md]="isUser()"
          [class.shadow-[0_6px_20px_rgba(99,102,241,0.35)\]]="isUser()"
          [class.bg-white-80]="!isUser()"
          [class.backdrop-blur-xl]="!isUser()"
          [class.text-slate-700]="!isUser()"
          [class.rounded-bl-md]="!isUser()"
          [class.border]="!isUser()"
          [class.border-white-80]="!isUser()"
          [class.shadow-[0_4px_16px_rgba(15,23,42,0.06)]]="!isUser()"
        >
          <div
            class="text-[10px] font-bold uppercase tracking-wider mb-1.5"
            [class.text-indigo-100]="isUser()"
            [class.text-indigo-400]="!isUser()"
          >
            {{ isUser() ? 'You' : 'Sage Engine' }}
          </div>

          @if (msg().role === 'user') {
            <div class="whitespace-pre-wrap break-words text-base leading-relaxed">
              {{ msg().content }}
            </div>
          } @else {
            <markdown
              class="prose prose-slate prose-sm max-w-none"
              [data]="displayedContent()">
            </markdown>
          }
        </div>

        @if (!isUser()) {
          <div class="flex items-center gap-1 mt-1.5 pl-1">
            <button
              type="button"
              (click)="onCopy()"
              title="Copy"
              class="p-1.5 rounded-lg text-slate-400 hover:text-indigo-500 hover:bg-white/70 transition-colors cursor-pointer"
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
              class="p-1.5 rounded-lg text-slate-400 hover:text-indigo-500 hover:bg-white/70 transition-colors cursor-pointer"
            >
              <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <path d="M3 12a9 9 0 1 0 3-6.7"></path>
                <path d="M3 4v5h5"></path>
              </svg>
            </button>
          </div>
        }
      </div>

      @if (isUser()) {
        <div class="shrink-0 h-8 w-8 rounded-full bg-slate-200 flex items-center justify-center text-slate-500 text-xs font-bold mt-0.5">
          U
        </div>
      }
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
