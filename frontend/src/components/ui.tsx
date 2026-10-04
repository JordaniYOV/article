import { useEffect, useRef, useState, type ReactNode } from 'react';
import { ArrowUpFromLine, Check, LoaderCircle, X, type LucideIcon } from 'lucide-react';

export type Perform = (key: string, action: () => Promise<unknown>, success?: string) => Promise<void>;
export function Panel({ id, icon: Icon, title, number, children, className = '' }:
  { id: string; icon: LucideIcon; title: string; number?: string; children: ReactNode; className?: string }) {
  return <section id={id} className={`panel ${className}`}>
    <header className="panel-heading"><span className="heading-icon"><Icon size={21} strokeWidth={2.2} /></span>
      <h2>{title}</h2>{number && <span className="step-number">{number}</span>}</header>{children}
  </section>;
}
export function DropZone({ accept, title, subtitle, onFile, disabled = false }:
  { accept: string; title: string; subtitle: string; onFile: (file: File) => void; disabled?: boolean }) {
  const input = useRef<HTMLInputElement>(null);
  const [dragging, setDragging] = useState(false);
  return <>
    <button type="button" disabled={disabled} className={`drop-zone ${dragging ? 'dragging' : ''}`}
      onClick={() => input.current?.click()} onDragOver={(event) => { event.preventDefault(); setDragging(true); }}
      onDragLeave={() => setDragging(false)} onDrop={(event) => {
        event.preventDefault(); setDragging(false);
        if (!disabled && event.dataTransfer.files[0]) onFile(event.dataTransfer.files[0]);
      }}>
      <span className="upload-icon"><ArrowUpFromLine size={27} /></span>
      <strong>{title}</strong><span>или <span className="purple-text">выберите файл</span></span>
      <small>{subtitle}</small>
    </button>
    <input ref={input} type="file" accept={accept} className="visually-hidden" aria-label={title}
      onChange={(event) => { if (event.target.files?.[0]) onFile(event.target.files[0]); event.target.value = ''; }} />
  </>;
}
export function Busy({ children }: { children?: ReactNode }) {
  return <><LoaderCircle size={16} className="spin" />{children}</>;
}
export function Empty({ icon: Icon, title, children }: { icon: LucideIcon; title: string; children?: ReactNode }) {
  return <div className="empty-state"><div className="empty-icon"><Icon size={26} /></div>
    <h3>{title}</h3>{children && <p>{children}</p>}</div>;
}
export function Notice({ message, kind, close }: { message: string; kind: 'error' | 'success'; close: () => void }) {
  return <div className={`toast ${kind}`} role={kind === 'error' ? 'alert' : 'status'}>
    <span className="toast-symbol">{kind === 'success' ? <Check size={18} /> : <X size={18} />}</span>
    <span>{message}</span><button type="button" className="icon-button" onClick={close} aria-label="Закрыть уведомление"><X size={16} /></button>
  </div>;
}
export function Modal({ title, children, close }: { title: string; children: ReactNode; close: () => void }) {
  const ref = useRef<HTMLDivElement>(null);
  const previousFocus = useRef(document.activeElement);
  useEffect(() => {
    const overflow = document.body.style.overflow;
    document.body.style.overflow = 'hidden';
    return () => {
      document.body.style.overflow = overflow;
      if (previousFocus.current instanceof HTMLElement && previousFocus.current.isConnected) previousFocus.current.focus();
    };
  }, []);
  // Autofocus and trap focus without a second dialog library.
  return <div className="modal-backdrop" onMouseDown={(event) => { if (event.target === event.currentTarget) close(); }}>
    <div className="modal" role="dialog" aria-modal="true" aria-label={title} ref={ref} tabIndex={-1}
      onKeyDown={(event) => {
        if (event.key === 'Escape') { event.preventDefault(); close(); }
        if (event.key === 'Tab') {
          const controls = ref.current?.querySelectorAll<HTMLElement>('button:not(:disabled), input:not(:disabled), select:not(:disabled), textarea:not(:disabled), a[href]');
          if (!controls?.length) return;
          const first = controls[0], last = controls[controls.length - 1];
          if (event.shiftKey && (document.activeElement === first || document.activeElement === ref.current)) { event.preventDefault(); last.focus(); }
          if (!event.shiftKey && document.activeElement === last) { event.preventDefault(); first.focus(); }
        }
      }}>
      <header className="modal-heading"><h2>{title}</h2><button type="button" autoFocus className="icon-button" onClick={close} aria-label="Закрыть окно"><X size={20} /></button></header>
      {children}
    </div>
  </div>;
}
