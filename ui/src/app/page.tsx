"use client";

import { useState, useCallback, useRef } from "react";
import {
  Upload,
  FileText,
  Sparkles,
  MessageSquare,
  Loader2,
  Send,
  Eye,
} from "lucide-react";

type Mode = "extract" | "describe" | "ask";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

interface PageResult {
  page: number;
  content: string;
  has_extracted_text: boolean;
}

export default function Home() {
  const [file, setFile] = useState<File | null>(null);
  const [mode, setMode] = useState<Mode>("describe");
  const [prompt, setPrompt] = useState("Extract all information from this document");
  const [question, setQuestion] = useState("");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<string | PageResult[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [dragOver, setDragOver] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  const handleFile = useCallback((f: File) => {
    setFile(f);
    setResult(null);
    setError(null);
  }, []);

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setDragOver(false);
      const f = e.dataTransfer.files[0];
      if (f) handleFile(f);
    },
    [handleFile],
  );

  const handleSubmit = async () => {
    if (!file) return;
    setLoading(true);
    setError(null);
    setResult(null);

    const formData = new FormData();
    formData.append("file", file);

    try {
      let endpoint = "/describe";
      if (mode === "extract") {
        endpoint = "/extract";
        formData.append("prompt", prompt);
      } else if (mode === "ask") {
        endpoint = "/ask";
        formData.append("question", question || "What is this document about?");
      }

      const res = await fetch(`${API_BASE}${endpoint}`, { method: "POST", body: formData });
      if (!res.ok) {
        const txt = await res.text();
        throw new Error(`${res.status}: ${txt.slice(0, 200)}`);
      }
      const data = await res.json();

      if (mode === "extract") {
        setResult(data.pages as PageResult[]);
      } else if (mode === "describe") {
        setResult(data.description as string);
      } else {
        setResult(data.answer as string);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Request failed");
    } finally {
      setLoading(false);
    }
  };

  const modeTabs: { id: Mode; label: string; icon: typeof Eye }[] = [
    { id: "describe", label: "Describe", icon: Eye },
    { id: "extract", label: "Extract", icon: FileText },
    { id: "ask", label: "Ask", icon: MessageSquare },
  ];

  return (
    <div className="flex min-h-full flex-col">
      {/* Header */}
      <header className="flex items-center justify-between border-b border-border bg-surface/80 px-6 py-3 backdrop-blur-sm">
        <div className="flex items-center gap-2.5">
          <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-accent-subtle">
            <Sparkles className="h-4 w-4 text-accent" />
          </div>
          <div>
            <h1 className="text-sm font-semibold">mmdoc</h1>
            <p className="text-[10px] text-foreground-subtle">Multi-modal Document Understanding</p>
          </div>
        </div>
        <div className="text-xs text-foreground-subtle">Qwen3.5-0.8B · local GPU</div>
      </header>

      {/* Main */}
      <div className="mx-auto w-full max-w-3xl flex-1 px-4 py-8">
        {/* Upload area */}
        <div
          onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
          onDragLeave={() => setDragOver(false)}
          onDrop={handleDrop}
          onClick={() => inputRef.current?.click()}
          className={`flex cursor-pointer flex-col items-center justify-center rounded-2xl border-2 border-dashed py-12 transition-colors ${
            dragOver ? "border-accent bg-accent-subtle" : "border-border bg-surface-2 hover:border-border-hover"
          }`}
        >
          <input
            ref={inputRef}
            type="file"
            accept=".pdf,.png,.jpg,.jpeg,.gif,.tiff,.bmp,.webp"
            className="hidden"
            onChange={(e) => e.target.files?.[0] && handleFile(e.target.files[0])}
          />
          {file ? (
            <div className="flex flex-col items-center gap-2">
              <FileText className="h-10 w-10 text-accent" />
              <span className="text-sm font-medium">{file.name}</span>
              <span className="text-xs text-foreground-subtle">
                {(file.size / 1024).toFixed(0)} KB · click to replace
              </span>
            </div>
          ) : (
            <div className="flex flex-col items-center gap-2">
              <Upload className="h-10 w-10 text-foreground-subtle" />
              <span className="text-sm text-foreground-muted">Drop a file or click to upload</span>
              <span className="text-xs text-foreground-subtle">PDF · PNG · JPG · GIF · TIFF · BMP · WebP</span>
            </div>
          )}
        </div>

        {/* Mode tabs */}
        <div className="mt-5 flex gap-1.5">
          {modeTabs.map((tab) => {
            const Icon = tab.icon;
            return (
              <button
                key={tab.id}
                onClick={() => setMode(tab.id)}
                className={`flex items-center gap-2 rounded-lg px-4 py-2 text-sm font-medium transition-colors ${
                  mode === tab.id
                    ? "bg-surface-3 text-foreground"
                    : "text-foreground-muted hover:bg-surface-2 hover:text-foreground"
                }`}
              >
                <Icon className="h-4 w-4" />
                {tab.label}
              </button>
            );
          })}
        </div>

        {/* Mode-specific inputs */}
        <div className="mt-4">
          {mode === "extract" && (
            <textarea
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
              rows={2}
              placeholder="What to extract..."
              className="w-full resize-none rounded-xl border border-border bg-surface-2 px-4 py-3 text-sm text-foreground placeholder:text-foreground-subtle focus:border-accent/50 focus:outline-none"
            />
          )}
          {mode === "ask" && (
            <input
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              placeholder="Ask a question about the document..."
              className="w-full rounded-xl border border-border bg-surface-2 px-4 py-3 text-sm text-foreground placeholder:text-foreground-subtle focus:border-accent/50 focus:outline-none"
            />
          )}
        </div>

        {/* Submit button */}
        <button
          onClick={handleSubmit}
          disabled={!file || loading}
          className="mt-4 flex w-full items-center justify-center gap-2 rounded-xl bg-accent py-3 text-sm font-medium text-white transition-colors hover:bg-accent-hover disabled:opacity-40"
        >
          {loading ? (
            <>
              <Loader2 className="h-4 w-4 animate-spin" />
              Processing...
            </>
          ) : (
            <>
              <Send className="h-4 w-4" />
              {mode === "describe" ? "Describe Document" : mode === "extract" ? "Extract Data" : "Ask Question"}
            </>
          )}
        </button>

        {/* Error */}
        {error && (
          <div className="mt-4 rounded-xl border border-error/30 bg-error/10 px-4 py-3 text-sm text-error">
            {error}
          </div>
        )}

        {/* Results */}
        {result && (
          <div className="mt-6 animate-fade-in-up">
            <h3 className="mb-3 text-xs font-semibold uppercase tracking-wide text-foreground-subtle">
              Result
            </h3>
            {typeof result === "string" ? (
              <div className="rounded-xl border border-border bg-surface-2 p-4">
                <p className="whitespace-pre-wrap text-sm leading-relaxed text-foreground-muted">
                  {result}
                </p>
              </div>
            ) : (
              <div className="space-y-3">
                {(result as PageResult[]).map((page) => (
                  <div key={page.page} className="rounded-xl border border-border bg-surface-2 p-4">
                    <div className="mb-2 flex items-center gap-2">
                      <span className="rounded-md bg-surface-3 px-2 py-0.5 text-xs font-medium text-foreground-subtle">
                        Page {page.page}
                      </span>
                      {page.has_extracted_text && (
                        <span className="text-xs text-foreground-subtle">digital text detected</span>
                      )}
                    </div>
                    <p className="whitespace-pre-wrap text-sm leading-relaxed text-foreground-muted">
                      {page.content}
                    </p>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
