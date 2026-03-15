import { useState, useRef, useEffect, useCallback } from "react";
import { ImagePlus, Send, Square, X, ArrowUp, Mic, MicOff } from "lucide-react";
import { toast } from "sonner";

interface Props {
  onSend: (text: string, imageBase64?: string) => void;
  onStop: () => void;
  streaming: boolean;
  disabled?: boolean;
}

const MAX_IMAGE_SIZE = 10 * 1024 * 1024; // 10MB

export default function ChatInput({ onSend, onStop, streaming, disabled }: Props) {
  const [text, setText] = useState("");
  const [imagePreview, setImagePreview] = useState<string | null>(null);
  const [imageBase64, setImageBase64] = useState<string | null>(null);
  const [focused, setFocused] = useState(false);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // ── Voice recording state ──
  const [recording, setRecording] = useState(false);
  const [voiceEnabled, setVoiceEnabled] = useState(false);
  const [transcribing, setTranscribing] = useState(false);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);

  // Check if voice is enabled on mount
  useEffect(() => {
    const checkVoice = async () => {
      try {
        const resp = await fetch("/api/voice/transcribe", { method: "POST" });
        // If it's not a 404, voice endpoints exist (might be 400/422 due to no file, that's OK)
        setVoiceEnabled(resp.status !== 404);
      } catch {
        setVoiceEnabled(false);
      }
    };
    checkVoice();
  }, []);

  useEffect(() => {
    if (!streaming) textareaRef.current?.focus();
  }, [streaming]);

  const handleFile = useCallback((file: File) => {
    if (!file.type.startsWith("image/")) return;
    if (file.size > MAX_IMAGE_SIZE) {
      toast.error("Image must be under 10MB");
      return;
    }
    const reader = new FileReader();
    reader.onload = () => {
      const result = reader.result as string;
      setImagePreview(result);
      const base64 = result.split(",")[1] || result;
      setImageBase64(base64);
    };
    reader.readAsDataURL(file);
  }, []);

  const clearImage = () => {
    setImagePreview(null);
    setImageBase64(null);
    if (fileInputRef.current) fileInputRef.current.value = "";
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      submit();
    }
    if (e.key === "Escape" && streaming) {
      onStop();
    }
  };

  const handlePaste = (e: React.ClipboardEvent) => {
    const items = e.clipboardData?.items;
    if (!items) return;
    for (const item of items) {
      if (item.type.startsWith("image/")) {
        e.preventDefault();
        const file = item.getAsFile();
        if (file) handleFile(file);
        return;
      }
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    const file = e.dataTransfer?.files?.[0];
    if (file && file.type.startsWith("image/")) {
      handleFile(file);
    }
  };

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
  };

  const submit = () => {
    const trimmed = text.trim();
    if ((!trimmed && !imageBase64) || streaming || disabled) return;
    const query = trimmed || "What is in this image?";
    onSend(query, imageBase64 || undefined);
    setText("");
    clearImage();
  };

  // ── Voice recording handlers ──

  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const recorder = new MediaRecorder(stream);
      chunksRef.current = [];

      recorder.ondataavailable = (e) => {
        if (e.data.size > 0) chunksRef.current.push(e.data);
      };

      recorder.onstop = async () => {
        stream.getTracks().forEach((t) => t.stop());
        const blob = new Blob(chunksRef.current, { type: "audio/webm" });

        if (blob.size === 0) return;

        setTranscribing(true);
        try {
          const form = new FormData();
          form.append("file", blob, "recording.webm");

          const controller = new AbortController();
          const timeout = setTimeout(() => controller.abort(), 30000);

          const resp = await fetch("/api/voice/transcribe", {
            method: "POST",
            body: form,
            signal: controller.signal,
          });
          clearTimeout(timeout);

          if (!resp.ok) {
            const statusMsg =
              resp.status === 413 ? "Recording too large — try a shorter clip"
              : resp.status === 422 ? "Audio format not supported"
              : resp.status === 503 ? "Transcription service unavailable — is Whisper loaded?"
              : `Transcription failed (HTTP ${resp.status})`;
            throw new Error(statusMsg);
          }
          const data = await resp.json();
          if (data.text) {
            setText((prev) => (prev ? prev + " " + data.text : data.text));
            textareaRef.current?.focus();
          }
        } catch (err) {
          const error = err as Error;
          if (error.name === "AbortError") {
            toast.error("Transcription timed out — try a shorter recording");
          } else if (error.message.includes("Failed to fetch") || error.message.includes("NetworkError")) {
            toast.error("Network error — check your connection and try again");
          } else {
            toast.error(`Voice transcription failed: ${error.message}`);
          }
        } finally {
          setTranscribing(false);
        }
      };

      recorder.start();
      mediaRecorderRef.current = recorder;
      setRecording(true);
    } catch (err) {
      const error = err as Error;
      if (error.name === "NotAllowedError") {
        toast.error("Microphone permission denied — allow access in browser settings");
      } else if (error.name === "NotFoundError") {
        toast.error("No microphone found — connect a microphone and try again");
      } else if (error.name === "NotReadableError") {
        toast.error("Microphone is in use by another application");
      } else {
        toast.error(`Microphone error: ${error.message}`);
      }
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current && mediaRecorderRef.current.state !== "inactive") {
      mediaRecorderRef.current.stop();
    }
    setRecording(false);
  };

  const canSend = (text.trim() || imageBase64) && !disabled;

  return (
    <div
      className="border-t border-nova-border bg-nova-surface/80 backdrop-blur-md px-4 py-3"
      onDrop={handleDrop}
      onDragOver={handleDragOver}
    >
      {/* Image preview */}
      {imagePreview && (
        <div className="mb-2.5 flex items-center gap-3 animate-scale-in">
          <div className="relative">
            <img
              src={imagePreview}
              alt="Upload preview"
              className="h-16 w-16 rounded-lg border border-nova-border object-cover shadow-[var(--shadow-nova-sm)]"
            />
            <button
              onClick={clearImage}
              className="absolute -right-1.5 -top-1.5 flex h-5 w-5 items-center justify-center rounded-full bg-nova-error text-white hover:bg-red-500 transition-colors shadow-sm"
            >
              <X size={10} />
            </button>
          </div>
          <span className="text-xs text-nova-text-dim">Image attached</span>
        </div>
      )}

      <div className={`flex items-end gap-2 rounded-xl border bg-nova-bg px-2 py-1.5 transition-all duration-300 ${
        focused
          ? "border-nova-accent/40 shadow-[var(--shadow-nova-glow)]"
          : "border-nova-border hover:border-nova-border-bright"
      }`}>
        {/* Image upload button */}
        <button
          onClick={() => fileInputRef.current?.click()}
          disabled={streaming || disabled}
          className="rounded-lg p-2 text-nova-text-dim hover:bg-nova-surface-elevated hover:text-nova-text disabled:opacity-40 transition-all"
          title="Attach image"
          aria-label="Attach image"
        >
          <ImagePlus size={18} />
        </button>
        <input
          ref={fileInputRef}
          type="file"
          accept="image/*"
          className="hidden"
          onChange={(e) => {
            const file = e.target.files?.[0];
            if (file) handleFile(file);
          }}
        />

        <textarea
          ref={textareaRef}
          value={text}
          onChange={(e) => setText(e.target.value)}
          onKeyDown={handleKeyDown}
          onPaste={handlePaste}
          onFocus={() => setFocused(true)}
          onBlur={() => setFocused(false)}
          placeholder={transcribing ? "Transcribing..." : imageBase64 ? "Describe what you want to know..." : "Message Nova_..."}
          rows={1}
          className="flex-1 resize-none bg-transparent px-1 py-2 text-sm text-nova-text placeholder:text-nova-text-dim/60 outline-none"
          style={{ maxHeight: 150 }}
          onInput={(e) => {
            const el = e.currentTarget;
            el.style.height = "auto";
            el.style.height = Math.min(el.scrollHeight, 150) + "px";
          }}
        />

        {/* Mic button (only shown when voice is enabled) */}
        {voiceEnabled && (
          <button
            onClick={recording ? stopRecording : startRecording}
            disabled={streaming || disabled || transcribing}
            className={`rounded-lg p-2 transition-all disabled:opacity-40 ${
              recording
                ? "bg-nova-error/20 text-nova-error animate-pulse"
                : "text-nova-text-dim hover:bg-nova-surface-elevated hover:text-nova-text"
            }`}
            title={recording ? "Stop recording" : transcribing ? "Transcribing..." : "Start voice recording"}
            aria-label={recording ? "Stop recording" : "Start voice recording"}
          >
            {recording ? (
              <MicOff size={18} />
            ) : (
              <Mic size={18} />
            )}
          </button>
        )}

        {streaming ? (
          <button
            onClick={onStop}
            className="flex items-center justify-center rounded-lg bg-nova-error/90 p-2 text-white hover:bg-nova-error transition-colors"
            title="Stop generating"
          >
            <Square size={16} />
          </button>
        ) : (
          <button
            onClick={submit}
            disabled={!canSend}
            className={`flex items-center justify-center rounded-lg p-2 text-white transition-all duration-200 ${
              canSend
                ? "bg-nova-accent hover:bg-nova-accent-hover shadow-[var(--shadow-nova-glow)]"
                : "bg-nova-border text-nova-text-dim opacity-50"
            }`}
            title="Send message"
          >
            <ArrowUp size={16} strokeWidth={2.5} />
          </button>
        )}
      </div>
    </div>
  );
}
