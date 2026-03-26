import { useEffect, useState, useRef, useCallback } from "react";
import { FileText, Search, Upload, File, CheckCircle, AlertTriangle, ChevronDown, ChevronUp } from "lucide-react";
import { toast } from "sonner";
import {
  getDocuments,
  deleteDocument,
  ingestDocument,
  searchDocuments,
} from "../lib/api";
import { formatDate, formatFileSize } from "../lib/utils";
import type { DocumentInfo, DocumentSearchResult } from "../lib/types";
import {
  PageHeader,
  Card,
  Button,
  FormInput,
  FormTextarea,
  ConfirmDialog,
  EmptyState,
  Skeleton,
} from "../components/ui";

const ACCEPTED_EXTENSIONS = [".txt", ".md", ".json"];
const ACCEPTED_MIME_TYPES = [
  "text/plain",
  "text/markdown",
  "application/json",
];

function getFileExtension(name: string): string {
  const idx = name.lastIndexOf(".");
  return idx >= 0 ? name.slice(idx).toLowerCase() : "";
}

// MED-4: Expandable search result card
function SearchResultCard({ result }: { result: DocumentSearchResult }) {
  const [expanded, setExpanded] = useState(false);
  const isTruncated = result.content.length > 300;

  return (
    <Card>
      <div className="mb-1 flex items-center justify-between">
        <span className="font-medium text-sm">{result.title || result.source}</span>
        <span className="text-xs text-nova-text-dim">Score: {result.score.toFixed(3)}</span>
      </div>
      <p className="text-sm text-nova-text-dim">
        {expanded ? result.content : result.content.slice(0, 300)}
        {!expanded && isTruncated && "..."}
      </p>
      {isTruncated && (
        <button
          onClick={() => setExpanded(!expanded)}
          className="mt-1.5 flex items-center gap-1 text-xs text-nova-accent hover:text-nova-accent-hover transition-colors"
        >
          {expanded ? <ChevronUp size={12} /> : <ChevronDown size={12} />}
          {expanded ? "Show less" : `Show more (${result.content.length} chars)`}
        </button>
      )}
    </Card>
  );
}

export default function DocumentsPage() {
  const [documents, setDocuments] = useState<DocumentInfo[]>([]);
  const [loading, setLoading] = useState(true);
  const [confirmId, setConfirmId] = useState<number | null>(null);

  // Ingest form
  const [ingestMode, setIngestMode] = useState<"text" | "url" | "file">("text");
  const [ingestTitle, setIngestTitle] = useState("");
  const [ingestText, setIngestText] = useState("");
  const [ingestUrl, setIngestUrl] = useState("");
  const [ingesting, setIngesting] = useState(false);

  // File upload state
  const [dragActive, setDragActive] = useState(false);
  const [uploadFile, setUploadFile] = useState<File | null>(null);
  const [uploadStatus, setUploadStatus] = useState<"idle" | "reading" | "uploading" | "success" | "error">("idle");
  const [uploadError, setUploadError] = useState("");
  const [uploadChunks, setUploadChunks] = useState<number | null>(null);
  const fileDropRef = useRef<HTMLDivElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Search
  const [searchQuery, setSearchQuery] = useState("");
  const [searchResults, setSearchResults] = useState<DocumentSearchResult[] | null>(null);
  const [searching, setSearching] = useState(false);

  const refresh = () => {
    setLoading(true);
    getDocuments()
      .then((v) => setDocuments(Array.isArray(v) ? v : []))
      .catch(() => setDocuments([]))
      .finally(() => setLoading(false));
  };

  useEffect(() => {
    refresh();
  }, []);

  const handleDelete = async () => {
    if (confirmId === null) return;
    try {
      await deleteDocument(confirmId);
      setConfirmId(null);
      toast.success("Document deleted");
      refresh();
    } catch {
      toast.error("Failed to delete document");
      setConfirmId(null);
    }
  };

  const handleIngest = async () => {
    setIngesting(true);
    try {
      const data: { text?: string; url?: string; title?: string } = {};
      if (ingestTitle) data.title = ingestTitle;
      if (ingestMode === "text") {
        data.text = ingestText;
      } else {
        data.url = ingestUrl;
      }
      const res = await ingestDocument(data);
      toast.success(`Ingested (ID: ${res.document_id})`);
      setIngestText("");
      setIngestUrl("");
      setIngestTitle("");
      refresh();
    } catch (e) {
      toast.error(`Ingest failed: ${e}`);
    } finally {
      setIngesting(false);
    }
  };

  // ── File upload handlers ──

  const validateFile = useCallback((file: File): string | null => {
    const ext = getFileExtension(file.name);
    if (!ACCEPTED_EXTENSIONS.includes(ext) && !ACCEPTED_MIME_TYPES.includes(file.type)) {
      return `Unsupported file type. Accepted: ${ACCEPTED_EXTENSIONS.join(", ")}`;
    }
    if (file.size > 10 * 1024 * 1024) {
      return "File too large. Max 10MB.";
    }
    return null;
  }, []);

  const processFile = useCallback(async (file: File) => {
    const error = validateFile(file);
    if (error) {
      setUploadError(error);
      setUploadStatus("error");
      setUploadFile(file);
      return;
    }

    setUploadFile(file);
    setUploadError("");
    setUploadChunks(null);
    setUploadStatus("reading");

    try {
      const fileText = await file.text();
      setUploadStatus("uploading");

      const title = file.name.replace(/\.[^.]+$/, "");
      const res = await ingestDocument({ text: fileText, title });
      // MED-9: Fetch chunk count from refreshed documents
      const docs = await getDocuments();
      const ingested = Array.isArray(docs) ? docs.find((d) => d.id === res.document_id) : null;
      setUploadChunks(ingested?.chunk_count ?? null);
      setUploadStatus("success");
      toast.success(`File ingested (ID: ${res.document_id})`);
      setDocuments(Array.isArray(docs) ? docs : []);

      // Auto-reset after success
      setTimeout(() => {
        setUploadFile(null);
        setUploadStatus("idle");
        setUploadChunks(null);
      }, 3000);
    } catch (e) {
      setUploadError(`Upload failed: ${e}`);
      setUploadStatus("error");
    }
  }, [validateFile]);

  const handleFileDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    const file = e.dataTransfer?.files?.[0];
    if (file) processFile(file);
  }, [processFile]);

  const handleFileDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(true);
  }, []);

  const handleFileDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
  }, []);

  const handleFileSelect = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) processFile(file);
    if (fileInputRef.current) fileInputRef.current.value = "";
  }, [processFile]);

  const handleSearch = async () => {
    if (!searchQuery.trim()) {
      setSearchResults(null);
      return;
    }
    setSearching(true);
    try {
      const results = await searchDocuments(searchQuery);
      setSearchResults(Array.isArray(results) ? results : []);
    } catch {
      setSearchResults([]);
    } finally {
      setSearching(false);
    }
  };

  return (
    <div className="flex-1 overflow-y-auto">
      <div className="mx-auto max-w-4xl w-full px-4 md:px-6 py-6">
        <PageHeader icon={<FileText size={22} />} title="Documents" />

        {/* Ingest form */}
        <Card className="mb-8">
          <h2 className="mb-3 text-sm font-medium text-nova-text-dim">Ingest Document</h2>
          <div className="mb-3 flex gap-2">
            <Button
              size="sm"
              variant={ingestMode === "text" ? "primary" : "secondary"}
              onClick={() => setIngestMode("text")}
            >
              Text
            </Button>
            <Button
              size="sm"
              variant={ingestMode === "url" ? "primary" : "secondary"}
              onClick={() => setIngestMode("url")}
            >
              URL
            </Button>
            <Button
              size="sm"
              variant={ingestMode === "file" ? "primary" : "secondary"}
              onClick={() => setIngestMode("file")}
              icon={<Upload size={14} />}
            >
              File
            </Button>
          </div>

          {ingestMode === "file" ? (
            /* File upload drop zone */
            <div className="space-y-3">
              <div
                ref={fileDropRef}
                onDrop={handleFileDrop}
                onDragOver={handleFileDragOver}
                onDragLeave={handleFileDragLeave}
                onClick={() => fileInputRef.current?.click()}
                className={`flex cursor-pointer flex-col items-center justify-center rounded-lg border-2 border-dashed p-8 transition-all ${
                  dragActive
                    ? "border-nova-accent bg-nova-accent/5"
                    : uploadStatus === "error"
                      ? "border-nova-error/50 bg-nova-error/5"
                      : uploadStatus === "success"
                        ? "border-nova-success/50 bg-nova-success/5"
                        : "border-nova-border hover:border-nova-accent/40 hover:bg-nova-surface/50"
                }`}
              >
                {uploadStatus === "idle" && (
                  <>
                    <Upload size={32} className="mb-2 text-nova-text-dim" />
                    <p className="text-sm text-nova-text-dim">
                      Drag & drop a file here, or click to select
                    </p>
                    <p className="mt-1 text-xs text-nova-text-dim/60">
                      Accepts: .txt, .md, .json (max 10MB)
                    </p>
                  </>
                )}

                {(uploadStatus === "reading" || uploadStatus === "uploading") && (
                  <>
                    <div className="mb-2 h-6 w-6 animate-spin rounded-full border-2 border-nova-accent border-t-transparent" />
                    <p className="text-sm text-nova-text-dim">
                      {uploadStatus === "reading"
                        ? `Reading file${uploadFile ? ` (${formatFileSize(uploadFile.size)})` : ""}...`
                        : "Uploading to Nova..."}
                    </p>
                    {uploadFile && (
                      <p className="mt-1 flex items-center gap-1.5 text-xs text-nova-text-dim/60">
                        <File size={12} />
                        {uploadFile.name}
                      </p>
                    )}
                  </>
                )}

                {uploadStatus === "success" && (
                  <>
                    <CheckCircle size={32} className="mb-2 text-nova-success" />
                    <p className="text-sm text-nova-success">
                      Ingested{uploadChunks !== null ? ` (${uploadChunks} chunk${uploadChunks !== 1 ? "s" : ""})` : " successfully"}
                    </p>
                    {uploadFile && (
                      <p className="mt-1 flex items-center gap-1.5 text-xs text-nova-text-dim/60">
                        <File size={12} />
                        {uploadFile.name} ({formatFileSize(uploadFile.size)})
                      </p>
                    )}
                  </>
                )}

                {uploadStatus === "error" && (
                  <>
                    <AlertTriangle size={32} className="mb-2 text-nova-error" />
                    <p className="text-sm text-nova-error">{uploadError}</p>
                    {uploadFile && (
                      <p className="mt-1 flex items-center gap-1.5 text-xs text-nova-text-dim/60">
                        <File size={12} />
                        {uploadFile.name}
                      </p>
                    )}
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        setUploadFile(null);
                        setUploadStatus("idle");
                        setUploadError("");
                      }}
                      className="mt-2 text-xs text-nova-accent hover:underline"
                    >
                      Try again
                    </button>
                  </>
                )}

                <input
                  ref={fileInputRef}
                  type="file"
                  accept={ACCEPTED_EXTENSIONS.join(",")}
                  className="hidden"
                  onChange={handleFileSelect}
                />
              </div>
            </div>
          ) : (
            /* Text / URL modes */
            <div className="space-y-2">
              <FormInput
                value={ingestTitle}
                onChange={(e) => setIngestTitle(e.target.value)}
                placeholder="Title (optional)"
              />
              {ingestMode === "text" ? (
                <FormTextarea
                  value={ingestText}
                  onChange={(e) => setIngestText(e.target.value)}
                  placeholder="Paste text content here..."
                  rows={4}
                />
              ) : (
                <FormInput
                  type="url"
                  value={ingestUrl}
                  onChange={(e) => setIngestUrl(e.target.value)}
                  placeholder="https://example.com/article"
                />
              )}
              <Button
                onClick={handleIngest}
                loading={ingesting}
                disabled={ingestMode === "text" ? !ingestText.trim() : !ingestUrl.trim()}
              >
                Ingest
              </Button>
            </div>
          )}
        </Card>

        {/* Search */}
        <section className="mb-8">
          <h2 className="mb-3 text-sm font-medium text-nova-text-dim">Search Documents</h2>
          <div className="flex gap-2">
            <FormInput
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && handleSearch()}
              placeholder="Search ingested documents..."
              icon={<Search size={14} />}
              className="flex-1"
            />
            <Button onClick={handleSearch} loading={searching}>
              Search
            </Button>
          </div>
          {searchResults !== null && (
            <div className="mt-3 space-y-2">
              {searchResults.length === 0 ? (
                <p className="text-sm text-nova-text-dim">No results found.</p>
              ) : (
                searchResults.map((r, i) => (
                  <SearchResultCard key={i} result={r} />
                ))
              )}
            </div>
          )}
        </section>

        {/* Document list */}
        <section>
          <h2 className="mb-3 text-sm font-medium text-nova-text-dim">
            All Documents ({documents.length})
          </h2>
          {loading ? (
            <Skeleton lines={4} />
          ) : documents.length === 0 ? (
            <EmptyState
              icon={<FileText size={40} strokeWidth={1.5} />}
              title="No documents ingested yet."
              description="Upload text files, paste content, or provide a URL above. Documents are chunked, embedded, and searchable."
            />
          ) : (
            <div className="overflow-x-auto rounded-lg border border-nova-border">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b border-nova-border bg-nova-surface text-left text-xs text-nova-text-dim">
                    <th className="px-3 py-2">Title</th>
                    <th className="px-3 py-2">Source</th>
                    <th className="px-3 py-2 text-center">Chunks</th>
                    <th className="px-3 py-2">Created</th>
                    <th className="px-3 py-2 text-right">Actions</th>
                  </tr>
                </thead>
                <tbody>
                  {documents.map((d) => (
                    <tr
                      key={d.id}
                      className="border-b border-nova-border last:border-0 hover:bg-nova-surface/50"
                    >
                      <td className="max-w-[200px] truncate px-3 py-2 font-medium">{d.title || "Untitled"}</td>
                      <td className="max-w-[200px] truncate px-3 py-2 text-nova-text-dim">{d.source}</td>
                      <td className="px-3 py-2 text-center">{d.chunk_count}</td>
                      <td className="px-3 py-2 text-nova-text-dim">{formatDate(d.created_at)}</td>
                      <td className="px-3 py-2 text-right">
                        <Button variant="ghost" size="sm" onClick={() => setConfirmId(d.id)}>
                          Delete
                        </Button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </section>

        {confirmId !== null && (
          <ConfirmDialog
            message="Delete this document and all its chunks? This cannot be undone."
            onConfirm={handleDelete}
            onCancel={() => setConfirmId(null)}
          />
        )}
      </div>
    </div>
  );
}
