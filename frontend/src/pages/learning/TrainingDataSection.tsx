import { GraduationCap, Download, Play, Loader2 } from "lucide-react";
import { Card, Button, EmptyState, Skeleton } from "../../components/ui";
import { formatDate } from "../../lib/utils";
import type { TrainingDataEntry, FinetuneStatus, FinetuneHistoryRun, TrainingDataStatsDetailed } from "../../lib/types";

interface Props {
  trainingData: TrainingDataEntry[];
  trainingStats: TrainingDataStatsDetailed | null;
  finetuneStatus: FinetuneStatus | null;
  finetuneHistory: FinetuneHistoryRun[];
  loading: boolean;
  finetuneRunning: boolean;
  hasMore: boolean;
  onLoadMore: () => void;
  onExport: () => void;
  onTriggerFinetune: () => void;
}

export default function TrainingDataSection({
  trainingData,
  trainingStats,
  finetuneStatus,
  finetuneHistory,
  loading,
  finetuneRunning,
  hasMore,
  onLoadMore,
  onExport,
  onTriggerFinetune,
}: Props) {
  if (loading && trainingData.length === 0) {
    return <Skeleton lines={4} />;
  }

  return (
    <section>
      {/* Training Data Stats */}
      {trainingStats && trainingStats.total_pairs > 0 && (
        <Card className="mb-4">
          <h3 className="mb-2 text-xs font-medium text-nova-text-dim">Training Data Stats</h3>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3 text-sm">
            <div>
              <span className="text-xs text-nova-text-dim">Total Pairs</span>
              <p className="font-medium">{trainingStats.total_pairs}</p>
            </div>
            <div>
              <span className="text-xs text-nova-text-dim">Valid Pairs</span>
              <p className="font-medium">{trainingStats.valid_pairs}</p>
            </div>
            <div>
              <span className="text-xs text-nova-text-dim">Avg Chosen Length</span>
              <p className="font-medium">{trainingStats.avg_chosen_length} chars</p>
            </div>
            <div>
              <span className="text-xs text-nova-text-dim">Avg Rejected Length</span>
              <p className="font-medium">{trainingStats.avg_rejected_length} chars</p>
            </div>
          </div>
          {trainingStats.date_range && trainingStats.date_range.earliest && (
            <div className="mt-2 text-xs text-nova-text-dim">
              Date range: {formatDate(trainingStats.date_range.earliest)} — {trainingStats.date_range.latest ? formatDate(trainingStats.date_range.latest) : "now"}
            </div>
          )}
        </Card>
      )}

      {/* Fine-tune controls */}
      {finetuneStatus && (
        <div className="mb-4 flex flex-wrap items-center gap-4 rounded-lg border border-nova-border p-3">
          <div>
            <span className="text-sm font-medium">{finetuneStatus.training_pairs} pairs</span>
            <span className="ml-2 text-xs text-nova-text-dim">
              (min: {finetuneStatus.min_required})
            </span>
          </div>
          <span
            className={`rounded px-2 py-0.5 text-xs font-medium ${
              finetuneStatus.ready
                ? "bg-nova-success/20 text-nova-success"
                : "bg-nova-error/20 text-nova-error"
            }`}
          >
            {finetuneStatus.ready ? "Ready for fine-tuning" : "Not enough data"}
          </span>
          {finetuneRunning && (
            <span className="flex items-center gap-1.5 rounded px-2 py-0.5 text-xs font-medium bg-nova-warning/20 text-nova-warning">
              <Loader2 size={12} className="animate-spin" />
              Running...
            </span>
          )}
          <div className="ml-auto flex gap-2">
            <Button
              size="sm"
              variant="secondary"
              onClick={onTriggerFinetune}
              disabled={!finetuneStatus.ready || finetuneRunning}
              loading={finetuneRunning}
              icon={<Play size={14} />}
            >
              Start Fine-Tuning
            </Button>
            <Button
              size="sm"
              onClick={onExport}
              disabled={trainingData.length === 0}
              icon={<Download size={14} />}
            >
              Export DPO
            </Button>
          </div>
        </div>
      )}

      {/* Training data entries */}
      {trainingData.length === 0 ? (
        <EmptyState
          icon={<GraduationCap size={40} strokeWidth={1.5} />}
          title="No training pairs yet."
          description="Correct Nova to generate them."
        />
      ) : (
        <>
          <div className="space-y-2">
            {trainingData.map((entry, i) => (
              <Card key={i}>
                <div className="mb-1 flex items-center justify-between">
                  <span className="font-medium text-sm">Q: {entry.query}</span>
                  {entry.created_at && (
                    <span className="text-xs text-nova-text-dim">{formatDate(entry.created_at)}</span>
                  )}
                </div>
                <div className="grid grid-cols-2 gap-2 text-xs">
                  <div>
                    <span className="text-nova-success">Chosen:</span>
                    <p className="text-nova-text-dim">{entry.chosen.slice(0, 150)}</p>
                  </div>
                  <div>
                    <span className="text-nova-error">Rejected:</span>
                    <p className="text-nova-text-dim">{entry.rejected.slice(0, 150)}</p>
                  </div>
                </div>
              </Card>
            ))}
          </div>

          {hasMore && (
            <div className="mt-4 flex justify-center">
              <Button
                variant="secondary"
                size="sm"
                onClick={onLoadMore}
                loading={loading}
              >
                Load More
              </Button>
            </div>
          )}
        </>
      )}

      {/* Finetune History */}
      {finetuneHistory.length > 0 && (
        <div className="mt-8">
          <h3 className="mb-3 text-sm font-medium text-nova-text-dim">
            Fine-Tune History ({finetuneHistory.length})
          </h3>
          <div className="overflow-x-auto rounded-lg border border-nova-border">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-nova-border bg-nova-surface text-left text-xs text-nova-text-dim">
                  <th className="px-3 py-2">Date</th>
                  <th className="px-3 py-2">Status</th>
                  <th className="px-3 py-2 text-center">Pairs</th>
                  <th className="px-3 py-2 text-center">New</th>
                  <th className="px-3 py-2">Eval Results</th>
                </tr>
              </thead>
              <tbody>
                {finetuneHistory.map((run, i) => (
                  <tr
                    key={i}
                    className="border-b border-nova-border last:border-0 hover:bg-nova-surface/50"
                  >
                    <td className="whitespace-nowrap px-3 py-2 text-nova-text-dim">
                      {run.started_at ? formatDate(run.started_at) : "\u2014"}
                    </td>
                    <td className="px-3 py-2">
                      <span className={`rounded px-1.5 py-0.5 text-[10px] font-medium ${
                        run.status === "deployed" || run.status === "success"
                          ? "bg-nova-success/20 text-nova-success"
                          : run.status === "rejected" || run.status === "failed"
                          ? "bg-nova-error/20 text-nova-error"
                          : "bg-nova-border text-nova-text-dim"
                      }`}>
                        {run.status || "unknown"}
                      </span>
                    </td>
                    <td className="px-3 py-2 text-center">{run.training_pairs}</td>
                    <td className="px-3 py-2 text-center">{run.new_pairs}</td>
                    <td className="px-3 py-2 text-xs text-nova-text-dim">
                      {run.eval ? (
                        <span>
                          Win rate: {run.eval.win_rate !== null ? `${Math.round(run.eval.win_rate * 100)}%` : "\u2014"}
                          {run.eval.candidate_is_better !== null && (
                            <span className={`ml-1.5 rounded px-1 py-0.5 text-[9px] font-medium ${
                              run.eval.candidate_is_better
                                ? "bg-nova-success/20 text-nova-success"
                                : "bg-nova-error/20 text-nova-error"
                            }`}>
                              {run.eval.candidate_is_better ? "better" : "worse"}
                            </span>
                          )}
                        </span>
                      ) : run.reason || "\u2014"}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </section>
  );
}
