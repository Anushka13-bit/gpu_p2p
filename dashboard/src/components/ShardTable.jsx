import { statusColor, fmtPct, fmtAcc, shortId } from '../utils/fmt';
import styles from './ShardTable.module.css';

/**
 * Progress bar driven by registry_snapshot's progress_pct,
 * which the scheduler computes as:
 *   100 * (min(last_reported_index, image_end-1) - image_start + 1) / (image_end - image_start)
 */
function ProgressBar({ pct, color }) {
  const safePct = Math.min(100, Math.max(0, pct ?? 0));
  return (
    <div className={styles.barTrack}>
      <div
        className={styles.barFill}
        style={{
          width: `${safePct}%`,
          background: `linear-gradient(90deg, ${color}99, ${color})`,
          boxShadow: `0 0 8px ${color}60`,
        }}
      />
    </div>
  );
}

export default function ShardTable({ taskTable }) {
  if (!taskTable || Object.keys(taskTable).length === 0) return null;

  return (
    <div className={styles.wrapper}>
      <div className={styles.tableHead}>
        <span>Shard</span>
        <span>Status</span>
        <span>Worker</span>
        <span>Range</span>
        <span>Progress</span>
        <span>Last Idx</span>
        <span>Eval Acc</span>
        <span>Epochs</span>
      </div>
      {Object.entries(taskTable).map(([tid, info]) => {
        const col = statusColor(info.status);
        const rangeLabel = Array.isArray(info.range) && info.range.length === 2
          ? `${info.range[0]}–${info.range[1]}`
          : '—';
        return (
          <div key={tid} className={styles.row}>
            {/* Shard ID */}
            <span className={styles.shardId}>{tid}</span>

            {/* Status badge */}
            <span>
              <span className={styles.badge} style={{ color: col.fg, background: col.bg }}>
                <span className={styles.dot} style={{ background: col.dot }} />
                {info.status}
              </span>
            </span>

            {/* Assigned worker (short UUID) */}
            <span className={styles.mono}>
              {info.worker ? shortId(info.worker) : '—'}
            </span>

            {/* Image index range from TaskShard */}
            <span className={styles.mono} style={{ fontSize: 10, color: 'var(--text-muted)' }}>
              {rangeLabel}
            </span>

            {/* Progress bar — value comes from scheduler.registry_snapshot() */}
            <span className={styles.progressCell}>
              <ProgressBar pct={info.progress_pct} color={col.dot} />
              <span className={styles.pctLabel}>{fmtPct(info.progress_pct)}</span>
            </span>

            {/* Last reported index (last_index in registry_snapshot/health_snapshot) */}
            <span className={styles.mono} style={{ fontSize: 10 }}>
              {info.last_index != null && info.last_index >= 0 ? info.last_index : '—'}
            </span>

            {/* eval_acc — set by submit_weights after shard_complete */}
            <span style={{ color: info.eval_acc != null ? '#10b981' : 'var(--text-muted)', fontSize: 12 }}>
              {fmtAcc(info.eval_acc)}
            </span>

            {/* epochs string — e.g. "2/5" (from last_epochs_completed/last_epochs_planned) */}
            <span className={styles.mono}>{info.epochs ?? '—'}</span>
          </div>
        );
      })}
    </div>
  );
}
