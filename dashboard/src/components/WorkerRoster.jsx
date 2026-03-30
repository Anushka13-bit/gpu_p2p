import { fmtPct, fmtAcc, shortId, formatAgo, gpuLabel } from '../utils/fmt';
import styles from './WorkerRoster.module.css';

function LiveBar({ pct, color = '#3b82f6' }) {
  return (
    <div className={styles.liveBarTrack}>
      <div
        className={styles.liveBarFill}
        style={{ width: `${Math.min(100, pct || 0)}%`, background: color }}
      />
    </div>
  );
}

export default function WorkerRoster({ nodes }) {
  if (!nodes || nodes.length === 0) {
    return <p style={{ color: 'var(--text-muted)', fontSize: 13 }}>No workers registered.</p>;
  }

  return (
    <div className={styles.grid}>
      {nodes.map((w) => {
        const alive = w.alive;
        const hasTask = !!w.current_shard;
        return (
          <div key={w.worker_id} className={styles.card} data-alive={alive}>
            {/* Header */}
            <div className={styles.cardHeader}>
              <div className={styles.workerIdRow}>
                <span className={styles.statusDot} data-alive={alive} />
                <span className={styles.workerId}>{shortId(w.worker_id)}</span>
              </div>
              <span className={styles.hostLabel}>{w.host_label || '—'}</span>
            </div>

            {/* Specs */}
            <div className={styles.specs}>
              <span className={styles.spec}>
                <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <rect x="2" y="3" width="20" height="14" rx="2"/><path d="M8 21h8M12 17v4"/>
                </svg>
                {gpuLabel(w.gpu_vram_mb)}
              </span>
              <span className={styles.spec}>
                <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <rect x="4" y="4" width="16" height="16" rx="2"/><rect x="9" y="9" width="6" height="6"/>
                  <path d="M15 2v2M9 2v2M15 20v2M9 20v2M2 15h2M2 9h2M20 15h2M20 9h2"/>
                </svg>
                {w.cpu_count} CPUs
              </span>
              <span className={styles.spec} style={{ color: alive ? '#10b981' : '#ef4444' }}>
                {formatAgo(w.last_seen_age_sec)}
              </span>
            </div>

            {/* Task info */}
            {hasTask ? (
              <div className={styles.taskSection}>
                <div className={styles.taskHeader}>
                  <span className={styles.shardTag}>{w.current_shard}</span>
                  {w.live_epoch != null && (
                    <span className={styles.epochTag}>
                      epoch {w.live_epoch}/{w.live_epoch_total}
                    </span>
                  )}
                </div>
                <LiveBar
                  pct={w.live_progress_pct ?? w.current_shard_progress_pct}
                  color="#3b82f6"
                />
                <div className={styles.taskMeta}>
                  <span>{fmtPct(w.live_progress_pct ?? w.current_shard_progress_pct)}</span>
                  {w.live_train_acc != null && (
                    <span style={{ color: '#10b981' }}>acc {fmtAcc(w.live_train_acc)}</span>
                  )}
                  {w.current_shard_eval_acc != null && (
                    <span style={{ color: '#6366f1' }}>eval {fmtAcc(w.current_shard_eval_acc)}</span>
                  )}
                </div>
              </div>
            ) : (
              <div className={styles.idle}>
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <circle cx="12" cy="12" r="10"/><path d="M10 15l5-3-5-3v6z"/>
                </svg>
                Idle
              </div>
            )}
          </div>
        );
      })}
    </div>
  );
}
