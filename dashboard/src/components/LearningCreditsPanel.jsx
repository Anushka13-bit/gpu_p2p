import { shortId } from '../utils/fmt';

/**
 * Proof-of-Learning: leaderboard + recent events from tracker ``learning_credits`` JSON.
 * Read-only display; no client-side scoring.
 */
export default function LearningCreditsPanel({ learningCredits }) {
  if (!learningCredits || typeof learningCredits !== 'object') {
    return (
      <p style={{ color: 'var(--text-muted)', fontSize: 13 }}>
        No credit data yet (tracker may be older build or no submits).
      </p>
    );
  }

  const board = Array.isArray(learningCredits.leaderboard) ? learningCredits.leaderboard : [];
  const events = Array.isArray(learningCredits.recent_events) ? learningCredits.recent_events : [];

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>
      <div style={{ overflowX: 'auto' }}>
        <table className="history-table">
          <thead>
            <tr>
              <th>#</th>
              <th>Worker</th>
              <th>Host</th>
              <th>Credits</th>
              <th>Reputation</th>
              <th>Streak</th>
              <th>Events</th>
            </tr>
          </thead>
          <tbody>
            {board.length === 0 ? (
              <tr>
                <td colSpan={7} style={{ color: 'var(--text-muted)', padding: '12px 14px' }}>
                  No workers on leaderboard yet.
                </td>
              </tr>
            ) : (
              board.map((row, i) => (
                <tr key={row.worker_id || i}>
                  <td className="mono">{i + 1}</td>
                  <td className="mono">{shortId(row.worker_id)}</td>
                  <td style={{ fontSize: 11, color: 'var(--text-muted)' }}>{row.host_label || '—'}</td>
                  <td style={{ color: '#f59e0b', fontWeight: 600 }}>{Number(row.credits_total ?? 0).toFixed(2)}</td>
                  <td style={{ color: '#6366f1' }}>{Number(row.reputation ?? 50).toFixed(1)}</td>
                  <td>{row.positive_streak_rounds ?? 0}</td>
                  <td className="mono">{row.credit_events_count ?? 0}</td>
                </tr>
              ))
            )}
          </tbody>
        </table>
      </div>

      <div>
        <p style={{ fontSize: 11, fontWeight: 600, color: 'var(--text-muted)', marginBottom: 8, letterSpacing: '0.06em' }}>
          RECENT CREDIT EVENTS
        </p>
        <div style={{ overflowX: 'auto', maxHeight: 220, overflowY: 'auto' }}>
          <table className="history-table">
            <thead>
              <tr>
                <th>Time</th>
                <th>Worker</th>
                <th>Phase</th>
                <th>Credits</th>
              </tr>
            </thead>
            <tbody>
              {events.length === 0 ? (
                <tr>
                  <td colSpan={4} style={{ color: 'var(--text-muted)', padding: '12px 14px' }}>
                    No events yet.
                  </td>
                </tr>
              ) : (
                [...events].reverse().slice(0, 25).map((ev, i) => {
                  const ts = ev.ts;
                  const tlabel = ts != null ? new Date(ts * 1000).toLocaleTimeString() : '—';
                  return (
                    <tr key={`${ev.worker_id}-${i}-${ts}`} style={{ opacity: Math.max(0.5, 1 - i * 0.02) }}>
                      <td className="mono">{tlabel}</td>
                      <td className="mono">{shortId(ev.worker_id)}</td>
                      <td style={{ fontSize: 11 }}>{ev.phase ?? '—'}</td>
                      <td style={{
                        color: (ev.credits ?? 0) >= 0 ? '#10b981' : '#ef4444',
                        fontWeight: 600,
                      }}>
                        {ev.credits != null ? Number(ev.credits).toFixed(3) : '—'}
                      </td>
                    </tr>
                  );
                })
              )}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
