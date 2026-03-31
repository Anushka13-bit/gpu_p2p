import './App.css';
import { useTrackerData } from './hooks/useTrackerData';
import StatCard from './components/StatCard';
import SectionCard from './components/SectionCard';
import ShardTable from './components/ShardTable';
import WorkerRoster from './components/WorkerRoster';
import LearningCreditsPanel from './components/LearningCreditsPanel';
import {
  ActiveNodesChart,
  TaskStatusChart,
  DataShardBar,
} from './components/Charts';
import { fmtAcc, shortId } from './utils/fmt';

/* ── Inline SVG icons ──────────────────────────────────────────────────────── */
const Icon = {
  cpu: <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><rect x="4" y="4" width="16" height="16" rx="2" /><rect x="9" y="9" width="6" height="6" /><path d="M15 2v2M9 2v2M15 20v2M9 20v2M2 15h2M2 9h2M20 15h2M20 9h2" /></svg>,
  nodes: <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><circle cx="5" cy="12" r="2" /><circle cx="19" cy="5" r="2" /><circle cx="19" cy="19" r="2" /><path d="M7 12h10M17 6l-2 6M17 18l-2-6" /></svg>,
  task: <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M9 11l3 3L22 4" /><path d="M21 12v7a2 2 0 01-2 2H5a2 2 0 01-2-2V5a2 2 0 012-2h11" /></svg>,
  round: <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><polyline points="23 4 23 10 17 10" /><path d="M20.49 15a9 9 0 11-2.12-9.36L23 10" /></svg>,
  brain: <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M9.5 2a5.5 5.5 0 015.5 5.5c0 .177-.009.352-.025.525A4.5 4.5 0 0119 12.5v.5a5 5 0 01-5 5H10a5 5 0 01-5-5v-.5a4.5 4.5 0 014.025-4.475A5.5 5.5 0 019.5 2Z" /></svg>,
  chart: <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12" /></svg>,
  history: <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><circle cx="12" cy="12" r="10" /><polyline points="12 6 12 12 16 14" /></svg>,
  workers: <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M17 21v-2a4 4 0 00-4-4H5a4 4 0 00-4 4v2" /><circle cx="9" cy="7" r="4" /><path d="M23 21v-2a4 4 0 00-3-3.87M16 3.13a4 4 0 010 7.75" /></svg>,
  shard: <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2" /></svg>,
  stop: <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><rect x="4" y="4" width="16" height="16" rx="2" /></svg>,
  offline: <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><line x1="1" y1="1" x2="23" y2="23" /><path d="M16.72 11.06A10.94 10.94 0 0119 12.55M5 12.55a10.94 10.94 0 015.17-2.39M10.71 5.05A16 16 0 0122.56 9M1.42 9a15.91 15.91 0 014.7-2.88M8.53 16.11a6 6 0 016.95 0M12 20h.01" /></svg>,
};

/* ── Connection status badge ───────────────────────────────────────────────── */
function ConnectionBadge({ status, lastUpdated }) {
  const map = {
    online: { color: '#10b981', label: 'Tracker Online', pulse: true },
    offline: { color: '#ef4444', label: 'Tracker Offline', pulse: false },
    connecting: { color: '#6366f1', label: 'Connecting…', pulse: true },
  };
  const { color, label, pulse } = map[status] ?? map.connecting;
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
      <span style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: 12, color }}>
        <span
          style={{ width: 8, height: 8, background: color, borderRadius: '50%', boxShadow: `0 0 6px ${color}`, flexShrink: 0 }}
          className={pulse ? 'animate-pulse' : ''}
        />
        {label}
      </span>
      {lastUpdated && (
        <span style={{ fontSize: 11, color: 'var(--text-muted)' }}>
          {lastUpdated.toLocaleTimeString()}
        </span>
      )}
    </div>
  );
}

/* ── Policy pill (only shown when tracker online) ─────────────────────────── */
function PolicyPill({ label, value }) {
  if (value == null) return null;
  return (
    <span style={{
      display: 'inline-flex', alignItems: 'center', gap: 4,
      background: 'rgba(99,102,241,0.1)', border: '1px solid rgba(99,102,241,0.2)',
      borderRadius: 99, padding: '2px 10px', fontSize: 11, color: '#94a3b8',
    }}>
      <span style={{ fontWeight: 600, color: '#6366f1' }}>{label}</span>
      <span>{value}</span>
    </span>
  );
}

/* ── Training-stopped banner ──────────────────────────────────────────────── */
function StoppedBanner({ reason }) {
  return (
    <div style={{
      background: 'rgba(239,68,68,0.08)', border: '1px solid rgba(239,68,68,0.25)',
      borderRadius: 12, padding: '10px 16px', display: 'flex', alignItems: 'center',
      gap: 10,
    }}>
      <span style={{ color: '#ef4444' }}>{Icon.stop}</span>
      <div>
        <p style={{ fontSize: 13, fontWeight: 600, color: '#ef4444' }}>Training Stopped</p>
        {reason && <p style={{ fontSize: 12, color: 'var(--text-muted)', marginTop: 2 }}>{reason}</p>}
      </div>
    </div>
  );
}

/* ── Offline overlay for charts/tables ───────────────────────────────────── */
function OfflinePlaceholder({ height = 180 }) {
  return (
    <div style={{
      height, display: 'flex', flexDirection: 'column', alignItems: 'center',
      justifyContent: 'center', gap: 10, color: 'var(--text-muted)',
    }}>
      <span style={{ opacity: 0.4 }}>{Icon.offline}</span>
      <p style={{ fontSize: 12 }}>Waiting for tracker…</p>
    </div>
  );
}

/* ── Main App ──────────────────────────────────────────────────────────────── */
export default function App() {
  const { snapshot, metricHistory, connectionStatus, lastUpdated } = useTrackerData();

  // snapshot is null when tracker is offline / connecting
  const online = connectionStatus === 'online' && snapshot != null;

  // Derived values — all null/0 when offline; real values from snapshot when online
  const taskTable = online ? snapshot.taskTable : {};
  const nodes = online ? snapshot.nodes : [];
  const nodeRegistry = online ? snapshot.nodeRegistry : {};
  const stopPolicy = online ? snapshot.stop_policy : {};
  const learningCredits = online ? snapshot.learningCredits : null;
  const polBoard = online && learningCredits && Array.isArray(learningCredits.leaderboard)
    ? learningCredits.leaderboard
    : [];
  const polTop = polBoard.length > 0 ? polBoard[0] : null;

  const totalShards = Object.keys(taskTable).length;
  const completedCount = Object.values(taskTable).filter(t => t.status === 'COMPLETED').length;
  const inProgressCount = Object.values(taskTable).filter(t => t.status === 'IN_PROGRESS').length;

  return (
    <div className="app-root">
      {/* ── Main ── */}
      <main className="main-content">
        {/* Top bar */}
        <header className="topbar">
          <div>
            <h1 className="page-title">Dashboard</h1>
            <p className="page-sub">Real-time federated learning monitor</p>
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: 10, flexWrap: 'wrap' }}>
            {/* Policy pills — only when tracker online */}
            {online && (
              <>
                <PolicyPill
                  label="max rounds"
                  value={stopPolicy.max_fed_rounds != null
                    ? (stopPolicy.max_fed_rounds === 0 ? '∞' : stopPolicy.max_fed_rounds)
                    : null}
                />
                <PolicyPill
                  label="patience"
                  value={stopPolicy.earlystop_patience != null
                    ? (stopPolicy.earlystop_patience === 0 ? 'off' : stopPolicy.earlystop_patience)
                    : null}
                />
                <PolicyPill
                  label="min δ"
                  value={stopPolicy.earlystop_min_delta != null
                    ? stopPolicy.earlystop_min_delta.toFixed(4)
                    : null}
                />
              </>
            )}
            <div style={{ width: 1, height: 24, background: 'var(--border)' }} />
            <ConnectionBadge status={connectionStatus} lastUpdated={lastUpdated} />
          </div>
        </header>

        <div className="content-scroll">
          {/* Training-stopped banner (only when tracker online and training has stopped) */}
          {online && snapshot.training_stopped && (
            <StoppedBanner reason={snapshot.stop_reason} />
          )}

          {/* ── KPI Stat Cards ── */}
          <div className="kpi-grid">
            {/* Active Workers — from node_registry in /health or /registry totals */}
            <StatCard
              loading={!online}
              icon={Icon.nodes}
              label="Active Workers"
              value={online
                ? `${nodeRegistry.active_nodes} / ${nodeRegistry.total_nodes}`
                : null}
              sub={online
                ? `Heartbeat ≤ ${nodeRegistry.heartbeat_timeout_sec}s • ${nodeRegistry.nodes_on_shard ?? 0} on shard`
                : null}
              accent="#3b82f6"
              badge="Cluster"
            />

            {/* Shards — from real task_table statuses */}
            <StatCard
              loading={!online}
              icon={Icon.task}
              label="Shards Complete"
              value={online
                ? `${completedCount} / ${totalShards}`
                : null}
              sub={online
                ? `${inProgressCount} in progress`
                : null}
              accent="#10b981"
              badge="Tasks"
            />

            {/* Best val acc — set by scheduler after FedAvg; null until first round */}
            <StatCard
              loading={!online}
              icon={Icon.brain}
              label="Best Val Acc"
              value={online ? fmtAcc(snapshot.best_val_acc) : null}
              sub={online && snapshot.rounds_without_improve != null
                ? `${snapshot.rounds_without_improve} rounds without improve`
                : null}
              accent="#6366f1"
              badge="Global"
            />

            {/* Last FedAvg metrics (val + test acc) — from state_manager snapshot */}
            <StatCard
              loading={!online}
              icon={Icon.cpu}
              label="Last Val / Test Acc"
              value={online
                ? `${fmtAcc(snapshot.last_val_acc)} / ${fmtAcc(snapshot.last_test_acc)}`
                : null}
              sub={online
                ? (snapshot.has_global_weights ? 'Global weights ready' : 'No global weights yet')
                : null}
              accent="#06b6d4"
              badge="Eval"
            />

            <StatCard
              loading={!online}
              icon={Icon.round}
              label="PoL leaderboard #1"
              value={online ? (polTop ? shortId(polTop.worker_id) : '—') : null}
              sub={online
                ? (polTop
                  ? `${Number(polTop.credits_total ?? 0).toFixed(2)} credits · rep ${Number(polTop.reputation ?? 50).toFixed(0)}`
                  : (learningCredits ? 'No credit grants yet' : 'No PoL payload from tracker'))
                : null}
              accent="#f59e0b"
              badge="PoL"
            />
          </div>

          {/* ── Proof-of-Learning (high in page so it is not missed) ── */}
          <SectionCard
            title="Proof-of-Learning Credits"
            subtitle="Leaderboard and recent events (GET /credits + tracker snapshots)"
            icon={Icon.round}
            accent="#f59e0b"
          >
            {online
              ? <LearningCreditsPanel learningCredits={learningCredits} />
              : <OfflinePlaceholder height={160} />}
          </SectionCard>

          {/* ── Charts row 1: Data Shard Progress ── */}
          <div className="charts-row two-col">
            <SectionCard
              title="Data Shard Progress"
              subtitle="One bar = 100% of dataset; segments fill as each worker eats its shard"
              icon={Icon.shard}
              accent="#ec4899"
            >
              {online && totalShards > 0
                ? <DataShardBar taskTable={taskTable} />
                : <OfflinePlaceholder height={220} />}
            </SectionCard>
          </div>

          {/* ── Charts row 2: Active Nodes + Task Status ── */}
          <div className="charts-row two-col">
            <SectionCard
              title="Active Nodes over Time"
              subtitle="Workers with heartbeat within timeout window"
              icon={Icon.nodes}
              accent="#3b82f6"
            >
              {online && metricHistory.length > 0
                ? <ActiveNodesChart data={metricHistory} />
                : <OfflinePlaceholder />}
            </SectionCard>

            <SectionCard
              title="Task Status Distribution"
              subtitle="Current breakdown of shard states from task_table"
              icon={Icon.task}
              accent="#10b981"
            >
              {online && totalShards > 0
                ? <TaskStatusChart taskTable={taskTable} />
                : <OfflinePlaceholder />}
            </SectionCard>
          </div>

          {/* ── Shard task table ── */}
          <SectionCard
            title="Task Shard Table"
            subtitle={online
              ? `${totalShards} shards × ${totalShards > 0 ? Object.values(taskTable)[0]?.range?.length > 0 ? `${Object.values(taskTable)[0]?.range?.[1] - Object.values(taskTable)[0]?.range?.[0]} images each` : '' : ''} — MNIST/FashionMNIST`
              : 'Waiting for tracker…'}
            icon={Icon.task}
            accent="#10b981"
          >
            {online && totalShards > 0
              ? <ShardTable taskTable={taskTable} />
              : (
                <OfflinePlaceholder height={140} />
              )}
          </SectionCard>

          {/* ── Worker registry ── */}
          <SectionCard
            title="Worker Registry"
            subtitle={online
              ? `${nodes.length} registered nodes — ${nodeRegistry.active_nodes ?? 0} active`
              : 'Waiting for tracker…'}
            icon={Icon.workers}
            accent="#3b82f6"
          >
            {online && nodes.length > 0
              ? <WorkerRoster nodes={nodes} />
              : (
                <OfflinePlaceholder height={160} />
              )}
          </SectionCard>

          {/* ── Metric history table ── */}
          <SectionCard
            title="Metric History"
            subtitle={`${metricHistory.length} live snapshots collected (online only)`}
            icon={Icon.history}
            accent="#6366f1"
          >
            {metricHistory.length > 0 ? (
              <div style={{ overflowX: 'auto' }}>
                <table className="history-table">
                  <thead>
                    <tr>
                      <th>Time</th>
                      <th>Round</th>
                      <th>Val Acc</th>
                      <th>Test Acc</th>
                      <th>Active Nodes</th>
                      <th>Completed</th>
                      <th>In Progress</th>
                    </tr>
                  </thead>
                  <tbody>
                    {[...metricHistory].reverse().slice(0, 20).map((row, i) => (
                      <tr key={i} style={{ opacity: Math.max(0.45, 1 - i * 0.035) }}>
                        <td className="mono">{row.label}</td>
                        <td className="mono">v{row.round_no}</td>
                        <td style={{ color: '#6366f1' }}>{fmtAcc(row.val_acc)}</td>
                        <td style={{ color: '#10b981' }}>{fmtAcc(row.test_acc)}</td>
                        <td>{row.active_nodes}</td>
                        <td style={{ color: '#10b981' }}>{row.completed}</td>
                        <td style={{ color: '#3b82f6' }}>{row.in_progress}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            ) : (
              <OfflinePlaceholder height={140} />
            )}
          </SectionCard>

        </div>
      </main>
    </div>
  );
}
