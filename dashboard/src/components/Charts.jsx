import {
  LineChart, Line, AreaChart, Area, BarChart, Bar,
  PieChart, Pie, Cell,
  XAxis, YAxis, CartesianGrid, Tooltip, Legend,
  ResponsiveContainer,
} from 'recharts';

const TOOLTIP_STYLE = {
  contentStyle: {
    background: 'rgba(13,19,33,0.95)',
    border: '1px solid rgba(99,102,241,0.3)',
    borderRadius: 10,
    color: '#f1f5f9',
    fontSize: 12,
    backdropFilter: 'blur(12px)',
  },
  labelStyle: { color: '#94a3b8', fontWeight: 600 },
  cursor: { stroke: 'rgba(99,102,241,0.2)', strokeWidth: 1 },
};

/**
 * Accuracy over time
 * data: Array<{ label, val_acc: number|null, test_acc: number|null }>
 * val_acc / test_acc are null until the first FedAvg round completes — we skip
 * null entries so recharts draws a gap rather than a flat line at 0.
 */
export function GpuSpecsChart({ data }) {
  const filtered = data.filter(d => d.gpu_vram_gb != null || d.cpu_total != null);
  if (filtered.length === 0) return null;

  const vramKeys = Object.keys(filtered[filtered.length - 1] || {}).filter(k => k.startsWith('vram_'));
  const colors = ['#06b6d4', '#3b82f6', '#6366f1', '#10b981', '#f59e0b', '#ec4899'];

  return (
    <ResponsiveContainer width="100%" height={220}>
      <AreaChart data={filtered} margin={{ top: 8, right: 8, left: -20, bottom: 0 }}>
        <XAxis dataKey="label" tick={{ fill: '#475569', fontSize: 10 }} axisLine={false} tickLine={false} interval="preserveStartEnd" />
        <YAxis
          yAxisId="left"
          tick={{ fill: '#475569', fontSize: 10 }}
          axisLine={false}
          tickLine={false}
          width={44}
          tickFormatter={(v) => `${v.toFixed(1)}GB`}
        />
        <YAxis
          yAxisId="right"
          orientation="right"
          tick={{ fill: '#475569', fontSize: 10 }}
          axisLine={false}
          tickLine={false}
          width={34}
          allowDecimals={false}
        />
        <Tooltip
          {...TOOLTIP_STYLE}
          formatter={(v, name) => {
            if (v == null) return ['—', name];
            if (String(name).includes('VRAM')) return [`${Number(v).toFixed(2)} GB`, name];
            return [`${Number(v)}`, name];
          }}
        />
        <Legend wrapperStyle={{ fontSize: 12, color: '#94a3b8' }} />
        {vramKeys.length > 0 ? (
          vramKeys.map((k, idx) => (
            <Area
              key={k}
              yAxisId="left"
              type="monotone"
              dataKey={k}
              stackId="vram"
              name={`VRAM ${k.replace('vram_', '')}`}
              stroke={colors[idx % colors.length]}
              fill={colors[idx % colors.length]}
              fillOpacity={0.18}
              strokeWidth={2}
              dot={false}
              activeDot={{ r: 3, fill: colors[idx % colors.length] }}
              connectNulls={false}
            />
          ))
        ) : (
          <Area
            yAxisId="left"
            type="monotone"
            dataKey="gpu_vram_gb"
            name="Total VRAM"
            stroke="#06b6d4"
            fill="#06b6d4"
            fillOpacity={0.18}
            strokeWidth={2.5}
            dot={false}
            activeDot={{ r: 3, fill: '#06b6d4' }}
            connectNulls={false}
          />
        )}
        <Line
          yAxisId="right"
          type="monotone"
          dataKey="cpu_total"
          name="Total CPU"
          stroke="#f59e0b"
          strokeWidth={2.5}
          dot={false}
          activeDot={{ r: 4, fill: '#f59e0b' }}
        />
      </AreaChart>
    </ResponsiveContainer>
  );
}

/**
 * Active nodes bar chart
 * data: Array<{ label, active_nodes: number }>
 */
export function ActiveNodesChart({ data }) {
  const maxVal = Math.max(0, ...(data || []).map(d => Number(d?.active_nodes) || 0));
  if (maxVal === 0) {
    return (
      <div style={{
        height: 180,
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        color: 'var(--text-muted)',
        fontSize: 12,
      }}>
        No active nodes yet (waiting for worker heartbeats)
      </div>
    );
  }
  return (
    <ResponsiveContainer width="100%" height={180}>
      <BarChart data={data} margin={{ top: 4, right: 8, left: -20, bottom: 0 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.04)" />
        <XAxis dataKey="label" tick={{ fill: '#475569', fontSize: 10 }} axisLine={false} tickLine={false} interval="preserveStartEnd" />
        <YAxis tick={{ fill: '#475569', fontSize: 10 }} axisLine={false} tickLine={false} allowDecimals={false} />
        <Tooltip {...TOOLTIP_STYLE} />
        <Bar
          dataKey="active_nodes" name="Active Nodes"
          fill="#3b82f6" radius={[4, 4, 0, 0]}
          background={{ fill: 'rgba(255,255,255,0.02)', radius: [4, 4, 0, 0] }}
        />
      </BarChart>
    </ResponsiveContainer>
  );
}

/**
 * Task status pie chart — built from the live task_table (not from history)
 * so it always reflects the current backend state.
 *
 * taskTable: Record<string, { status: 'PENDING'|'ASSIGNED'|'IN_PROGRESS'|'COMPLETED'|'ORPHANED' }>
 */
export function TaskStatusChart({ taskTable }) {
  const counts = {};
  Object.values(taskTable).forEach(t => {
    counts[t.status] = (counts[t.status] ?? 0) + 1;
  });

  const colorMap = {
    COMPLETED:   '#10b981',
    IN_PROGRESS: '#3b82f6',
    ASSIGNED:    '#6366f1',
    PENDING:     '#f59e0b',
    ORPHANED:    '#ef4444',
  };

  const pie = Object.entries(counts).map(([name, value]) => ({
    name,
    value,
    color: colorMap[name] ?? '#94a3b8',
  }));

  if (pie.length === 0) return null;

  return (
    <ResponsiveContainer width="100%" height={180}>
      <PieChart>
        <Pie
          data={pie}
          cx="50%" cy="50%"
          innerRadius={48} outerRadius={72}
          paddingAngle={3} dataKey="value"
          strokeWidth={0}
        >
          {pie.map((entry) => (
            <Cell key={entry.name} fill={entry.color} />
          ))}
        </Pie>
        <Tooltip {...TOOLTIP_STYLE} />
        <Legend wrapperStyle={{ fontSize: 12, color: '#94a3b8' }} />
      </PieChart>
    </ResponsiveContainer>
  );
}

/**
 * Fed round step chart
 * data: Array<{ label, round_no: number }>
 */
export function RoundsChart({ data }) {
  return (
    <ResponsiveContainer width="100%" height={180}>
      <LineChart data={data} margin={{ top: 4, right: 8, left: -20, bottom: 0 }}>
        <XAxis dataKey="label" tick={{ fill: '#475569', fontSize: 10 }} axisLine={false} tickLine={false} interval="preserveStartEnd" />
        <YAxis tick={{ fill: '#475569', fontSize: 10 }} axisLine={false} tickLine={false} allowDecimals={false} />
        <Tooltip {...TOOLTIP_STYLE} />
        <Line
          type="stepAfter" dataKey="round_no" name="Round"
          stroke="#f59e0b" strokeWidth={2.5}
          dot={false} activeDot={{ r: 4, fill: '#f59e0b' }}
        />
      </LineChart>
    </ResponsiveContainer>
  );
}

/**
 * Per-shard horizontal bar chart
 * taskTable: Record<string, { progress_pct, eval_acc, status }>
 * progress_pct comes from registry_snapshot (computed from last_reported_index vs range)
 * eval_acc comes from shard.last_eval_acc (set after submit_weights with shard_complete)
 */
export function ShardProgressChart({ taskTable }) {
  const barData = Object.entries(taskTable).map(([tid, info]) => ({
    name: tid,
    progress: info.progress_pct != null ? parseFloat(info.progress_pct.toFixed(1)) : 0,
    eval_acc: info.eval_acc    != null ? parseFloat(parseFloat(info.eval_acc).toFixed(1)) : null,
  }));

  if (barData.length === 0) return null;

  return (
    <ResponsiveContainer width="100%" height={Math.max(200, barData.length * 52)}>
      <BarChart data={barData} margin={{ top: 8, right: 16, left: -10, bottom: 0 }} layout="vertical">
        <XAxis
          type="number" domain={[0, 100]}
          tick={{ fill: '#475569', fontSize: 10 }} axisLine={false} tickLine={false} unit="%"
        />
        <YAxis
          dataKey="name" type="category"
          tick={{ fill: '#94a3b8', fontSize: 11, fontFamily: 'JetBrains Mono, monospace' }}
          axisLine={false} tickLine={false} width={65}
        />
        <Tooltip
          {...TOOLTIP_STYLE}
          formatter={(v, name) => [v != null ? `${v}%` : '—', name]}
        />
        <Legend wrapperStyle={{ fontSize: 12, color: '#94a3b8' }} />
        <Bar dataKey="progress"  name="Progress %" fill="#3b82f6" radius={[0, 4, 4, 0]} barSize={12} />
        <Bar dataKey="eval_acc"  name="Eval Acc %" fill="#10b981" radius={[0, 4, 4, 0]} barSize={12} />
      </BarChart>
    </ResponsiveContainer>
  );
}

/**
 * Data shard "eaten" progress bar
 * - Single horizontal bar representing 100% of the dataset.
 * - Split into segments (one per shard) sized by that shard's range length.
 * - Each segment fills with shard progress_pct; shows assigned worker + range.
 *
 * taskTable: Record<string, { range:[start,end], progress_pct:number, worker:string|null, status:string }>
 */
export function DataShardBar({ taskTable }) {
  const shards = Object.entries(taskTable || {})
    .map(([tid, info]) => ({
      tid,
      start: Array.isArray(info?.range) ? Number(info.range[0]) : 0,
      end: Array.isArray(info?.range) ? Number(info.range[1]) : 0,
      progress: typeof info?.progress_pct === 'number' ? info.progress_pct : Number(info?.progress_pct) || 0,
      worker: info?.worker || '—',
      status: info?.status || 'PENDING',
    }))
    .filter(s => Number.isFinite(s.start) && Number.isFinite(s.end) && s.end > s.start)
    .sort((a, b) => a.start - b.start);

  if (shards.length === 0) return null;

  const total = shards.reduce((acc, s) => acc + (s.end - s.start), 0) || 1;
  const statusColor = (st) => {
    switch (st) {
      case 'COMPLETED': return '#10b981';
      case 'IN_PROGRESS': return '#3b82f6';
      case 'ASSIGNED': return '#6366f1';
      case 'ORPHANED': return '#ef4444';
      case 'PENDING':
      default: return '#475569';
    }
  };

  return (
    <div>
      <div style={{
        display: 'flex',
        width: '100%',
        height: 28,
        borderRadius: 10,
        overflow: 'hidden',
        border: '1px solid rgba(255,255,255,0.08)',
        background: 'rgba(255,255,255,0.03)',
      }}>
        {shards.map((s, i) => {
          const segPct = ((s.end - s.start) / total) * 100;
          const fill = Math.max(0, Math.min(100, s.progress || 0));
          const base = statusColor(s.status);
          return (
            <div
              key={s.tid}
              title={`${s.worker} • ${s.start}-${s.end} • ${fill.toFixed(1)}%`}
              style={{
                width: `${segPct}%`,
                position: 'relative',
                borderRight: i === shards.length - 1 ? 'none' : '1px solid rgba(255,255,255,0.06)',
              }}
            >
              <div style={{ position: 'absolute', inset: 0, background: 'rgba(255,255,255,0.02)' }} />
              <div style={{
                position: 'absolute',
                inset: 0,
                width: `${fill}%`,
                background: `${base}cc`,
                boxShadow: `0 0 16px ${base}55 inset`,
              }} />
            </div>
          );
        })}
      </div>

      <div style={{
        marginTop: 10,
        display: 'grid',
        gridTemplateColumns: 'repeat(auto-fit, minmax(240px, 1fr))',
        gap: 10,
      }}>
        {shards.map((s) => {
          const fill = Math.max(0, Math.min(100, s.progress || 0));
          const base = statusColor(s.status);
          const shortWorker = s.worker && s.worker !== '—' ? String(s.worker).slice(0, 12) : '—';
          return (
            <div key={`${s.tid}-meta`} style={{
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'space-between',
              gap: 10,
              padding: '6px 10px',
              borderRadius: 10,
              border: '1px solid rgba(255,255,255,0.06)',
              background: 'rgba(255,255,255,0.02)',
              fontSize: 12,
              color: 'var(--text-secondary)',
            }}>
              <span style={{ fontFamily: 'JetBrains Mono, monospace', color: 'var(--text-primary)' }}>
                {s.start}–{s.end}
              </span>
              <span style={{ display: 'inline-flex', alignItems: 'center', gap: 8 }}>
                <span style={{ color: base, fontWeight: 700 }}>{fill.toFixed(0)}%</span>
                <span style={{
                  padding: '2px 8px',
                  borderRadius: 99,
                  background: `${base}1a`,
                  color: base,
                  border: `1px solid ${base}33`,
                  fontFamily: 'JetBrains Mono, monospace',
                  fontSize: 11,
                }}>
                  {shortWorker}
                </span>
              </span>
            </div>
          );
        })}
      </div>
    </div>
  );
}
