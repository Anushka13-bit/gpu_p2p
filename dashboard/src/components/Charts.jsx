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
export function AccuracyChart({ data }) {
  // Filter to only points where at least one accuracy value is present
  const filtered = data.filter(d => d.val_acc != null || d.test_acc != null);
  if (filtered.length === 0) return null;

  return (
    <ResponsiveContainer width="100%" height={220}>
      <AreaChart data={filtered} margin={{ top: 8, right: 8, left: -20, bottom: 0 }}>
        <defs>
          <linearGradient id="valGrad" x1="0" y1="0" x2="0" y2="1">
            <stop offset="5%"  stopColor="#6366f1" stopOpacity={0.35} />
            <stop offset="95%" stopColor="#6366f1" stopOpacity={0}    />
          </linearGradient>
          <linearGradient id="testGrad" x1="0" y1="0" x2="0" y2="1">
            <stop offset="5%"  stopColor="#10b981" stopOpacity={0.30} />
            <stop offset="95%" stopColor="#10b981" stopOpacity={0}    />
          </linearGradient>
        </defs>
        <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.04)" />
        <XAxis dataKey="label" tick={{ fill: '#475569', fontSize: 10 }} axisLine={false} tickLine={false} interval="preserveStartEnd" />
        <YAxis domain={['auto', 'auto']} tick={{ fill: '#475569', fontSize: 10 }} axisLine={false} tickLine={false} unit="%" />
        <Tooltip {...TOOLTIP_STYLE} formatter={(v) => v != null ? [`${v.toFixed(2)}%`] : ['—']} />
        <Legend wrapperStyle={{ fontSize: 12, color: '#94a3b8' }} />
        <Area
          type="monotone" dataKey="val_acc"  name="Val Acc"
          stroke="#6366f1" strokeWidth={2} fill="url(#valGrad)"
          dot={false} activeDot={{ r: 4, fill: '#6366f1' }}
          connectNulls={false}
        />
        <Area
          type="monotone" dataKey="test_acc" name="Test Acc"
          stroke="#10b981" strokeWidth={2} fill="url(#testGrad)"
          dot={false} activeDot={{ r: 4, fill: '#10b981' }}
          connectNulls={false}
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
        <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.04)" />
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
        <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.04)" horizontal={false} />
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
