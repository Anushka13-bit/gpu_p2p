export function formatDuration(seconds) {
  if (seconds == null) return '—';
  const s = Math.floor(seconds);
  if (s < 60) return `${s}s`;
  if (s < 3600) return `${Math.floor(s / 60)}m ${s % 60}s`;
  return `${Math.floor(s / 3600)}h ${Math.floor((s % 3600) / 60)}m`;
}

export function formatAgo(secs) {
  if (secs == null) return '—';
  if (secs < 2) return 'just now';
  if (secs < 60) return `${Math.floor(secs)}s ago`;
  return `${Math.floor(secs / 60)}m ago`;
}

export function shortId(id) {
  if (!id) return '—';
  return id.slice(0, 8) + '…';
}

export function fmtPct(v) {
  if (v == null) return '—';
  return `${Number(v).toFixed(1)}%`;
}

export function fmtAcc(v) {
  if (v == null) return '—';
  return `${Number(v).toFixed(2)}%`;
}

export function statusColor(status) {
  switch (status) {
    case 'COMPLETED':   return { fg: '#10b981', bg: 'rgba(16,185,129,0.12)', dot: '#10b981' };
    case 'IN_PROGRESS': return { fg: '#3b82f6', bg: 'rgba(59,130,246,0.12)', dot: '#3b82f6' };
    case 'ASSIGNED':    return { fg: '#6366f1', bg: 'rgba(99,102,241,0.12)', dot: '#6366f1' };
    case 'ORPHANED':    return { fg: '#ef4444', bg: 'rgba(239,68,68,0.12)',  dot: '#ef4444' };
    case 'PENDING':     return { fg: '#f59e0b', bg: 'rgba(245,158,11,0.12)', dot: '#f59e0b' };
    default:            return { fg: '#94a3b8', bg: 'rgba(148,163,184,0.1)', dot: '#94a3b8' };
  }
}

export function gpuLabel(mb) {
  if (!mb) return 'CPU';
  return mb >= 1024 ? `${(mb / 1024).toFixed(0)} GB` : `${mb} MB`;
}
