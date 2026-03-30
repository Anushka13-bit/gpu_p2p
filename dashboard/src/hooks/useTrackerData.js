import { useState, useEffect, useCallback, useRef } from 'react';

// When Vite proxy is active (npm run dev) VITE_TRACKER_URL is empty → calls go through proxy to localhost:8000
// When deployed standalone, set VITE_TRACKER_URL=http://<host>:8000
const TRACKER_BASE = import.meta.env.VITE_TRACKER_URL || '';
const POLL_MS = 3000;

async function fetchJSON(path) {
  const res = await fetch(`${TRACKER_BASE}${path}`, {
    signal: AbortSignal.timeout(4000),
  });
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return res.json();
}

/**
 * Derives a single normalised snapshot from the raw /health + /registry responses.
 *
 * /health  → { status, round_no, tasks: { ...health_snapshot fields } }
 *            tasks contains: workers, worker_roster, task_table, training_stopped,
 *            stop_reason, best_val_acc, rounds_without_improve, stop_policy,
 *            node_registry, round_no, version_label, has_global_weights,
 *            last_val_acc, last_test_acc
 *
 * /registry → registry_snapshot fields: total_nodes, active_nodes, nodes_on_shard,
 *             heartbeat_timeout_sec, training_stopped, stop_reason, best_val_acc,
 *             rounds_without_improve, stop_policy, nodes[], task_table{}
 *
 * We trust /registry for per-node live data (nodes[]) and /health for
 * global FedAvg metrics (last_val_acc, last_test_acc, round_no, etc.).
 */
function buildSnapshot(health, registry) {
  const h = health?.tasks ?? {};
  const r = registry ?? {};

  // task_table: prefer registry (has progress_pct), fallback to health
  const taskTable = r.task_table && Object.keys(r.task_table).length > 0
    ? r.task_table
    : h.task_table ?? {};

  // node list: only /registry has the per-worker live progress fields
  const nodes = r.nodes ?? h.worker_roster ?? [];

  // stop policy
  const stopPolicy = r.stop_policy ?? h.stop_policy ?? {};

  // node registry summary
  const nodeRegistry = {
    total_nodes:           r.total_nodes           ?? h.node_registry?.total_nodes           ?? nodes.length,
    active_nodes:          r.active_nodes          ?? h.node_registry?.active_nodes          ?? 0,
    nodes_on_shard:        r.nodes_on_shard        ?? 0,
    heartbeat_timeout_sec: r.heartbeat_timeout_sec ?? h.node_registry?.heartbeat_timeout_sec ?? 12,
  };

  return {
    // Global FedAvg model metrics — from /health (most authoritative)
    round_no:              health?.round_no ?? h.round_no ?? 1,
    version_label:         h.version_label ?? `Global Model v${health?.round_no ?? 1}`,
    has_global_weights:    h.has_global_weights ?? false,
    last_val_acc:          h.last_val_acc  ?? null,   // null until FedAvg completes first round
    last_test_acc:         h.last_test_acc ?? null,
    best_val_acc:          r.best_val_acc  ?? h.best_val_acc ?? null,
    rounds_without_improve: r.rounds_without_improve ?? h.rounds_without_improve ?? 0,

    // Training state
    training_stopped: r.training_stopped ?? h.training_stopped ?? false,
    stop_reason:      r.stop_reason      ?? h.stop_reason      ?? null,
    stop_policy:      stopPolicy,

    // Nodes & tasks
    nodes,
    nodeRegistry,
    taskTable,
  };
}

export function useTrackerData() {
  // null means "no data yet / tracker offline"
  const [snapshot, setSnapshot] = useState(null);
  const [metricHistory, setMetricHistory] = useState([]);
  // 'connecting' | 'online' | 'offline'
  const [connectionStatus, setConnectionStatus] = useState('connecting');
  const [lastUpdated, setLastUpdated] = useState(null);

  const historyRef = useRef([]);

  const pushHistoryPoint = useCallback((snap) => {
    const now = Date.now();
    const tt = snap.taskTable;
    const point = {
      t: now,
      label:        new Date(now).toLocaleTimeString(),
      val_acc:      snap.last_val_acc,   // null until first FedAvg round
      test_acc:     snap.last_test_acc,
      round_no:     snap.round_no,
      active_nodes: snap.nodeRegistry.active_nodes,
      completed:    Object.values(tt).filter(t => t.status === 'COMPLETED').length,
      in_progress:  Object.values(tt).filter(t => t.status === 'IN_PROGRESS').length,
    };
    historyRef.current = [...historyRef.current.slice(-49), point];
    setMetricHistory([...historyRef.current]);
  }, []);

  const poll = useCallback(async () => {
    try {
      // Fetch both endpoints; /registry may not exist yet on older deployments
      const [health, registry] = await Promise.all([
        fetchJSON('/health'),
        fetchJSON('/registry').catch(() => null),
      ]);

      const snap = buildSnapshot(health, registry);
      setSnapshot(snap);
      setConnectionStatus('online');
      pushHistoryPoint(snap);
      setLastUpdated(new Date());
    } catch {
      // Tracker is unreachable — clear snapshot so UI shows empty layout
      setSnapshot(null);
      setConnectionStatus('offline');
      // Do NOT push to history when offline
    }
  }, [pushHistoryPoint]);

  useEffect(() => {
    poll();
    const id = setInterval(poll, POLL_MS);
    return () => clearInterval(id);
  }, [poll]);

  return { snapshot, metricHistory, connectionStatus, lastUpdated };
}
