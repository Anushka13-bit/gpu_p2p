import styles from './StatCard.module.css';

/**
 * StatCard
 * - loading=true  → shimmer skeleton (tracker connecting/offline)
 * - value=null    → show '—' (tracker online but metric not yet available, e.g. before first FedAvg)
 * - value=string  → display the real value
 */
export default function StatCard({ icon, label, value, sub, accent = '#6366f1', trend, badge, loading }) {
  return (
    <div className={styles.card} style={{ '--accent': accent }}>
      <div className={styles.header}>
        <div className={styles.iconWrap} style={{ background: `${accent}20`, color: accent }}>
          {icon}
        </div>
        {badge && (
          <span className={styles.badge} style={{ background: `${accent}18`, color: accent }}>
            {badge}
          </span>
        )}
      </div>

      <>
        <p className={styles.value}>
          {loading ? '—' : (value ?? '—')}
        </p>
        <p className={styles.label}>{label}</p>
        {sub != null && <p className={styles.sub}>{sub}</p>}
        {trend != null && !loading && (
          <span className={styles.trend} style={{ color: trend >= 0 ? '#10b981' : '#ef4444' }}>
            {trend >= 0 ? '▲' : '▼'} {Math.abs(trend).toFixed(2)}%
          </span>
        )}
      </>

      <div
        className={styles.glow}
        style={{ background: `radial-gradient(ellipse at 50% 100%, ${accent}22 0%, transparent 70%)` }}
      />
    </div>
  );
}
