import styles from './SectionCard.module.css';

export default function SectionCard({ title, subtitle, icon, children, action, accent }) {
  return (
    <div className={styles.card}>
      <div className={styles.header}>
        <div className={styles.titleGroup}>
          {icon && (
            <span className={styles.icon} style={{ color: accent || 'var(--accent-purple)' }}>
              {icon}
            </span>
          )}
          <div>
            <h2 className={styles.title}>{title}</h2>
            {subtitle && <p className={styles.subtitle}>{subtitle}</p>}
          </div>
        </div>
        {action && <div className={styles.action}>{action}</div>}
      </div>
      <div className={styles.body}>{children}</div>
    </div>
  );
}
