export function MetricChart({ values, label, color = "#62d98f" }: { values: number[]; label: string; color?: string }) {
  if (!values.length) return <div className="chart-empty">Metrics appear after the first validation epoch.</div>;
  const width = 440;
  const height = 132;
  const max = Math.max(...values, 0.001);
  const min = Math.min(...values, 0);
  const range = Math.max(max - min, 0.001);
  const points = values.map((value, index) => {
    const x = values.length === 1 ? 0 : (index / (values.length - 1)) * width;
    const y = height - ((value - min) / range) * (height - 16) - 8;
    return `${x},${y}`;
  }).join(" ");
  return (
    <div className="metric-chart">
      <div className="chart-label"><span>{label}</span><strong>{values.at(-1)?.toFixed(3)}</strong></div>
      <svg viewBox={`0 0 ${width} ${height}`} preserveAspectRatio="none" role="img" aria-label={`${label} chart`}>
        <defs><linearGradient id={`gradient-${label.replaceAll(" ", "-")}`} x1="0" y1="0" x2="0" y2="1"><stop offset="0" stopColor={color} stopOpacity=".3"/><stop offset="1" stopColor={color} stopOpacity="0"/></linearGradient></defs>
        <polyline points={`0,${height} ${points} ${width},${height}`} fill={`url(#gradient-${label.replaceAll(" ", "-")})`} stroke="none" />
        <polyline points={points} fill="none" stroke={color} strokeWidth="3" vectorEffect="non-scaling-stroke" />
      </svg>
    </div>
  );
}

