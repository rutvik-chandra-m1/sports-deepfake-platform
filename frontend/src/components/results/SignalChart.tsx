import { Bar, BarChart, Cell, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { signalLabel } from "@/lib/signalMeta";
import type { DetectorBreakdown } from "@/types/api";

interface SignalChartProps {
  breakdown: DetectorBreakdown;
}

interface ChartRow {
  name: string;
  score: number;
  weight: number;
}

// Thresholds mirror backend settings.explanation_low_threshold /
// explanation_high_threshold (0.35 / 0.65 by default) -- kept as a visual
// approximation here since the frontend doesn't fetch backend config.
function bandColor(scorePercent: number): string {
  if (scorePercent >= 65) return "var(--color-suspicious)";
  if (scorePercent <= 35) return "var(--color-authentic)";
  return "var(--color-accent)";
}

export function SignalChart({ breakdown }: SignalChartProps) {
  const data: ChartRow[] = Object.entries(breakdown.signals)
    .map(([name, info]) => ({
      name: signalLabel(name),
      score: Math.round(info.score * 100),
      weight: Math.round(info.weight * 100),
    }))
    .sort((a, b) => b.score - a.score);

  if (data.length === 0) {
    return (
      <p className="text-sm text-text-muted">No signals were available to score this analysis.</p>
    );
  }

  return (
    <div style={{ width: "100%", height: Math.max(data.length * 36, 120) }}>
      <ResponsiveContainer>
        <BarChart data={data} layout="vertical" margin={{ left: 8, right: 24, top: 4, bottom: 4 }}>
          <XAxis
            type="number"
            domain={[0, 100]}
            tick={{ fill: "var(--color-text-faint)", fontSize: 11, fontFamily: "var(--font-mono)" }}
            axisLine={{ stroke: "var(--color-border)" }}
            tickLine={false}
            unit="%"
          />
          <YAxis
            type="category"
            dataKey="name"
            width={160}
            tick={{ fill: "var(--color-text-muted)", fontSize: 12, fontFamily: "var(--font-body)" }}
            axisLine={{ stroke: "var(--color-border)" }}
            tickLine={false}
          />
          <Tooltip
            cursor={{ fill: "var(--color-surface-hover)" }}
            contentStyle={{
              background: "var(--color-bg-raised)",
              border: "1px solid var(--color-border-strong)",
              borderRadius: 6,
              fontFamily: "var(--font-mono)",
              fontSize: 12,
            }}
            labelStyle={{ color: "var(--color-text)" }}
            formatter={(value) => [`${value}%`, "Suspicion score"]}
          />
          <Bar dataKey="score" radius={[0, 3, 3, 0]} maxBarSize={16}>
            {data.map((row) => (
              <Cell key={row.name} fill={bandColor(row.score)} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
