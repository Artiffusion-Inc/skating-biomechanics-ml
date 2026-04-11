const periods = [
  { value: "7d", label: "7 дн" },
  { value: "30d", label: "30 дн" },
  { value: "90d", label: "90 дн" },
  { value: "all", label: "Всё" },
]

export function PeriodSelector({ value, onChange }: { value: string; onChange: (v: string) => void }) {
  return (
    <div className="flex gap-1 rounded-lg bg-muted p-1">
      {periods.map((p) => (
        <button
          key={p.value}
          onClick={() => onChange(p.value)}
          className={`rounded-md px-3 py-1 text-xs font-medium transition-colors ${
            value === p.value ? "bg-background shadow-sm" : "text-muted-foreground hover:text-foreground"
          }`}
        >
          {p.label}
        </button>
      ))}
    </div>
  )
}
