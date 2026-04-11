"use client"

const ELEMENTS = [
  { id: "three_turn", label: "Тройка" },
  { id: "waltz_jump", label: "Вальсовый" },
  { id: "toe_loop", label: "Перекидной" },
  { id: "flip", label: "Флип" },
  { id: "salchow", label: "Сальхов" },
  { id: "loop", label: "Петля" },
  { id: "lutz", label: "Лютц" },
  { id: "axel", label: "Аксель" },
]

export function ElementPicker({ value, onChange }: { value: string | null; onChange: (id: string) => void }) {
  return (
    <div className="grid grid-cols-3 gap-2 sm:grid-cols-4">
      {ELEMENTS.map((el) => (
        <button
          key={el.id}
          onClick={() => onChange(el.id)}
          className={`truncate rounded-xl border px-2 py-2.5 text-center text-xs transition-colors sm:p-3 sm:text-sm ${
            value === el.id ? "border-primary bg-primary/10 text-primary" : "border-border hover:bg-accent/50"
          }`}
        >
          {el.label}
        </button>
      ))}
    </div>
  )
}
