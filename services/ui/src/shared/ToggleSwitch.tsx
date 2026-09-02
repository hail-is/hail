export function ToggleSwitch({ checked, onChange, label = '% of total' }: { checked: boolean; onChange: (v: boolean) => void; label?: string }) {
  return (
    <label className="flex items-center gap-1.5 cursor-pointer select-none">
      <input type="checkbox" className="sr-only" checked={checked} onChange={e => onChange(e.target.checked)} />
      <div className={`relative w-8 h-4 rounded-full transition-colors ${checked ? 'bg-sky-500' : 'bg-zinc-300'}`}>
        <div className={`absolute top-0.5 w-3 h-3 rounded-full bg-white shadow transition-transform ${checked ? 'translate-x-4' : 'translate-x-0.5'}`} />
      </div>
      <span className="text-xs text-zinc-600">{label}</span>
    </label>
  );
}
