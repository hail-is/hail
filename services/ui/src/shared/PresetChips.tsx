export function PresetChips({ presets, activeNum, activeDen, onSelect }: {
  presets: { label: string; num: string; den: string }[];
  activeNum: string; activeDen: string;
  onSelect: (num: string, den: string) => void;
}) {
  return (
    <div className="flex flex-wrap gap-1.5 py-2">
      {presets.map(p => {
        const active = p.num === activeNum && p.den === activeDen;
        return (
          <button
            key={p.label}
            onClick={() => onSelect(p.num, p.den)}
            className={`text-xs px-2 py-0.5 rounded-full border transition-colors ${active ? 'bg-sky-500 border-sky-500 text-white' : 'border-zinc-300 text-zinc-500 hover:border-sky-400 hover:text-sky-600 bg-white'}`}
          >
            {p.label}
          </button>
        );
      })}
    </div>
  );
}

export function ScatterPresetChips({ presets, activeX, activeY, onSelect }: {
  presets: { label: string; x: string; y: string }[];
  activeX: string; activeY: string;
  onSelect: (x: string, y: string) => void;
}) {
  return (
    <div className="flex flex-wrap gap-1.5 py-2">
      {presets.map(p => {
        const active = p.x === activeX && p.y === activeY;
        return (
          <button
            key={p.label}
            onClick={() => onSelect(p.x, p.y)}
            className={`text-xs px-2 py-0.5 rounded-full border transition-colors ${active ? 'bg-sky-500 border-sky-500 text-white' : 'border-zinc-300 text-zinc-500 hover:border-sky-400 hover:text-sky-600 bg-white'}`}
          >
            {p.label}
          </button>
        );
      })}
    </div>
  );
}
