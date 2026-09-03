export function StatLabel({ label, tooltip }: { label: string; tooltip: string }) {
  return (
    <span title={tooltip} className="cursor-help underline decoration-dotted decoration-zinc-400">
      {label}
    </span>
  );
}
