import { useState } from 'react';

type TipState = { text: string; x: number; y: number } | null;

export function useTip(): [TipState, (text: string) => (e: React.MouseEvent) => void, () => void] {
  const [tip, setTip] = useState<TipState>(null);
  const onEnter = (text: string) => (e: React.MouseEvent) => setTip({ text, x: e.clientX, y: e.clientY });
  const onLeave = () => setTip(null);
  return [tip, onEnter, onLeave];
}

export function FloatingTip({ tip }: { tip: TipState }): JSX.Element | null {
  if (!tip) return null;
  return (
    <div
      className="fixed z-50 pointer-events-none bg-zinc-800 text-white text-xs px-2 py-1 rounded shadow whitespace-nowrap"
      style={{ left: tip.x + 12, top: tip.y - 32 }}
    >
      {tip.text}
    </div>
  );
}
