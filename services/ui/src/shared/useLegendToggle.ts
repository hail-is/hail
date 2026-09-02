import { useCallback, useState } from 'react';

export function useLegendToggle(allKeys: readonly string[]) {
  const [hidden, setHidden] = useState<Set<string>>(new Set());
  const onLegendClick = useCallback(
    (e: { dataKey?: string | number | ((obj: unknown) => unknown) }, _index: number, event: { shiftKey: boolean }) => {
      if (typeof e.dataKey !== 'string') return;
      const key = e.dataKey;
      if (event.shiftKey) {
        // shift-click: toggle this series on/off
        setHidden(prev => {
          const next = new Set(prev);
          if (next.has(key)) next.delete(key); else next.add(key);
          return next;
        });
      } else {
        // click: solo this series (or restore all if already soloed)
        setHidden(prev => {
          const visible = allKeys.filter(k => !prev.has(k));
          const isSolo = visible.length === 1 && visible[0] === key;
          return isSolo ? new Set() : new Set(allKeys.filter(k => k !== key));
        });
      }
    },
    [allKeys]
  );
  const isHidden = (key: string) => hidden.has(key);
  return { onLegendClick, isHidden, setHidden };
}
