const MIN_VISIBLE_COST = 0.01;

export function CostDisplay({ cost }: { cost: number }): JSX.Element {
  if (cost > 0 && cost < MIN_VISIBLE_COST) {
    return (
      <span title={`$${cost}`} className="cursor-help">
        {`< $${MIN_VISIBLE_COST.toFixed(2)}`}
      </span>
    );
  }
  return <>${cost.toFixed(2)}</>;
}
