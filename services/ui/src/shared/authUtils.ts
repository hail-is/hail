export function hasPermission(name: string): boolean {
  const content = document.head.querySelector('meta[name="system-permissions"]')?.getAttribute('content') ?? '{}';
  return (JSON.parse(content) as Record<string, boolean>)[name] === true;
}
