// Note that this enables UI element visibility checks, but is not a security check.
// All data and requests still have to go via the backend, and the backend enforces auth independently.
export function hasPermission(name: string): boolean {
  const content = document.head.querySelector('meta[name="system-permissions"]')?.getAttribute('content') ?? '{}';
  const perms = new Map<string, boolean>(Object.entries(JSON.parse(content) as Record<string, boolean>));
  return perms.get(name) === true;
}
