type SystemPermissions = Record<string, boolean>;

export function hasPermission(name: string): boolean {
  const content = document.head.querySelector('meta[name="system-permissions"]')?.getAttribute('content') ?? '{}';
  const perms: SystemPermissions = JSON.parse(content) as SystemPermissions;
  return perms[name] === true;
}
