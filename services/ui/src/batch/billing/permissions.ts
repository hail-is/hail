export type BillingRole = 'global_bm' | 'quote_owner' | 'quote_manager' | 'bp_member';

export type Permission =
  | 'view_quote'
  | 'edit_quote'
  | 'close_quote'
  | 'add_manager'
  | 'manage_managers'
  | 'view_bp'
  | 'create_bp'
  | 'edit_bp_limit'
  | 'edit_bp_metadata'
  | 'add_bp_member'
  | 'manage_bp_members'
  | 'close_reopen_bp'
  | 'change_bp_quote'
  | 'view_events';

const ROLE_PERMISSIONS: Record<BillingRole, Set<Permission>> = {
  global_bm: new Set([
    'view_quote', 'edit_quote', 'close_quote', 'add_manager', 'manage_managers',
    'view_bp', 'create_bp', 'edit_bp_limit', 'edit_bp_metadata',
    'add_bp_member', 'manage_bp_members', 'close_reopen_bp', 'change_bp_quote', 'view_events',
  ]),
  quote_owner: new Set([
    'view_quote', 'edit_quote', 'close_quote', 'manage_managers',
    'view_bp', 'create_bp', 'edit_bp_limit', 'edit_bp_metadata',
    'manage_bp_members', 'close_reopen_bp', 'change_bp_quote', 'view_events',
  ]),
  quote_manager: new Set([
    'view_quote', 'edit_quote',
    'view_bp', 'create_bp', 'edit_bp_limit', 'edit_bp_metadata',
    'manage_bp_members', 'close_reopen_bp', 'change_bp_quote', 'view_events',
  ]),
  bp_member: new Set([
    'view_bp', 'edit_bp_metadata', 'manage_bp_members', 'view_events',
  ]),
};

export function can(role: BillingRole | null | undefined, permission: Permission): boolean {
  if (!role) return false;
  return ROLE_PERMISSIONS[role]?.has(permission) ?? false;
}
