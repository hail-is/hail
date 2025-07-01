from enum import Enum

class SystemRole(str,Enum):
    DEVELOPER = 'developer'
    BILLING_MANAGER = 'billing_manager'

    @classmethod
    def from_string(cls, role_name: str) -> 'SystemRole':
        for role in cls:
            if role.value == role_name:
                return role
        raise ValueError(f'Unknown system role name: {role_name}')


class SystemPermission(str, Enum):
    # User admin
    CREATE_USERS = 'create_users'
    READ_USERS = 'read_users'
    UPDATE_USERS = 'update_users'
    DELETE_USERS = 'delete_users'
    # Role admin
    ASSIGN_SYSTEM_ROLES = 'assign_system_roles'
    READ_SYSTEM_ROLES = 'read_system_roles'
    # Developer environments
    CREATE_DEVELOPER_ENVIRONMENTS = 'create_developer_environments'
    READ_DEVELOPER_ENVIRONMENTS = 'read_developer_environments'
    UPDATE_DEVELOPER_ENVIRONMENTS = 'update_developer_environments'
    DELETE_DEVELOPER_ENVIRONMENTS = 'delete_developer_environments'
    # Logging and monitoring
    VIEW_MONITORING_DASHBOARDS = 'view_monitoring_dashboards'
    # Billing projects
    CREATE_BILLING_PROJECTS = 'create_billing_projects'
    READ_ALL_BILLING_PROJECTS = 'read_all_billing_projects'
    UPDATE_ALL_BILLING_PROJECTS = 'update_all_billing_projects'
    DELETE_ALL_BILLING_PROJECTS = 'delete_all_billing_projects'
    ASSIGN_USERS_TO_ALL_BILLING_PROJECTS = 'assign_users_to_all_billing_projects'

    @classmethod
    def from_string(cls, permission_name: str) -> 'SystemPermission':
        for permission in cls:
            if permission.value == permission_name:
                return permission
        raise ValueError(f'Unknown system permission name: {permission_name}')


system_role_permissions = {
    SystemRole.DEVELOPER: [
        SystemPermission.CREATE_USERS,
        SystemPermission.READ_USERS,
        SystemPermission.UPDATE_USERS,
        SystemPermission.DELETE_USERS,
        SystemPermission.ASSIGN_SYSTEM_ROLES,
        SystemPermission.CREATE_DEVELOPER_ENVIRONMENTS,
        SystemPermission.READ_DEVELOPER_ENVIRONMENTS,
        SystemPermission.UPDATE_DEVELOPER_ENVIRONMENTS,
        SystemPermission.DELETE_DEVELOPER_ENVIRONMENTS,
        SystemPermission.VIEW_MONITORING_DASHBOARDS,
    ],
    SystemRole.BILLING_MANAGER: [
        SystemPermission.READ_USERS,
        SystemPermission.CREATE_BILLING_PROJECTS,
        SystemPermission.READ_ALL_BILLING_PROJECTS,
        SystemPermission.UPDATE_ALL_BILLING_PROJECTS,
        SystemPermission.DELETE_ALL_BILLING_PROJECTS,
        SystemPermission.ASSIGN_USERS_TO_ALL_BILLING_PROJECTS,
    ],
}
