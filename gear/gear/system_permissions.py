from enum import Enum

# Note that the system permission values must match the entries in the system_permissions table in the auth database.
# Everything else (which roles have which permissions, who has which roles) is managed by table joins in the database at run time.


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
