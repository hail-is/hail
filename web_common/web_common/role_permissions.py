system_roles = [
    'developer',
    'billing_manager',
]

system_permissions = [
    # User admin
    'create_users',
    'read_users',
    'update_users',
    'delete_users',
    'assign_system_roles',
    # Developer environments
    'create_developer_environments',
    'read_developer_environments',
    'update_developer_environments',
    'delete_developer_environments',
    # Logging and monitoring
    'view_monitoring_dashboards',
    # Billing projects
    'create_billing_projects',
    'read_all_billing_projects',
    'update_all_billing_projects',
    'delete_all_billing_projects',
    'assign_users_to_all_billing_projects',
]

system_role_permissions = {
    'developer': [
        'create_users',
        'read_users',
        'update_users',
        'delete_users',
        'assign_system_roles',
        'create_developer_environments',
        'read_developer_environments',
        'update_developer_environments',
        'delete_developer_environments',
        'view_monitoring_dashboards',
    ],
    'billing_manager': [
        'read_users',
        'create_billing_projects',
        'read_all_billing_projects',
        'update_all_billing_projects',
        'delete_all_billing_projects',
        'assign_users_to_all_billing_projects',
    ],
}
