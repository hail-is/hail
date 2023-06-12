CREATE INDEX aggregated_billing_project_user_resources_v2_all ON `aggregated_billing_project_user_resources_v2` (`billing_project`, `user`, resource_id, `token`);
CREATE INDEX aggregated_billing_project_user_resources_by_date_v2_token ON `aggregated_billing_project_user_resources_by_date_v2` (billing_date, billing_project, `user`, `token`);
