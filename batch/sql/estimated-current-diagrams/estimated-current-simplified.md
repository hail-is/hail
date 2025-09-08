# Batch Database UML Diagram (simplified)

This diagram is is a simplified version of `estimated-current.md`, whose description can be found on that diagram. Like `estimated-current.md`, this serves to be a visual representation of `estimated-current.sql`, however it has been simplified by **only including the primary and foreign keys for a given table**.

Conventions:
- Primary keys are listed in order as the first rows of each table, and are denoted by **bold text**.
- Foreign keys are denoted by `FK` in the rightmost column, and are further denoted via connections between two tables.
```mermaid
---
config:
  layout: dagre
---
erDiagram
direction TB

	%% For future editors: I encourage you to try and find a way to make this Mermaid diagram more organized.
	%% It is, as of now, a task beyond my capabilities.

	inst_colls {
		varchar(255) **name** ""
	}
	pools {
		varchar(255) **name** FK ""
	}
	user_inst_coll_resources {
		varchar(100) **user** ""
		varchar(255) **inst_coll** "FK"
		int **token** ""
	}
	batches {
		bigint **id** ""
		varchar(100) billing_project FK ""
	}
	jobs {
		bigint **batch_id** FK ""
		int **job_id** ""
		varchar(255) inst_coll FK ""
		int update_id FK ""
		int job_group_id FK ""
	}
	batch_updates{
		bigint **batch_id** FK ""
		int **update_id** ""
		int **start_job_group_id** ""
		int **start_job_id** ""
	}
	job_parents {
		bigint **batch_id** FK ""
		int **job_id** FK ""
		int **parent_id** FK ""
	}
	job_attributes {
		bigint **batch_id** FK ""
		int **job_id** FK ""
		varchar(100) **key** ""
	}
	job_group_attributes {
		bigint **batch_id** FK ""
		int **job_group_id** FK ""
		varchar(100) **key**  ""
	}
	batch_bunches {
		bigint **batch_id** FK ""
		int **start_job_id** FK ""
	}
	attempt_resources {
		bigint **batch_id** FK ""
		int **job_id** FK ""
		varchar(40) **attempt_id** FK ""
		int **resource_id** FK ""
	}
	attempts {
		bigint **batch_id** FK ""
		int **job_id** FK ""
		varchar(40) **attempt_id** ""
		varchar(100) instance_name FK ""
	}
	billing_project_users {
		varchar(100) **billing_project** FK ""
		varchar(100) **user** ""
	}
	billing_projects {
		varchar(100) **name** ""
	}
	job_groups {
		bigint **batch_id** FK ""
		int **job_group_id** ""
	}
	aggregated_billing_project_user_resources_v3 {
		VARCHAR(100) **billing_project** FK ""
		VARCHAR(100) **user** ""
		INT **resource_id** ""
		INT **token** ""
	}
	resources {
		varchar(100) **resource** ""
	}
	instances {
		varchar(100) **name** FK ""
	}

	%% batches ||--o{ attempt_resources : "id"
	batches ||--o{ batch_updates : "id"
	%% batch_updates ||--o{ jobs : "batch_id, update_id"
	jobs ||--o{ attempt_resources : "batch_id,job_id"
	attempts ||--o{ attempt_resources : "batch_id,job_id,attempt_id"
	resources ||--o{ attempt_resources : "resource"
	%% batches||--o{attempts:"id"
	jobs||--o{attempts:"batch_id,job_id"
	instances||--o{attempts:"instance_name"
	jobs||--o{batch_bunches:"batch_id,job_id"
	%% user_resources ||--o{ batches:"user"
	billing_projects ||--o{ batches:"name"
	billing_projects ||--o{ aggregated_billing_project_user_resources_v3:"name"
	resources ||--o{ aggregated_billing_project_user_resources_v3:"resource_id"
	billing_projects ||--o{ billing_project_users :"name"
	%% batches||--o{jobs:"id"
	%% instances||--o{instances_free_cores_mcpu:"name"
	%% batches||--o{job_attributes:"id"
	jobs||--o{job_attributes:"batch_id, job_id"
	%% batches||--o{job_group_attributes:"id"
	job_groups||--o{job_group_attributes:"batch_id,job_group_id"
	batches||--o{job_groups:"id"
	batch_updates ||--o{job_groups:"batch_id,update_id"
	%% batches||--o{job_parents:"id"
	jobs||--o{job_parents:"batch_id, job_id"
	inst_colls ||--o{ pools:"name"
	inst_colls ||--o{ instances:"name"
	inst_colls ||--o{ user_inst_coll_resources:"name"
	job_groups ||--o{ jobs : "batch_id, job_group_id"
```
