# Flaky Tests

Basic runbook to identify flaky tests.

## DB Queries

Run these queries in the `batch` database

### Find total number of CI (PR) batches in the last 30 days

This is useful to compare with the failure counts in subsequent queries.

```sql
SELECT COUNT(*) as total_pr_batches
FROM batches
WHERE 
    user = 'ci'
    AND JSON_EXTRACT(attributes, '$.pr') IS NOT NULL
    AND time_created > UNIX_TIMESTAMP(NOW() - INTERVAL 30 DAY) * 1000;
```

### Find the total number of CI (PR) batches which failed in the last 30 days

```sql
SELECT COUNT(*) as total_failed_pr_batches
FROM batches
WHERE 
    user = 'ci'
    AND JSON_EXTRACT(attributes, '$.pr') IS NOT NULL
    AND time_created > UNIX_TIMESTAMP(NOW() - INTERVAL 30 DAY) * 1000
    AND EXISTS (SELECT 1 FROM jobs j WHERE j.batch_id = batches.id AND j.state = 'Failed');
```

### Find total number of CI (deploy) batches in the last 30 days

```sql
SELECT COUNT(*) as total_deploy_batches
FROM batches
WHERE 
    user = 'ci'
    AND JSON_EXTRACT(attributes, '$.deploy') IS NOT NULL
    AND time_created > UNIX_TIMESTAMP(NOW() - INTERVAL 30 DAY) * 1000;
```

### Find the total number of CI (deploy) batches which failed in the last 30 days

```sql
SELECT COUNT(*) as total_failed_deploy_batches
FROM batches
WHERE 
    user = 'ci'
    AND JSON_EXTRACT(attributes, '$.deploy') IS NOT NULL
    AND time_created > UNIX_TIMESTAMP(NOW() - INTERVAL 30 DAY) * 1000
    AND EXISTS (SELECT 1 FROM jobs j WHERE j.batch_id = batches.id AND j.state = 'Failed');
```

### Find test suites with high failure rates

We scan for batches in the last 30 days which:
- were run by `ci`
- Have a 'pr' attribute 
- Have a status of 'failure'.

We join with the jobs table to get the name of any failed jobs and aggregate by test name.

```sql
SELECT 
    ja.value as job_name,
    COUNT(*) as failure_count,
    MAX(b.id) as latest_batch_id
FROM 
    batches b 
JOIN 
    jobs j ON b.id = j.batch_id 
LEFT JOIN 
    job_attributes ja ON j.batch_id = ja.batch_id AND j.job_id = ja.job_id AND ja.key = 'name' 
WHERE 
    b.user = 'ci'
    AND j.state = 'Failed'
    AND (JSON_EXTRACT(b.attributes, '$.pr') IS NOT NULL OR JSON_EXTRACT(b.attributes, '$.deploy') IS NOT NULL)
    AND b.time_created > UNIX_TIMESTAMP(NOW() - INTERVAL 30 DAY) * 1000
GROUP BY 
    job_name
ORDER BY
    failure_count DESC
LIMIT 20;
```

### Find test suites with high failure rates during deploys

Note: this should have a lower false positive rate because we've hopefully weeded out genuine test failures.
But the sample size is a lot smaller

```sql
SELECT 
    ja.value as job_name,
    COUNT(*) as failure_count,
    MAX(b.id) as latest_batch_id
FROM 
    batches b 
JOIN 
    jobs j ON b.id = j.batch_id 
LEFT JOIN 
    job_attributes ja ON j.batch_id = ja.batch_id AND j.job_id = ja.job_id AND ja.key = 'name' 
WHERE 
    b.user = 'ci'
    AND j.state = 'Failed'
    AND JSON_EXTRACT(b.attributes, '$.deploy') IS NOT NULL
    AND b.time_created > UNIX_TIMESTAMP(NOW() - INTERVAL 30 DAY) * 1000
GROUP BY 
    job_name
ORDER BY
    failure_count DESC
LIMIT 20;
```