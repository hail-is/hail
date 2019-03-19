## Users

Associate a user id to kubernetes service account, google cloud service account, and a google cloud bucket, by creating those resources, if that user id does not have a record of possession.

### Run

```sh
BATCH_USE_KUBE_CONFIG=1 SQL_HOST=127.0.0.1 python user_secrets.py "<some_user_id>"
```

### Test

```sh
BATCH_USE_KUBE_CONFIG=1 SQL_HOST=127.0.0.1 python test_users.py
```

To run on localhost, specify `BATCH_USE_KUBE_CONFIG=1 SQL_HOST=127.0.0.1` and use [Cloud SQL Proxy](https://cloud.google.com/sql/docs/mysql/sql-proxy)
