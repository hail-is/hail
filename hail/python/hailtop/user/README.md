## Users

Associate a user id to kubernetes service account, google cloud service account, and a google cloud bucket, by creating those resources, if that user id does not have a record of possession.

### Run

```sh
make user name=<some_name> op=<create|delete>
```

### Test

```sh
make test-local
```
