# Google Cloud Cookbook

Executable snippets for certain operational tasks. These snippets will likely bitrot and otherwise
be unreliable. The reader is recommended to test the snippet before use.

## Delete All Detached Disks

There were approximately 240,000 disks accidentally left unattached in a project. This script was
used to delete them. We only delete 990 at a time because there is a "queries per second" quota that
limits how many queries we can make every 100 seconds. We empirically determined that deleting 990
instances at a time did not exceed the quota.

```
for i in $(seq 0 280)
do
    gcloud compute disks list \
        --filter='LAST_ATTACH_TIMESTAMP < LAST_DETATCH_TIMESTAMP and name ~ "^batch-disk" and STATUS=Ready' \
        --limit=990 \
      | tail -n +2 \
      > /tmp/disks
    for zone in $(cat /tmp/disks | awk '{print $2}' | uniq )
    do
        echo $zone
        awk '$2=="'$zone'" {print $1}' /tmp/disks \
          | grep -Ee '^batch-disk' \
          | xargs /bin/bash -c 'yes | gcloud compute disks delete '--zone=$zone' $*' % &
    done
    wait
done
```
