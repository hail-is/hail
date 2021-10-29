# GSA Key Rotation

Every identity in batch, user or robot, has a corresponding Google Service
Account (GSA). A service or user job can authenticate with Google as
a specific service account with a Service Account Key. When a new user is
created, `auth` creates a service account, requests a key for that account, and
stores the key as a Kubernetes secret with the name `<username>-gsa-key`.

Service account keys are confidential and should be rotated at least every 90
days to mitigate the danger of attack if a key is leaked. The key rotation
strategy consists of two parts:

1. For each Google service account whose newest key is at least 60 days old,
create a new key and update the Kubernetes secret to reflect the new value.
2. For each Google service account whose newest key is older than 30 days old,
delete all but the newest key.

Step 1 ensures that all keys stored in k8s secrets are no more than two months old,
while Step 2 ensures that any key that is not in use is deleted.
Step 2 does **not** act on service accounts that just underwent Step 1 because
the old keys might still be in use by active pods/jobs. We assume that no
pod/job will run for longer than 30 days.

We consider the "active" key for a service account to be the key stored in a
k8s secret. If no secret is associated with a service account, we take the active
key to be the most recently created user-managed[^1] key. We then consider
service accounts to be in one of four states:

- Expired: The active key in Kubernetes is older than 90 days and should be
rotated immediately.
- In Progress: The active key was created in the past 30 days and there exist
old keys that may still be in use.
- Ready for Delete: The active key was created more than 30 days ago and there
exist old keys that can be safely deleted.
- Up to Date: The active key is valid and there are no redundant keys.


## Rotating keys

Make sure that you are first authenticated with the correct GCP project by
running `gcpsetcluster <PROJECT>` (you need to have sourced
`$HAIL/devbin/functions.sh` to run this command). Then run

```
python3 $HAIL/devbin/rotate_keys.py <PROJECT>
```

Don't worry, this won't do anything scary on its own!

The script's initial output shows the following:
- Which GSA key secrets in Kubernetes belong to which Google service accounts.
- Which secrets have no corresponding service account. This likely means
  the original service account was deleted without deleting its secret.
  If this occurs, the secret should probably be deleted, but check to ensure that
  nothing still depends on it first.
- Which service accounts have no corresponding key secret in Kubernetes. This
  can be OK, such as with the terraform service account. However, any user
  accounts should not be in this category, and it might indicate that they
  weren't properly deleted.

Each service account is listed with its rotation state.

NOTE: `rotate_keys.py` *only* checks secrets with a `key.json` field. This
is an invariant of `auth` and `terraform`-created accounts, but might not catch
legacy secrets.

Additionally, the script will warn of any service accounts with more than 1
corresponding key secret per namespace. This should not happen with the
exception of the test user, which is used in dev namespaces instead of other
robot service accounts. There should be no redundant secrets in the default
namespace.


### Updating keys

The `update` flow steps through each service account and lists its GSA keys,
sorted in reverse order of their date of creation.
Any keys that are found in Kubernetes will list the name and namespace
of the corresponding secret at the end of the row. Developer and test accounts
might have secrets in multiple namespaces (default and dev / test), but user
accounts should only have secrets present in `default`. The script will prompt
for each service account if you would like to create a new key. Enter `yes`
and the script will create a new GSA key and update any relevant secrets. Any
other input will do nothing. The output should then show the updated key at
the top of the list and all Kubernetes secrets pointing to the new key.
The script refreshes its Kubernetes secrets after updating them so this should
accurately reflect the state of the cluster.
Continue until all service/user keys have been updated.

### Deleting keys

Remember to complete this step no sooner than 2 days after updating keys.

The `delete` flow steps through each service account similarly to the update
flow. Entering `yes` for a particular service account will delete all
user-managed keys belonging to the service account except for the most
recently created key. The script will allow deletion of old keys for service
accounts in the Ready for Delete or In Progress state, though the latter requires
additional confirmation to prove you are confident that old keys are not still
in use.
Continue until all old service/user keys have been deleted.

[^1]: All GSA keys that we create are considered "user-managed". We are responsible
for creating, deleting, rotating and revoking them. In addition to user-managed
keys, Google has system-managed keys. These keys are managed by Google and thus
we cannot delete them. An attempt to delete system managed keys will output a
warning and the keys will remain in the key list for the service account. If the
only remaining keys after a delete system managed + the most recent user-managed
key, then everything has completed successfully.
