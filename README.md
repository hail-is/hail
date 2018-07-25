Getting Started Developing
---
```
conda env create hail-ci -f environment.yaml
source activate hail-ci
pip install /path/to/batch
```

get a [GitHub personal access token](https://github.com/settings/tokens), copy
it and place it (without trailing newlines) in a file called `oauth-token`:

```
pbpaste > oauth-token
```

then you gotta create a file with a secret string that protects the endpoints
that can change statuses. This isn't really important if you only run the server
locally.

```
printf 'SOME VERY SECRET STRING THAT ONLY YOU KNOW' > secret
```

now you can start the server with

```
python ci/ci.py
```

I spun up a very tiny (4USD per month) server in GCP to redirect requests to my
local machine:

```
gcloud compute \
  --project "broad-ctsa" \
  ssh \
  --zone "us-central1-f" \
  "dk-test" \
  --ssh-flag='-R 3000:127.0.0.1:5000' \
  --ssh-flag='-v'
```

I already added an web hook to github to send a request to this machine's IP on
port 3000 whenever there is a `pull_request` or `push` event on my
`docker-build-test` repository. The latter gets me master branch change
notifications.

Any repository would work, and you can set up a webhook through the github web
UI. Just use a port other than mine (3000)!

pr-build-script
---
The `pr-build-script` has access to the following shell variables:

 - `SOURCE_REPO_URL` -- a URL pointing to the `.git` file of the repo containing
   the source branch of the PR, e.g. `https://github.com/danking/hail.gt`

 - `SOURCE_BRANCH` -- the name of the source branch of the PR,
   e.g. `faster-bgen`

 - `SOURCE_SHA` -- the full hash of the commit on the source branch of the PR
   that the CI system wants built

 - `TARGET_BRANCH` -- the name of the target branch of the PR, e.g. `master`

 - `TARGET_SHA` -- the full hash of the commit on the target branch of the PR
   that the CI system wants to merge into the source branch

Developer Tips
---

https://developer.github.com/v3/repos/hooks/#list-hooks

so, very confusingly, github returns 404 if you're not authenticated rather than
returning 403, you gotta authenticate for the below stuff

list hooks for some repo
```
curl -LI api.github.com/repos/danking/docker-build-test/hooks
```

create ahook
```
curl api.github.com/repos/danking/docker-build-test/hooks \
     -X POST \
     -H "Content-Type: application/json" \
     -d '{ name: "hail-ci"
         , events: ["pull_request", "push"]
         , active: true
         , config: { url: 35.232.159.176:3000
                   , content_type: "json"
                   }
         }'
```


open a port on a GCP instance
```
gcloud compute firewall-rules create \
  dk-testing-port \
  --allow tcp:3000 \
  --source-tags=dk-test \
  --source-ranges=0.0.0.0/0 \
  --description="for testing pr builder"
```


forward a port from a GCP page
```
gcloud compute \
  --project "broad-ctsa" \
  ssh \
  --zone "us-central1-f" \
  "dk-test" \
  --ssh-flag='-R 3000:127.0.0.1:5000' \
  --ssh-flag='-v'
```

grant a service account object creation (but not overwrite or read) to a
particular bucket:

```
gsutil iam ch [MEMBER_TYPE]:[MEMBER_NAME]:[ROLE] gs://[BUCKET_NAME]
# e.g.
gsutil iam ch \
  serviceAccount:hail-ci-0-1@broad-ctsa.iam.gserviceaccount.com:objectCreator \
  gs://hail-ci-0-1
```

NB: it's not `roles/storage.objectCreator`, which is the [iam role
name](https://cloud.google.com/storage/docs/access-control/iam-roles). I'm not
sure what's going on here :shrug:.

Random Things
---

 - [Google python client library reference](https://googlecloudplatform.github.io/google-cloud-python/)


Secrets
---

[manual for k8s secrets](https://kubernetes.io/docs/concepts/configuration/secret/)

[design doc for k8s secrets
API](https://github.com/kubernetes/community/blob/master/contributors/design-proposals/auth/secrets.md)
(different from the Docker Swarm secrets)

I will add a hail-ci secret (probably a gcloud service account key) that we
should eliminate and re-create when we get serious about this cluster.

gcloud can [authorize from a service account key
file](https://cloud.google.com/sdk/gcloud/reference/auth/activate-service-account)

Create a key for the service account `hail-ci-0-1`
```
gcloud iam service-accounts keys create \
  hail-ci-0-1.key \
  --iam-account hail-ci-0-1@broad-ctsa.iam.gserviceaccount.com
```

activate a service account from a key file:
```
gcloud auth activate-service-account \
  hail-ci-0-1@broad-ctsa.iam.gserviceaccount.com \
  --key-file /secrets/hail-ci-0-1.key
```

Batch
---

here's how you use volume specs

```ipython
In [1]: from batch.client import *
In [2]: c = BatchClient('http://localhost:8888')
In [3]: j = c.create_job('google/cloud-sdk:alpine',
   ...:                  ['/bin/bash',
   ...:                   '-c',
   ...:                   'ls /secrets && gcloud auth activate-service-account hail-ci-0-1@broad-ctsa.iam.gserviceaccount.com --key-file=/secrets/hail-ci-0-1.key'],
   ...:     volumes=[{ 'volume': { 'name' : 'hail-ci-0-1-service-account-key',
   ...:                            'secret' : { 'optional': False, 'secretName': 'hail-ci-0-1-service-account-key' }},
   ...:                'volume_mount' : {'mountPath' : '/secrets', 'name': 'hail-ci-0-1-service-account-key', 'readOnly': True } }])
```

NB: `secretName` is camel case, matching the style used in YAML, not in
python. Same for mountPath and friends.
