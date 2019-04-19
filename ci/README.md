Getting Started Developing
---

get a [GitHub personal access token](https://github.com/settings/tokens), copy
it and place it (without trailing newlines) in a file called `oauth-token`:

```
pbpaste > oauth-token/oauth-token
```

To talk to the batch server in our k8s cluster and to receive notifications from
github, you'll need to start proxies to each, respectively. The following
command sets up a proxy to the batch server from your local port `8888`. If a
proxy was previously created by this command, it kills it.

```
make restart-batch-proxy
```

For `batch` and GitHub to contact your local instance, you need a proxy open to
the outside world. I spun up a very tiny (4USD per month) server in GCP to
redirect requests to my local machine. The following commands will poke a hole
in the firewall and open a proxy from `$HAIL_CI_REMOTE_PORT` to your local
`5000` port (the default one for `ci.py`). I use port 3000 on the remote server,
so please choose another one for yourself :). The `firewall-rules create` need
only be run once. Be sure to pick a unique name!

```
gcloud compute firewall-rules create \
  SOME_UNIQUE_NAME_FOR_THIS_RULE \
  --allow tcp:3001 \
  --source-tags=dk-test \
  --source-ranges=0.0.0.0/0 \
  --description="for testing pr builder"
HAIL_CI_REMOTE_PORT=3001 make restart-proxy
```


Webhooks can be configured in the Settings tab of a repo on GitHub.

Now you can start a local version of the hail-ci server:

```
HAIL_CI_REMOTE_PORT=3001 make run-local
```


Setting up a New Repo
---

Make sure you have a `hail-ci-build.sh` and a `hail-ci-build-image` (containing
the name of a docker image that the CI has privileges to access).

Execute `setup-endpoints.sh` to configure GitHub to send notifications of GitHub
behavior to the CI system.


Testing
---

The tests require two more oauth-tokens, see the <github-tokens/README.md>.

Run the tests:

```
make test-local
```

Buildable Repos ---

We can build any repo whose root contains the files:

 - `hail-ci-build-image` containing a publicly accessible docker image
   containing at least a `bash` installation at `/bin/bash` as well as `git`,
   `gcloud`, and `gsutil`. The author recommends
   [google/cloud-sdk:alpine](https://hub.docker.com/r/google/cloud-sdk/)

 - `hail-ci-build.sh` containing a script that, when executed in the image
   specified in `hail-ci-build-image`, produces exit code 0 for a passing commit
   and a non-zero exit code for a failing commit

pr-build-script
---

OUT OF DATE SECTION


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

External IP Addresses in GKE
---
https://stackoverflow.com/a/33830507/6823256

we can allocate one explicitly and then specify it in the `deployment.yaml`

How I Created a PR Deploy Service Account
---

This has creation privileges to the bucket where we publish Jars. It should
really only have privileges to create new jars in the `build/` folder. :shrug:.

NB: cannot delete, cannot replace, only create new files in the bucket

```sh
NAME=ci-deploy-0-1--hail-is-ci-test
gcloud iam service-accounts create ${NAME}
gsutil iam ch \
  serviceAccount:${NAME}@broad-ctsa.iam.gserviceaccount.com:objectCreator \
  gs://hail-ci-test
# really should only have list so that you can copy to an object path rather
# than to the root of a bucket, but, ugh, too many things to do not enough time
gsutil iam ch \
  serviceAccount:${NAME}@broad-ctsa.iam.gserviceaccount.com:objectViewer \
  gs://hail-ci-test
gcloud iam service-accounts keys create \
  ${NAME}.key \
  --iam-account ${NAME}@broad-ctsa.iam.gserviceaccount.com
kubectl create secret \
  generic \
  ${NAME}-service-account-key \
  --from-file=./${NAME}.key
```

Useful Shell Functions
---

probably need bash or zsh to execute correctly, but I use these to quickly learn
information about pods in our system:

```
function hailcipod () {
    PODS=$(kubectl get pods -l app=hail-ci --no-headers)
    [[ $(echo $PODS | wc -l) -eq 1 ]] || exit -1
    echo $PODS | awk '{print $1}'
}

function jobpod () {
    PODS=$(kubectl get pods --no-headers | grep -Ee "job-$1-.*(Running|Pending)")
    [[ $(echo $PODS | wc -l) -eq 1 ]] || exit -1
    echo $PODS | awk '{print $1}'
}
```

Example usage:

```
kubectl logs -f $(hailcipod) # nb: logs -f doesn't work with -l
kubectl logs $(jobpod 542)
```

Static IP for Hail CI Service
---
You must use a region address. I do not know why, but global addresses cause k8s
to report that the IP is not static or not assigned to the given load
balancer. :shrug:.
```
gcloud compute addresses create hail-ci-0-1 \
    --region us-central1
```
