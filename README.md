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

now you can start the server with

```
python ci.py
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
