Getting Started Developing
---
```
conda env create hail-ci -f environment.yaml
source activate hail-ci
pip install /path/to/batch
```
now you can start the server with
```
python ci.py
```

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
