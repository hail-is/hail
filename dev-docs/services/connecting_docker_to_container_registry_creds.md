# Connecting Docker to Container Registry Credentials

In GCP, something like:
```
gcloud auth login
gcloud auth configure-docker us-central1-docker.pkg.dev # Depending on where GAR is hosted
```


In Azure, something like:
```
az login
az acr login --name haildev # the container registry name is findable in the azure console
```

## Making skopeo work with azurecr.io auth

Even after az acr login, skopeo may have auth-missing issues. In this case you may need to do an extra step for skopeo 
to accept the azure login. See [this](https://github.com/containers/skopeo/issues/1534) skopeo issue from an illustrious
and wise questioner.

For me, the workaround advice given early in the issue above worked (see [this](https://github.com/containers/skopeo/issues/1534#issuecomment-1231287666) comment):

- Update `~/.docker/config.json`
- Add a `"haildev.azurecr.io": "desktop"` entry into the `credHelpers` object
- ie after editing it should look something like:
```json
{
  "credStore": "desktop",
  "credHelpers": {
    "[REDACTED].azurecr.io": "desktop"
  }
}
```

Note: I'm not 100% sure about this, but this might help only if you are using docker desktop, so your milage may vary.
