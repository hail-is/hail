# Overview of the Batch Control Plane External and Internal Load Balancers

Traffic flows into the Kubernetes cluster through two points of ingress: `gateway`,
which receives traffic from the internet, and `internal-gateway`, which manages traffic
from batch workers to the services in Kubernetes.

These reverse proxies/load balancers handle traffic routing to the appropriate
namespace/service, manage TLS, perform additional authorization checks for non-prod
namespaces, and enforce rate limits.
Our reverse proxy of choice is [Envoy](https://www.envoyproxy.io/).

The general routing rules for the gateways are as follows (Kubernetes DNS provides addresses
for `Service`s in the form of `<service>.<namespace>.svc.cluster.local`):

### Gateway
- `<service>.hail.is/<path> => <service>.default.svc.cluster.local/<path>`
- `internal.hail.is/<dev-or-pr>/<service>/<path> => <service>.<dev-or-pr>.svc.cluster.local/<developer>/<service>/<path>`[^1]

[^1]: At time of writing, developers cannot currently sign in to PR namespaces through the
browser because they are not assigned a callback for GCP/Azure OAuth flows.


### Internal Gateway
- `<service>.hail/<path> => <service>.default.svc.cluster.local/<path>`
- `internal.hail/<dev-or-pr>/<service>/<path> => <service>.<dev-or-pr>.svc.cluster.local/<developer>/<service>/<path>`

For Envoy to properly pool connections to K8s services, it needs to know
which "clusters" (services) exist at any point in time. This list is static for
production services, but PR namespaces are ephemeral and are
created/destroyed by CI many times per day. In order to notify the gateways
of new namespaces/services, CI tracks which namespaces are active and periodically
updates a K8s `ConfigMap` with fresh Envoy configuration. The gateways, using the
[Envoy xDS API](https://www.envoyproxy.io/docs/envoy/latest/intro/arch_overview/operations/dynamic_configuration#xds-configuration-api-overview)
can dynamically load this new configuration as it changes without dropping existing traffic.
You can see CI's current view of the cluster's namespaces/services at ci.hail.is/namespaces
and can inspect the current Envoy config at ci.hail.is/envoy-config/gateway and
ci.hail.is/envoy-config/internal-gateway.

## Operations

Both gateways are **unmanaged** — they are not deployed by CI and must be updated manually.

### Updating the Envoy version

Envoy releases quarterly minor versions (`1.33`, `1.34`, …) and patch releases within each series.
Patch releases are security/bug fixes only and are safe to apply without reviewing breaking changes.
Minor version bumps may contain breaking changes and should be done deliberately after reading the
[Envoy version history](https://www.envoyproxy.io/docs/envoy/latest/version_history/version_history).

To update to the latest patch release within the current minor series:

```bash
make update-gateways-envoy
```

This updates the image tag in `gateway/deployment.yaml` and `internal-gateway/deployment.yaml`
and deploys both gateways via `kubectl apply`.

### Deploying gateway changes

To apply any change to the gateway (image version, listener config, TLS params, etc.):

```bash
make -C gateway deploy
make -C internal-gateway deploy
```

This renders the Jinja2 templates and runs `kubectl apply`, which triggers a zero-downtime
rolling update of the gateway pods.

Note: `make -C gateway deploy` does **not** update the xDS routing ConfigMaps — those are
managed automatically by CI every 10 seconds and do not require a manual deploy.

If something goes wrong, roll back to the previous deployment immediately:

```bash
kubectl rollout undo deployment gateway-deployment
kubectl rollout undo deployment internal-gateway
```

Kubernetes retains the previous ReplicaSet so this is instant and zero-downtime.

### Renewing TLS certificates

See [letsencrypt.md](letsencrypt.md). After updating the `letsencrypt-config` secret, restart
the gateway pods to pick up the new certificate:

```bash
kubectl rollout restart deployment gateway-deployment
```

This is a pod restart only — it does not change the Envoy image or any config.
