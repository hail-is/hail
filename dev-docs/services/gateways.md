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
You can see CI's current view of the cluster's namespaces/services at ci.hail.is/namespaces.
