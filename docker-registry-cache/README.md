# Docker Registry Cache

This directory contains the Kubernetes deployment for a Docker registry cache/mirror that helps reduce Docker Hub rate limiting issues when worker VMs pull images.

## Overview

The Docker registry cache acts as a proxy cache for `docker.io`. When worker VMs pull images from Docker Hub, they will first check the cache. If the image is cached, it's served from the cache. If not, the cache pulls it from Docker Hub, caches it, and serves it to the worker.

This significantly reduces the number of requests to Docker Hub, helping avoid rate limits when behind CloudNAT.

## Architecture

- **Deployment**: Runs 2 replicas of `registry:2` with proxy cache configuration
- **Service**: Exposed as an internal LoadBalancer service accessible from worker VMs
- **Storage**: Uses `emptyDir` for caching (ephemeral, but sufficient for caching)

## Deployment

```bash
cd docker-registry-cache
make deploy
```

After deployment, get the internal IP address of the service:

```bash
kubectl get service docker-registry-cache -n default
```

## Configuration

The registry cache IP address needs to be configured in the worker VM startup scripts. This is done via:

1. **GCP**: Pass `docker_registry_cache_ip` as a metadata attribute when creating VMs
2. **Azure**: Add `docker_registry_cache_ip` to the userdata JSON when creating VMs

The worker VM startup scripts will automatically configure the Docker daemon to use this cache as a registry mirror if the IP is provided.

## How It Works

1. Worker VMs are configured with a registry mirror pointing to the cache service
2. When Docker pulls an image from `docker.io`, it first tries the cache
3. If the image is in the cache, it's served immediately
4. If not, the cache pulls from Docker Hub, caches it, and serves it
5. Subsequent requests for the same image are served from cache

## Notes

- The cache uses `emptyDir` storage, so cached images are lost when pods restart. This is acceptable for a cache.
- For persistent caching, consider using a PersistentVolume, but this adds complexity and may not be necessary.
- The registry cache is configured to proxy `registry-1.docker.io` (Docker Hub's official registry).
