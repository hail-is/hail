{% extends "dev-docs-page.html" %}
{% block title %} Large-scale Batch Operation {% endblock %}

{% block docs_content %}
Operating Batch at or near its limit is semi-manual because Batch currently is unable to regulate the amount of incoming work it accepts.

CPU is the best readily available metric of Batch’s remaining capacity, obtained by:

```
kubectl top pods -l app=batch-driver
```

Less readily available metrics include the request latency (which might naturally vary by request type) and scheduling latency.  The goal is to operate Batch around 95% CPU.  When Batch becomes overwhelmed, CPU is saturated and request latency increases.  New requests inject work into the system, but time out and are retried due to latency, creating a bad feedback loop.

The incoming request rate from workers is controlled by the `internal-gateway`.  The `internal-gateway` is fixed at 3 replicates and imposes a per-instance request limit:

```
limit_req_zone global zone=limit:1m rate=45r/s;

server {
  location {
    limit_req zone=limit burst=20 nodelay;
  }
}
```

The rate limit should be tuned so that Batch runs at ~95% CPU at the maximum request rate.  Changes to Batch will change the maximum acceptable request rate, so this setting should be revisited periodically, esp. when running a new large-scale workload.

When Batch is running a large-scale workload, the behavior under the maximum request rate can be simulated as follows:
Delete the `batch-driver` deployment.
Wait.  Jobs will complete on the workers, building up a pool of jobs that need to be marked complete.  Since the workers are unable to connect to the `batch-driver`, they will retry with back-off.
Redeploy `batch-driver`.
Watch the behavior under maximum load.  Normally the allocated cores increases for some time, as the job complete requests are handled.  Then, once the load decreases from job complete messages, the scheduler will be able to catch up and fill the free worker cores.

You can inspect the `internal-gateway` logs to determine if the request rate is  maximized.  When the maximum request rate is exceeded, `internal-gateway` nginx returns 503 and logs a message.

To determine if the cluster size is at the maximum, check the CPU and `internal-gateway` request rate when the cluster is not growing, but just replacing preempted nodes.  The CPU should not be pegged, and `internal-gateway` should reject requests at most in transient bursts.  In general, the load will be much lower at equilibrium because filling an empty node requires many operations.
{% endblock %}
