# Large-scale Batch Operation

Operating Batch at or near its limit is semi-manual because Batch currently is unable to regulate the amount of incoming work it accepts.

CPU is the best readily available metric of Batch’s remaining capacity, obtained by:

```
kubectl top pods -l app=batch-driver
```

Less readily available metrics include the request latency (which might naturally vary by request type) and scheduling latency.  The goal is to operate Batch around 95% CPU.  When Batch becomes overwhelmed, CPU is saturated and request latency increases.  New requests inject work into the system, but time out and are retried due to latency, creating a bad feedback loop.

The incoming request rate from workers is controlled by the `internal-gateway`.  The `internal-gateway` is fixed at 3 replicates and imposes a per-instance, per-namespace request limit:

```
map $service $batch_driver_limit_key {
    "batch-driver" "$namespace";
    default "";  # no key => no limit
}

limit_req_zone $batch_driver_limit_key zone=batch_driver:1m rate=10r/s;

server {
  location {
    limit_req zone=batch_driver burst=20 nodelay;
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

## Quotas

When using Local SSDs on preemptible machines there are only two quotas that matter: "Preemptible
Local SSD (GB)" (`PREEMPTIBLE_LOCAL_SSD_GB`) and "Preemptible CPUs" (`PREEMPTIBLE_CPUS`). The former
is measured in GB, so you'll need 375 GB of quota for every machine with a Local SSD. The latter is
measured in cores. For example, if you are using a mix of n1 and n2 machines with 8 cores and 1
Local SSD, a 5000 machine (40,000 core) cluster will need:

- 1,875,000 GB of Preemptible Local SSD quota, and

- 40,000 cores of Preemptible CPUs quota.

In practice, we use Local SSD quota much faster than CPU quota. Google will freely gives us a 5,000
core quota in any given zone. We've also received quotas as high as 300,000 cores. Google is
hesitant to grant a quota of more than 400,000 GB in a zone. The largest Preemptible Local SSD quota
we have been granted in one zone is 640,000 GB.

We recommend requesting double your quota when you're using 80-90% of the current quota. Repeating
this process will generally allow you to quickly scale the cluster.

A higher or lower quota is requested by going to the "Quota Metric Details" page for a specific
quota (e.g. `PREEMPTIBLE_CPUS`), selecting the regions or zones of interest, and clicking "Edit
Quotas".

Quota requests during Eastern Time business hours appear to be approved faster. We presume this is
because our Technical Account Managers work in the Cambridge, MA Google office.
