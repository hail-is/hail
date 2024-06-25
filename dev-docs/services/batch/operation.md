# Large-scale Batch Operation

Operating Batch at or near its limit is semi-manual because Batch currently is unable to regulate the amount of incoming work it accepts.

Alerts in the #grafana Zulip stream fire when the Batch Driver or workers log errors. You can follow these alerts to the
corresponding metric at [grafana.hail.is](grafana.hail.is). The upper-left corner of the panels for the alerts should have a link to
the logs in Google Logging.

Other than error logs, CPU usage is the best measure of Batch's remaining capacity. CPU usage for the Batch Driver and the database
are displayed on the [performance panel](https://grafana.hail.is/d/TVytxvb7z/performance) of Grafana[^1].

[^1]: CPU utilization is listed as a percentage of its *limit*, and at time of writing the CPU limit for the Batch Driver is set
to `1.5vCPU`. As such, 60% utilization is using almost a full core, which is all we can reasonably expect from a single python process.

The goal is to operate the Batch Driver around 95% CPU. When Batch becomes overwhelmed, CPU is saturated and request latency increases.
New requests from workers inject work into the system, but time out and are retried, creating a bad feedback loop.

Driver CPU usage is driven largely job scheduling and job completion.
Floods of job completion, the main source of traffic from workers -> driver, are controlled by rate limits in the `internal-gateway`.
Altering the rate limit at full load is the most direct way to modulate system throughput.

You can reduce load on the system in one of two ways:
1. Gradual load shedding: Reduce the maximum cluster size to below its current size. As instances
  die off naturally, the Driver will not replenish them, and load will ease over time. You can also
  manually kill off `preemptible` instances if necessary to improve cluster health.
2. Throttle Mark Job Complete (MJC): reduce the rate limit [here](https://github.com/hail-is/hail/blob/923cc552c9460527101139970d46364948e4f6a8/ci/ci/envoy.py#L176)
  and manually redeploy CI with `make -C ci deploy NAMESPACE=default`.[^2] Note that this may need to be
  paired with reducing the cluster size as rate-limited instances are cost centers that we do not charge
  users for.

[^2]: This can be undone if CI merges a new commit to `main` and redeploys itself.

Other indicators of service health include API request latency (endpoints on the Batch Driver should last <1s),
and the size of the DB Connection Queue. Queued database transactions imply that `internal-gateway` is not applying
sufficient back-pressure to protect the Driver.

The rate limit should be tuned so that Batch runs at ~95% CPU at the maximum request rate.
Changes to Batch will change the maximum acceptable request rate, so this setting should be revisited periodically,
esp. when running a new large-scale workload.

To determine if the cluster size is at the maximum, check the CPU and rate limiting when the cluster is not growing,
but just replacing preempted nodes. The CPU should not be pegged, and `internal-gateway` should reject requests at most in transient bursts.
In general, the load will be much lower at equilibrium because filling an empty node requires many operations.

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
