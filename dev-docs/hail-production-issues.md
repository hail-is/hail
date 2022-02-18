# Process for Production Issues

When a production issue is entered into this system, it must be given a priority. These are the priorities [1]:
1. P1: Service is unusable or business impact is critical.
2. P2: Service use severely impaired or business impact is severe.
3. P3: Service use partially impaired or business impact is minor.

Examples of P1 issues:
- A user cannot submit a batch.
- A user cannot access the batch.hail.is UI.
- A majority of a user's jobs are failing due to non-user errors.
- Hail team is incurring unexpected cost 100 USD/hour or 500 USD/day. [2]
- A private key has been leaked.
- An exploit granting arbitrary code execution.
- An exploit is discovered that grants access to a user's data.
- CI is not merging PRs.

Examples of P2 issues:
- A user is experiencing latencies higher than 5 seconds to load a UI page.
- User-facing API endpoint latencies are 5x the expected latency (e.g. last month's average latency).
- A user is experiencing high latency for image pulling, job scheduling, or batch submission.
- Hail team is incurring unexpected cost >25 USD/hour or 100 USD/day.
- An exploit is discovered that, under special circumstances, could leak a private key.
- A bug present in the current version of the Hail PyPI library prevents use, but previous versions
  still work.
- CI transiently fails one in four PRs.

P3 issues are usually P1 or P2 issues that have been mitigated.

The urgency of the priorities:
1. A developer should immediately interrupt their daily work to address a P1 issue.
2. A developer should address a P2 issue within three business days.
3. A developer should resolve a P3 issue within twenty business days.

P1 issues can be mitigated such that they are re-characterized as a P2 issue. Likewise, P2 issues
may be mitigated into P3 issues.

Examples of P1 mitigations:
- A user cannot submit batches due to a bug in hailtop. If the user can revert to a previous version
  of Hail, then the issue is mitigated.
- Batch is off the rails churning through machines without doing any billable work. If freezing
  Batch stops the churn, then the issue is mitigated.
- A private key is leaked. If the key is rotated and the leaking channel is temporarily shut off,
  then the issue is mitigated to P2.

Examples of P2 mitigations:
- Due to database memory or CPU limitations, one in four CI jobs are failing. If doubling database
  memory or CPU resolves the issue, then the issue is mitigated.
- A bug is causing one in four CI jobs to fail. If a small number of tests are failing, disabling
  those tests mitigates the issue.
- Batch is experiencing higher than usual latencies for all user-facing APIs. If shrinking the
  cluster restores latencies to an acceptable range, then the issue is mitigated.


[1] Inspired by [Google's priorities](https://cloud.google.com/support/docs/procedures#support_case_priority).

[2] I arrived at these numbers by considering the cost to Hail team if a P1 issue goes unresolved
    for 16 hours (e.g. 5p - 9a) or three days (a long weekend). A 1500 USD unexpected charge is not
    a problem for Hail's budget, but repeated, unaddressed P1 issues would accumulate into a real
    problem.

