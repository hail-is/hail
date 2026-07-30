.. _sec-billing-management:

==================
Billing Management
==================

Overview
--------

Every job submitted to the Batch Service is charged to a **billing project**. Billing projects
group users and spending together, and can carry a spending limit that prevents runaway costs.

**Quotes** sit above billing projects as a budget envelope. A quote has an
``authorized_amount`` — the total dollars that may be spent across all billing projects assigned to
it — and a set of **managers** who administer it. One or more billing projects live under each
quote; the sum of their individual limits is guaranteed never to exceed the quote's
``authorized_amount``.

In everyday use, a PI or project lead holds a quote representing their grant or cost object. They
create billing projects under that quote for different subgroups or purposes (e.g., one per
collaborator team, one for production pipelines, one for exploratory analysis), each with its own
spending limit.

The INTERNAL Quote
------------------

Every Batch deployment includes a built-in quote named ``INTERNAL``. It is unlimited and acts as
the default container for billing projects that predate the quotes system, as well as for billing
projects created without specifying a quote. If you see ``INTERNAL`` as the quote name — for
example, on the trial billing project created automatically when you sign up — this is expected
and means the billing project is managed by the system itself rather than sitting under a quote.

Deployments that do not use quotes at all continue to work unchanged: all billing projects simply
remain under ``INTERNAL`` with no global spending cap.

The Hierarchy
-------------

.. code-block:: text

    Quote  (authorized_amount, open/closed)
    ├── Billing Project A  (limit, users)
    ├── Billing Project B  (limit, users)
    └── Billing Project C  (unlimited, users)

- A quote contains one or more billing projects.
- A billing project belongs to exactly one quote at a time (it can be moved).
- Users are added to individual billing projects, not to a quote directly.
- An unlimited billing project can only exist under an unlimited quote.

Invariants
----------

The system enforces the following invariants:

- **Sum of BP limits ≤ quote authorized_amount.** The sum of all billing project limits
  under a quote can never exceed the quote's ``authorized_amount``. This is checked on every
  create, limit edit, and move operation.
- **Unlimited billing projects require an unlimited quote.** A billing project with no spending
  limit can only exist under a quote that is itself unlimited (no ``authorized_amount`` set). Only
  global billing managers can create unlimited billing projects.
- **Total spend within a quote cannot exceed the quote's authorized_amount.** The total spend
  across all billing projects under a quote can never exceed the quote's ``authorized_amount``.
  This is an outcome of per-billing-project spend not exceeding the billing project's limit,
  and the sum of all billing project limits being less than or equal to the quote's ``authorized_amount``.
- **A quote cannot be closed until all its billing projects are closed or moved.** Close or
  migrate each billing project first.

Roles and Permissions
---------------------

Billing project management uses a role hierarchy whereby users can have different roles for each quote
and billing project.
Quote-level roles control who can administer quotes and billing projects. The the billing project level
there is only one role, ``bp_member``, which is assigned to users who are added to a billing project's
user list.

Global billing managers (``global_bm``) are a small group of designated administrators who hold the
``billing_manager`` system role. They have full access to all quotes and billing projects across the
deployment, and are the only users who can create new quotes.

Below ``global_bm``, roles are scoped to a specific quote. ``quote_owner`` and ``quote_manager``
are assigned per-quote via the managers list; ``bp_member`` is anyone added to a billing project's
user list.

.. list-table::
   :header-rows: 1
   :widths: 40 15 15 15 15

   * - Permission
     - global_bm
     - quote_owner
     - quote_manager
     - bp_member
   * - **Submit jobs** to a billing project†
     - No
     - No
     - No
     - Yes
   * - **Read job history** and see job details and logs in a billing project†
     - No
     - No
     - No
     - Yes
   * - View billing history for a quote
     - Yes
     - Yes
     - Yes
     - No
   * - View billing history for a billing project
     - Yes
     - Yes
     - Yes
     - Yes
   * - View quote details, quote-level event history, and billing project list
     - Yes
     - Yes
     - Yes
     - No
   * - View billing project details, billing project-level event history, and accrued cost
     - Yes
     - Yes
     - Yes
     - Yes
   * - Edit quote metadata (cost object, PI name, etc.)
     - Yes
     - Yes
     - Yes
     - No
   * - Create a billing project under a quote
     - Yes
     - Yes
     - Yes
     - No
   * - Edit billing project limits under a quote
     - Yes
     - Yes
     - Yes
     - No
   * - Edit billing project description
     - Yes
     - Yes
     - Yes
     - Yes
   * - Add billing project users
     - Yes
     - Yes
     - Yes
     - Yes
   * - Remove billing project users
     - Yes
     - Yes
     - Yes
     - Yes
   * - Close / reopen a billing project
     - Yes
     - Yes
     - Yes
     - No
   * - Move a billing project to a different quote
     - Yes
     - Yes†
     - Yes†
     - No
   * - Close / reopen a quote
     - Yes
     - Yes
     - No
     - No
   * - Add / remove quote managers
     - Yes
     - Yes
     - No
     - No
   * - Create a new quote
     - Yes
     - No
     - No
     - No

.. note::

    **† Moving billing projects between quotes.**
    To move a billing project between quotes, you must be a ``quote_owner`` or ``quote_manager`` in
    both the source and destination quotes. The destination quote must have sufficient headroom to
    accommodate the billing project's limit.

Lifecycle
---------

Billing projects and quotes each have a state that controls what operations are permitted.

**Billing project states:**

- **open** — the normal operating state. Users can submit jobs.
- **closed** — no new job submissions are accepted. Running jobs continue to completion and
  continue to accrue cost. A closed billing project can be reopened. A billing project cannot
  be closed while it has running batches. Note that allocated but unspent money in a closed 
  billing project is still associated with the billing project and will continue to use up 
  headroom in the quote.

**Quote states:**

- **open** — the normal operating state. Billing projects can be created under the quote.
- **closed** — no new billing projects can be created under the quote. Existing billing projects
  and their users are unaffected. A quote cannot be closed until all of its billing projects have
  been closed or moved to another quote.
