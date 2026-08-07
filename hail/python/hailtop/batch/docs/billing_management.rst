.. _sec-billing-management:

==================
Billing Management
==================

Overview
--------

Every job submitted to the Batch Service is associated with (and makes charges against) a **billing project**. 
Billing projects group users and spending together, and have a spending limit to prevent runaway costs.

**Quotes** sit above billing projects:
- Every billing project belongs to exactly one quote. 
- Every quote may contain many billing projects.

A quote has an ``authorized_amount``: the total that may be allocated between its billing projects. 
A quote also has an assigned set of **owners** and **managers** who administer it as well as global billing managers.

A billing project can be administered by quote owners and managers of the containing quote. 
when users are added to a billing project, they gain the ability to read the project's details and billing history, 
add users to the project, and submit jobs within the project.

As an example, a PI may hold a quote representing an overall grant or cost object. They
delegate to trusted quote managers to create billing projects under that quote for different subgroups
or purposes (e.g., one per collaborator team, one for production pipelines, one for exploratory analysis).
Each billing project will have its own spending limit, and the total is guaranteed never to exceed the quote's
overall ``authorized_amount``.

The INTERNAL Quote
------------------

Every Batch deployment includes a built-in quote named ``INTERNAL``. It is unlimited and acts as
the default container for billing projects that predate the quotes system, as well as for billing
projects created without specifying a quote. 

For example, all user-trial billing projects are created under the ``INTERNAL`` quote.

Deployments that do not wish to bother with quotes may continue unchanged. All billing projects would
be created, and remain, under ``INTERNAL`` with no overall global spending cap.

Invariants
----------

The system enforces the following invariants:

- **Sum of BP limits ≤ quote authorized_amount.** The sum of all billing project limits
  under a quote can never exceed the quote's ``authorized_amount``. This is checked on every
  create, limit edit, and move operation.
- **Unlimited billing projects require an unlimited quote.** A billing project with no spending
  limit can only exist under a quote that is itself unlimited (no ``authorized_amount`` set). 
- **Total spend within a quote cannot exceed the quote's authorized_amount.** The total spend
  across all billing projects under a quote can never exceed the quote's ``authorized_amount``.
  This is an outcome of per-billing-project spend not exceeding the billing project's limit,
  and the sum of all billing project limits being less than or equal to the quote's ``authorized_amount``.
- **A quote cannot be closed while it has open billing projects.** Billing projects must be closed
  or moved before the quote can be closed.

Roles and Permissions
---------------------

Global billing managers (``global_bm``) are a small group of designated administrators who hold the
``billing_manager`` system role. They have full access to all quotes and billing projects across the
deployment, and are the only users who can create new quotes.

Below ``global_bm``, roles are scoped to specific quotes and billing projects. 

- ``quote_owner`` and ``quote_manager`` are assigned per-quote.
- ``bp_member`` is the role of anyone in a billing project's user list.
- Roles and permissions are scoped to specific quotes and billing projects. A user may be a quote owner in
one quote and just a plain billing project member in another.

The table below lays out the permissions for each role type:

.. list-table::
   :header-rows: 1
   :widths: 40 15 15 15 15

   * - Permission
     - Global Billing Managers
     - Quote Owners
     - Quote Managers
     - Billing Project Members
   * - **Submit jobs** to a billing project†
     - ❌
     - ❌
     - ❌
     - ✅
   * - **Read job history** and see job details and logs in a billing project†
     - ❌
     - ❌
     - ❌
     - ✅
   * - View billing history across an entire quote
     - ✅
     - ✅
     - ✅
     - ❌
   * - View billing history for a billing project
     - ✅
     - ✅
     - ✅
     - ✅
   * - View quote details, quote-level event history, and billing project list
     - ✅
     - ✅
     - ✅
     - ❌
   * - View billing project details, billing project-level event history, and accrued cost
     - ✅
     - ✅
     - ✅
     - ✅
   * - Edit quote metadata (cost object, PI name, etc.)
     - ✅
     - ✅
     - ✅
     - ❌
   * - Create a billing project under a quote
     - ✅
     - ✅
     - ✅
     - ❌
   * - Edit billing project limits under a quote
     - ✅
     - ✅
     - ✅
     - ❌
   * - Edit billing project description
     - ✅
     - ✅
     - ✅
     - ✅
   * - Add billing project users ‡
     - ✅
     - 🔜
     - 🔜
     - 🔜
   * - Remove billing project users
     - ✅
     - ✅
     - ✅
     - ✅
   * - Request a billing project limit increase ‡
     - N/A
     - N/A
     - N/A
     - 🔜
   * - Close / reopen a billing project
     - ✅
     - ✅
     - ✅
     - ❌
   * - Move a billing project to a different quote †
     - ✅
     - ✅
     - ✅
     - ❌
   * - Close / reopen a quote
     - ✅
     - ✅
     - ❌
     - ❌
   * - Add quote managers ‡
     - ✅
     - 🔜
     - ❌
     - ❌
   * - Remove / change quote managers
     - ✅
     - ✅
     - ❌
     - ❌
   * - Create a new quote
     - ✅
     - ❌
     - ❌
     - ❌

.. note::

    **† Moving billing projects between quotes.**
    To move a billing project between quotes, you must be a ``quote_owner`` or ``quote_manager`` in
    both the source and destination quotes. The destination quote must have sufficient headroom to
    accommodate the billing project's limit.

.. note::

    **‡ Coming soon — user invitation flow.**
    Adding users to billing projects, adding quote managers, and requesting billing project limit
    increases will be handled via an invitation system in a future release. For now, these actions
    are performed by global billing managers on behalf of users.

Lifecycle
---------

Billing projects and quotes each have a state that controls what operations are permitted.

**Billing project states:**

- **open** — the normal operating state. Users can submit jobs.
- **closed** — no new job submissions are accepted. Running jobs continue to completion and
  continue to accrue cost. A closed billing project can be reopened. A billing project cannot
  be closed while it has running batches. Note that allocated but unspent money in a closed 
  billing project is still associated with the billing project and will continue to use up 
  headroom in the quote unless reduced and reallocated.

**Quote states:**

- **open** — the normal operating state. Billing projects can be created under the quote.
- **closed** — no new billing projects can be created under the quote. Existing billing projects
  must be closed (and therefore can no longer be spent against) or moved.
