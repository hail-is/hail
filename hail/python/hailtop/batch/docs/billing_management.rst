.. _sec-billing-management:

==================
Billing Management
==================

.. note::

    **Coming soon.** Quotes and the billing management features described on this page are being
    rolled out and are not yet available.

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
When users are added to a billing project, they gain the ability to submit jobs within the project,
view the project's details, event history, and accrued cost, view their own billing history in the
project, edit the project's description, and leave the project.

As an example, a PI may hold a quote representing an overall grant or cost object. They
delegate to trusted quote managers to create billing projects under that quote for different subgroups
or purposes (e.g., one per collaborator team, one for production pipelines, one for exploratory analysis).
Each billing project will have its own spending limit, and the sum of those limits is guaranteed never
to exceed the quote's overall ``authorized_amount``.

The INTERNAL Quote
------------------

Every Batch deployment includes a built-in quote named ``INTERNAL``. It is the only quote that
may be unlimited (that is, have no ``authorized_amount``), and it acts as the default container
for billing projects that predate the quotes system.

A quote must be specified explicitly whenever a billing project is created. The only exception is
the legacy (pre-quotes) billing project creation API, which is deprecated. When it is called without
a quote by a caller holding the ``create_billing_projects`` system permission (global billing managers
and the ``auth`` service, which creates user-trial billing projects at sign-up), the billing project is
placed under the ``INTERNAL`` quote.

Deployments that do not wish to bother with quotes can create all of their billing projects under
``INTERNAL``, which has no overall spending cap.

Invariants
----------

The system enforces the following invariants. They are checked in application code, and also by
database triggers that run before every insert or update of a quote or billing project. A
logic error in the application therefore cannot leave the database in a state that breaks them:

- **Sum of BP limits ≤ quote authorized_amount.** The sum of all billing project limits
  under a quote can never exceed the quote's ``authorized_amount``. This is checked on every
  billing project create, limit edit, and move, and on every change to a quote's ``authorized_amount``.
- **Only the INTERNAL quote can be unlimited.** Every other quote must have an ``authorized_amount``,
  so a missing or failed write cannot leave a quote with unlimited spending.
- **Unlimited billing projects can only exist under INTERNAL.** A billing project with no spending
  limit can only exist under the ``INTERNAL`` quote. Every billing project under any other quote
  must have a limit.
- **Open billing projects cannot exist under a closed quote.** A quote cannot be closed while it
  has open billing projects; they must be closed or moved first. Likewise, a billing project cannot
  be created under, reopened under, or moved into a closed quote.

Total spend within a quote is bounded by the quote's ``authorized_amount`` on a best-effort basis:
each billing project's spend is limited by its own limit, and those limits sum to at most the quote's
``authorized_amount``. Billing project limits are enforced asynchronously, however. New batches are
rejected once a billing project's accrued cost reaches its limit, and running batches are cancelled
shortly afterwards, but jobs that are already running may push spend somewhat past the limit.

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
   * - **Submit jobs** to a billing project
     - ❌
     - ❌
     - ❌
     - ✅
   * - **Read job history** and see job details and logs in a billing project
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
     - ✅ §
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
   * - Change a quote's ``authorized_amount``
     - ✅
     - ✅
     - ❌
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
   * - Remove other users from a billing project
     - ✅
     - ❌
     - ❌
     - ❌
   * - Leave a billing project you are a member of
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
   * - Remove quote owners / managers
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

    **‡ Coming soon.**
    Adding users to billing projects, adding quote managers, and requesting billing project limit
    increases will be handled through requests and invitations in a future release. For now, these
    actions are performed by global billing managers on behalf of users.

.. note::

    **§ Billing history for billing project members.**
    Billing project members see only their own spend in a billing project's billing history. The
    project's total accrued cost is shown on the billing project page.

Lifecycle
---------

Billing projects and quotes each have a state that controls what operations are permitted.

**Billing project states:**

- **open** — the normal operating state. Users can submit jobs.
- **closed** — no new batches can be created. A billing project cannot be closed while it has
  running batches. A closed billing project can be reopened, unless its quote is closed. A closed
  billing project's limit still counts against its quote's headroom, and cannot be edited while the
  project is closed; to release that allocation, reopen the project, lower its limit, and close it again.

**Quote states:**

- **open** — the normal operating state. Billing projects can be created under the quote.
- **closed** — no new billing projects can be created under the quote. Existing billing projects
  must be closed (and therefore can no longer be spent against) or moved.
