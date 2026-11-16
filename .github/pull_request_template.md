## Change Description

Fixes #<issue_number> (delete if N/A)

Brief description and justification of what this PR is doing.

## Merge Impact

What's the impact of merging this PR on end users?

- Immediately active: code that becomes active immediately and changes how actively-used parts of the system work.
  - Example: a change to how the batch driver's scheduler works
- Immediately available: code becomes active and available immediately but is not currently used by end users
  - Example: a new endpoint in the API that nobody is using yet
- Gated: The code becomes active immediately but requires opt-in or only impacts a small subset of users
  - Example: a UI element behind a feature flag
- Release-gated: The code only becomes active when a new release happens
  - Example: changes to the query compiler
- Hidden: The code does not have any entrypoint to users. It becomes available to subsequent
  - Example: helper function for the next PR to use
- Never active: No impact on any end user
  - Example: a change adding new tests and/or docs

## Reversability

How easy is it to roll back this PR if something goes wrong? (choose all that apply)

- Trivial: No persistent state is changed, no user behavior needs to change
  - Internal refactor of some logic with no externally visible difference in results
- Easy but verify: Some new state might be present, but the old code should be capable of handling it
  - We started recording some new metadata in job fields. The old code _should_ happily ignore the metadata
- User experience impacted: Users may have started to rely on functionality that would disappear
  - Example: a new CLI or UI feature gets removed. Users lose access to the cool new feature.
- Leaves bad state: Some new state might be present, and the reverted code may struggle to handle it
  - Example: Adding a new machine type which some VMs may be using. The old code would not recognize it and be unable to start
- Nightmare: It's almost certainly easier to roll forwards with an emergency fix than to roll back.
  - Example: A database migration moved fields around in non-trivial ways

## Validation

Choose all that are appropriate and fill in details:

- N/A
- None: YOLO
- Local unit testing
- Local functional testing
- Functional testing in a dev environment
- Functional testing in production (!)

## Security Assessment

Delete all except the correct answer:
- This change potentially impacts the Hail Batch instance as deployed by Broad Institute in GCP
  - The Impact Rating, Impact Description, and Appsec Review sections are required
- This change cannot impact the Hail Batch instance as deployed by Broad Institute in GCP
  - The Impact Rating, Impact Description, and Appsec Review sections can be deleted

### Impact Rating

Delete all except the correct answer:
- This change has a high security impact
- This change has a medium security impact
- This change has a low security impact
- This change has no security impact

### Impact Description

Replace this content with a description of the impact of the change:
- For none/low impact: a quick one/two sentence justification of the rating.
  - Example: "Docs only", "Low-level refactoring of non-security code", etc.
- For medium/high impact: provide a description of the impact and the mitigations in place.
  - Example: "New UI text field added in analogy to existing elements, with input strings escaped and validated against code injection"

### Appsec Review

- [ ] Required: The impact has been assessed and approved by appsec
