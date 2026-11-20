## Change Description

Fixes #<issue_number> (delete if N/A)

Brief description and justification of what this PR is doing.

<details>
  <summary>Additional details</summary>
  A place to put longform (or generated) change descriptions. Delete if unnecessary
</details>

## Merge Impact

What's the immediate impact of merging this PR? Note and distinguish between immediate, unavoidable impacts and 
impacts which are delayed, gated or only activate in specific situations.

## Reversibility

How easy is it to roll back this PR if something goes wrong? Consider reasons why a simple revert PR might not work
(including potentially needing to update the deployed image in k8s manually if CI is unable to deploy the revert).

## Validation

Standard: all PRs must pass the CI suite run against their changes. 
Additionally: Mention any testing or other validation performed beyond what CI would run.

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
