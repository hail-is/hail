### Change Description

<!-- If linking to an existing issue -->
Fixes #<issue_number>.

<!-- Otherwise -->
Brief description and justification of what this PR is doing.

### Security Assessment
<!-- If you already have an issue with an impact assessment, just reference it here and delete this whole section -->

<!-- Delete all except the correct answer -->
- This change has a high security impact
  - [ ] Required: The impact has been assessed and approved by appsec
- This change has a medium security impact
- This change has a low security impact
- This change has no security impact

<!-- Add a description of the security impact and necessary mitigations -->

- For none/low impact: a quick one/two sentence justification of the rating.
  - Example: "Docs only", "Low-level refactoring of non-security code", etc.
- For medium/high impact: provide a description of the impact and the mitigations in place.
  - Example: "New UI text field added in analogy to existing elements, with input strings escaped and validated against code injection"

<!-- Leave this line here as a reminder to reviewers -->
(Reviewers: please confirm the security impact before approving)
