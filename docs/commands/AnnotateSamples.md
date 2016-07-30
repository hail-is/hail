# `annotatesamples`

This module is a supercommand that contains four submodules:

Name | Docs | Description
:-:  | :-: | ---
`annotatesamples expr` | [**\[link\]**](AnnotateSamplesExpr.md) | Generate sample annotations using the Hail expr language, including the ability to aggregate all the genotypes per sample.
`annotatesamples list` | [**\[link\]**](AnnotateSamplesList.md) | Annotate samples from a list of IDs with a `Boolean` for membership in that list.
`annotatesamples table` | [**\[link\]**](AnnotateSamplesTable.md) | Read a file keyed by sample ID to sample annotations as a `Struct` using Hail's table parsing module.
`annotatesamples fam` | [**\[link\]**](AnnotateSamplesFam.md) | Import a plink .fam file to sample annotations.