This document describes hail's versioning policy.

## Hail Version Compatibility

Hail follows semantic versioning conventions. We do not intentionally break back-compatibility of interfaces or file formats. This means that a script developed to run on Hail 0.2.5 should continue to work in every subsequent release within the 0.2 major version. The exception to this rule is experimental functionality, denoted as such in the reference documentation, which may change at any time.

Please note that forward compatibility should not be expected, especially relating to file formats: this means that it may not be possible to use an earlier version of Hail to read files written in a later version.

Please also note that while there will not be breaking changes to the hail API within a single minor version, we may upgrade dependencies in a backwards incompatible way. For example, Hail might go from depending on numpy version 1.2 to 1.3 with an update to our patch version, and that may affect your pipelines if you use numpy directly. 

### Spark Versions

We make a stronger guarantee about Spark versioning. Hail will continue to support feature branch versions of Spark for 18 months after they are released. For example, Spark 3.1 was released in March of 2021, and so support should be expected until August 2022. For Spark's LTS releases (the last minor version before a major version is released, like Spark 2.4), we will continue support for 6 months after the last patch version is released. Whenever Spark 2.4.8 is released, we will continue to support it for 6 months afterwards. 
