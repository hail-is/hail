<important if="you are running tests">
- run all tests: `SCALA_VERSION=2.12 ./mill 'hail[].test'`
- run all tests in one test suite: `SCALA_VERSION=2.12 ./mill 'hail[].test.testOnly' is.hail.expr.ir.IRSuite`
- run all tests in one test suite matching a glob: `SCALA_VERSION=2.12 ./mill 'hail[].test.testOnly' is.hail.expr.ir.IRSuite -- -methods '*.testStr*'`
</important>
