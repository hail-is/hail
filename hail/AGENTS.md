<important if="you are running tests">
- run all tests: `sh mill 'hail.test'`
- run all tests in one test suite: `sh mill 'hail.test.testOnly' is.hail.expr.ir.IRSuite`
- run all tests in one test suite matching a glob: `sh mill 'hail.test.testOnly' is.hail.expr.ir.IRSuite -- -methods '*.testStr*'`
</important>
