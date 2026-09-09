<important if="you are running tests">
- run all tests: `./mill 'hail.test'`
- run all tests in one test suite: `./mill 'hail.test.testOnly' is.hail.expr.ir.IRSuite`
- run all tests in one test suite matching a glob: `./mill 'hail.test.testOnly' is.hail.expr.ir.IRSuite -- -methods '*.testStr*'`
</important>
