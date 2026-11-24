<important if="you are running tests">
- run all scala tests: `./mill 'hail.test'`
- run all scala tests in one test suite: `./mill 'hail.test.testOnly' is.hail.expr.ir.IRSuite`
- run all scala tests in one test suite matching a glob: `./mill 'hail.test.testOnly' is.hail.expr.ir.IRSuite -- -methods '*.testStr*'`
- run python (pytest) tests: `cd python && python -m pytest test/hail/<path>`
  - to use a specific backend, set the env var HAIL_QUERY_BACKEND to `local` or `spark` e.x.: `cd python && HAIL_QUERY_BACKEND=local python -m pytest test/hail/<path>`
</important>
