package is.hail

import is.hail.utils._
import org.testng.{ITestContext, ITestListener, ITestResult}

class LogTestListener extends ITestListener {
  def testString(result: ITestResult): String = {
    s"${ result.getTestClass.getName }.${ result.getMethod.getMethodName }"
  }

  def onTestStart(result: ITestResult) {
    info(s"starting test ${ testString(result) }...")
  }

  def onTestSuccess(result: ITestResult) {
    info(s"test ${ testString(result) } SUCCESS")
  }

  def onTestFailure(result: ITestResult) {
    info(s"test ${ testString(result) } FAILURE")
  }

  def onTestSkipped(result: ITestResult) {
    info(s"test ${ testString(result) } SKIPPED")
  }

  def onTestFailedButWithinSuccessPercentage(result: ITestResult) {

  }

  def onStart(context: ITestContext) {

  }

  def onFinish(context: ITestContext) {

  }
}
