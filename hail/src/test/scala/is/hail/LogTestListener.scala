package is.hail

import java.io.{PrintWriter, StringWriter}
import is.hail.utils._
import org.apache.log4j.{ConsoleAppender, PatternLayout}
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
    val cause = result.getThrowable
    if (cause != null) {
      val sw = new StringWriter()
      val pw = new PrintWriter(sw)
      cause.printStackTrace(pw)
      info(s"Exception:\n$sw")
    }
    info(s"test ${ testString(result) } FAILURE\n")
  }

  def onTestSkipped(result: ITestResult) {
    info(s"test ${ testString(result) } SKIPPED")
  }

  def onTestFailedButWithinSuccessPercentage(result: ITestResult) {

  }

  def onStart(context: ITestContext) {
    consoleLog.addAppender(new ConsoleAppender(new PatternLayout(HailContext.logFormat), "System.err"))
  }

  def onFinish(context: ITestContext) {

  }
}
