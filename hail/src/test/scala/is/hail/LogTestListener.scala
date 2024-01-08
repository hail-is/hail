package is.hail

import is.hail.utils._

import java.io.{PrintWriter, StringWriter}

import org.apache.log4j.{ConsoleAppender, PatternLayout}
import org.testng.{ITestContext, ITestListener, ITestResult}

class LogTestListener extends ITestListener {
  def testString(result: ITestResult): String =
    s"${result.getTestClass.getName}.${result.getMethod.getMethodName}"

  def onTestStart(result: ITestResult) {
    System.err.println(s"starting test ${testString(result)}...")
  }

  def onTestSuccess(result: ITestResult) {
    System.err.println(s"test ${testString(result)} SUCCESS")
  }

  def onTestFailure(result: ITestResult) {
    val cause = result.getThrowable
    if (cause != null) {
      val sw = new StringWriter()
      val pw = new PrintWriter(sw)
      cause.printStackTrace(pw)
      System.err.println(s"Exception:\n$sw")
    }
    System.err.println(s"test ${testString(result)} FAILURE\n")
  }

  def onTestSkipped(result: ITestResult) {
    System.err.println(s"test ${testString(result)} SKIPPED")
  }

  def onTestFailedButWithinSuccessPercentage(result: ITestResult) {}

  def onStart(context: ITestContext) {}

  def onFinish(context: ITestContext) {}
}
