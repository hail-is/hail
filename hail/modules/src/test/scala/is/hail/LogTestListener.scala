package is.hail

import java.io.{PrintWriter, StringWriter}

import org.testng.{ITestContext, ITestListener, ITestResult}

class LogTestListener extends ITestListener {
  def testString(result: ITestResult): String =
    s"${result.getTestClass.getName}.${result.getMethod.getMethodName}"

  override def onTestStart(result: ITestResult): Unit =
    System.err.println(s"starting test ${testString(result)}...")

  override def onTestSuccess(result: ITestResult): Unit =
    System.err.println(s"test ${testString(result)} SUCCESS")

  override def onTestFailure(result: ITestResult): Unit = {
    val cause = result.getThrowable
    if (cause != null) {
      val sw = new StringWriter()
      val pw = new PrintWriter(sw)
      cause.printStackTrace(pw)
      System.err.println(s"Exception:\n$sw")
    }
    System.err.println(s"test ${testString(result)} FAILURE\n")
  }

  override def onTestSkipped(result: ITestResult): Unit =
    System.err.println(s"test ${testString(result)} SKIPPED")

  override def onTestFailedButWithinSuccessPercentage(result: ITestResult): Unit = {}

  override def onStart(context: ITestContext): Unit = {}

  override def onFinish(context: ITestContext): Unit = {}
}
