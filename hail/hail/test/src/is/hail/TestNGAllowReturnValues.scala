package is.hail

import java.util

import org.testng.IAlterSuiteListener
import org.testng.xml.XmlSuite

class TestNGAllowReturnValues extends IAlterSuiteListener {
  override def alter(suites: util.List[XmlSuite]): Unit =
    suites.forEach(_.setAllowReturnValues(true))
}
