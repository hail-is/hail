package is.hail.stats

import is.hail.SparkSuite
import org.testng.annotations.Test

import scala.language._

class InbreedingCoefficientSuite extends SparkSuite {

  @Test def signatureIsCorrect() {
    assert(InbreedingCombiner.signature.typeCheck((new InbreedingCombiner()).asAnnotation))
  }
}
