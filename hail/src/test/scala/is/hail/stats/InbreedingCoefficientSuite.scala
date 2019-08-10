package is.hail.stats

import is.hail.HailSuite
import org.testng.annotations.Test

import scala.language._

class InbreedingCoefficientSuite extends HailSuite {

  @Test def signatureIsCorrect() {
    assert(InbreedingCombiner.signature.typeCheck((new InbreedingCombiner()).asAnnotation))
  }
}
