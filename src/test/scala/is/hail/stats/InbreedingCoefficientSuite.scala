package is.hail.stats

import is.hail.SparkSuite
import is.hail.check.Prop._
import is.hail.check.Properties
import is.hail.io.vcf.ExportVCF
import is.hail.methods.VariantQC
import is.hail.utils._
import is.hail.testUtils._
import is.hail.variant._
import org.testng.annotations.Test

import scala.language._
import scala.sys.process._

class InbreedingCoefficientSuite extends SparkSuite {

  @Test def signatureIsCorrect() {
    assert(InbreedingCombiner.signature.typeCheck((new InbreedingCombiner()).asAnnotation))
  }
}
