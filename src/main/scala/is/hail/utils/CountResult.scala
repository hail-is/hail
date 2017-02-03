package is.hail.utils

import java.util
import scala.collection.JavaConverters._

case class CountResult(nSamples: Int,
  nVariants: Long,
  nCalled: Option[Long]) {
  def nGenotypes: Long = nSamples * nVariants

  def callRate: Option[Double] =
    nCalled.flatMap(nCalled => divOption[Double](nCalled.toDouble * 100.0, nGenotypes))

  def toJavaMap: util.Map[String, Any] = {
    var m: Map[String, Any] = Map("nSamples" -> nSamples,
      "nVariants" -> nVariants,
      "nGenotypes" -> nGenotypes)
    nCalled.foreach { nCalled => m += "nCalled" -> nCalled }
    callRate.foreach { callRate => m += "callRate" -> callRate }
    m.asJava
  }
}