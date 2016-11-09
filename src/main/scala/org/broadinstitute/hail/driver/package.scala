package org.broadinstitute.hail

import java.util
import org.broadinstitute.hail.utils._
import org.broadinstitute.hail.variant.VariantDataset
import scala.collection.JavaConverters._

package object driver {

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

  def count(vds: VariantDataset, countGenotypes: Boolean): CountResult = {
    val (nVariants, nCalled) =
      if (countGenotypes) {
        val (nVar, nCalled) = vds.rdd.map { case (v, (va, gs)) =>
          (1L, gs.count(_.isCalled).toLong)
        }.fold((0L, 0L)) { (comb, x) =>
          (comb._1 + x._1, comb._2 + x._2)
        }
        (nVar, Some(nCalled))
      } else
        (vds.nVariants, None)

    CountResult(vds.nSamples, nVariants, nCalled)
  }
}
