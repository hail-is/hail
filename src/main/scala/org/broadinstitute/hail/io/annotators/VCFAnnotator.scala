package org.broadinstitute.hail.io.annotators

import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD
import org.broadinstitute.hail.annotations._
import org.broadinstitute.hail.expr
import org.broadinstitute.hail.methods.LoadVCF
import org.broadinstitute.hail.variant.{Genotype, Variant}
import org.broadinstitute.hail.driver.SplitMulti.split

object VCFAnnotator {
  def apply(sc: SparkContext, filename: String): (RDD[(Variant, Annotation)], expr.Type) = {

    val vds2 = LoadVCF(sc, filename)

    val (newSigs1, insertIndex) = vds2.metadata.vaSignature.insert(expr.TInt, List("aIndex"))
    val (newSigs2, insertSplit) = newSigs1.insert(expr.TBoolean, List("wasSplit"))
    val newVDS = vds2.copy[Genotype](
      wasSplit = true,
      vaSignature = newSigs2,
      rdd = vds2.rdd.flatMap[(Variant, Annotation, Iterable[Genotype])] { case (v, va, it) =>
        split(v, va, it, false, { (va, index, wasSplit) =>
          insertSplit(insertIndex(va, Some(index)), Some(wasSplit))
        })
      })

    (newVDS.variantsAndAnnotations, newVDS.vaSignature)
  }
}
