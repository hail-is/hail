package org.broadinstitute.hail.io.annotators

import org.apache.hadoop
import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD
import org.broadinstitute.hail.annotations._
import org.broadinstitute.hail.expr
import org.broadinstitute.hail.methods.LoadVCF
import org.broadinstitute.hail.variant.{Genotype, Variant}
import org.broadinstitute.hail.driver.SplitMulti.split

object VCFAnnotator {
  def apply(sc: SparkContext, filename: String): (RDD[(Variant, Annotation)], Signature) = {

    val vds2 = LoadVCF(sc, filename)

    val (newSigs1, insertIndex) = vds2.metadata.vaSignatures.insert(List("aIndex"),
      SimpleSignature(expr.TInt))
    val (newSigs2, insertSplit) = newSigs1.insert(List("wasSplit"),
      SimpleSignature(expr.TBoolean))
    val newVDS = vds2.copy[Genotype](
      wasSplit = true,
      vaSignatures = newSigs2,
      rdd = vds2.rdd.flatMap[(Variant, Annotation, Iterable[Genotype])] { case (v, va, it) =>
        split(v, va, it, false, insertIndex, insertSplit)
      })

    (newVDS.variantsAndAnnotations, newVDS.vaSignatures)
  }
}
