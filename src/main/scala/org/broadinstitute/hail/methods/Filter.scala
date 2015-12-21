package org.broadinstitute.hail.methods

import org.broadinstitute.hail.annotations._
import org.broadinstitute.hail.annotations.AnnotationClassBuilder._
import org.broadinstitute.hail.methods.FilterUtils.{FilterGenotypePostSA, FilterGenotypeWithSA}
import org.broadinstitute.hail.variant._
import scala.language.implicitConversions

class FilterString(val s: String) extends AnyVal {
  def ~(t: String): Boolean = s.r.findFirstIn(t).isDefined

  def !~(t: String): Boolean = !this.~(t)
}

class AnnotationValueString(val s: String) extends AnyVal {
  def toArrayInt: Array[Int] = s.split(",").map(i => i.toInt)

  def toArrayDouble: Array[Double] = s.split(",").map(i => i.toDouble)

  def toSetString: Set[String] = s.split(",").toSet
}

object FilterUtils {
  type FilterGenotypeWithSA = ((IndexedSeq[AnnotationData], IndexedSeq[String]) =>
    ((Variant, AnnotationData) => ((Int, Genotype) => Boolean)))
  type FilterGenotypePostSA = (Variant, AnnotationData) => ((Int, Genotype) => Boolean)

  implicit def toFilterString(s: String): FilterString = new FilterString(s)

  implicit def toAnnotationValueString(s: String): AnnotationValueString = new AnnotationValueString(s)
}

class FilterVariantCondition(cond: String, vas: AnnotationSignatures)
  extends Evaluator[(Variant, AnnotationData) => Boolean]({
    s"""(v: org.broadinstitute.hail.variant.Variant,
        |  __va: org.broadinstitute.hail.annotations.AnnotationData) => {
        |  import org.broadinstitute.hail.methods.FilterUtils._
        |  ${signatures(vas, "__va")}
        |  ${instantiate("va", "__va")}
        |  $cond
        |}: Boolean
    """.stripMargin
  }) {
  def apply(v: Variant, va: AnnotationData): Boolean = eval()(v, va)
}

class FilterSampleCondition(cond: String, sas: AnnotationSignatures)
  extends Evaluator[(Sample, AnnotationData) => Boolean](
    s"""(s: org.broadinstitute.hail.variant.Sample,
        |  __sa: org.broadinstitute.hail.annotations.AnnotationData) => {
        |  import org.broadinstitute.hail.methods.FilterUtils._
        |  ${signatures(sas, "__sa")}
        |  ${instantiate("sa", "__sa")}
        |  $cond
        |}: Boolean
    """.stripMargin) {
  def apply(s: Sample, sa: AnnotationData): Boolean = eval()(s, sa)
}

class FilterGenotypeCondition(cond: String, metadata: VariantMetadata)
  extends EvaluatorWithTransformation[FilterGenotypeWithSA, FilterGenotypePostSA](
    s"""(__sa: IndexedSeq[org.broadinstitute.hail.annotations.AnnotationData],
        |  __ids: IndexedSeq[String]) => {
        |  import org.broadinstitute.hail.methods.FilterUtils._
        |  ${signatures(metadata.sampleAnnotationSignatures, "__sa")}
        |  ${makeIndexedSeq("__saArray", "__sa", "__sa")}
        |  (v: org.broadinstitute.hail.variant.Variant,
        |    __va: org.broadinstitute.hail.annotations.AnnotationData) => {
        |    ${signatures(metadata.variantAnnotationSignatures, "__va")}
        |    ${instantiate("va", "__va")}
        |    (__sIndex: Int,
        |      g: org.broadinstitute.hail.variant.Genotype) => {
        |        val sa = __saArray(__sIndex)
        |        val s = org.broadinstitute.hail.variant.Sample(__ids(__sIndex))
        |        $cond
        |      }: Boolean
        |   }
        | }
      """.stripMargin,
    t => t(metadata.sampleAnnotations, metadata.sampleIds)) {
  def apply(v: Variant, va: AnnotationData)(sIndex: Int, g: Genotype): Boolean =
    eval()(v, va)(sIndex, g)
}
