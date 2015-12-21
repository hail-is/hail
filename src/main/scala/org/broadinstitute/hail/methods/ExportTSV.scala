package org.broadinstitute.hail.methods

import org.broadinstitute.hail.annotations.AnnotationClassBuilder._
import org.broadinstitute.hail.annotations._
import org.broadinstitute.hail.variant.{Sample, Variant, Genotype}
import scala.language.implicitConversions

object UserExportUtils {
  class ExportVariant(val v: Variant) extends AnyVal {
    def contig = v.contig

    def start = v.start

    def ref = v.ref

    def alt = v.alt

    def variantType = v.variantType

    def inParX = v.inParX

    def inParY = v.inParY

    def isSNP = v.isSNP

    def isMNP = v.isMNP

    def isInsertion = v.isInsertion

    def isDeletion = v.isDeletion

    def isIndel = v.isIndel

    def isComplex = v.isComplex

    def isTransition = v.isTransition

    def isTransversion = v.isTransversion

    def nMismatch = v.nMismatch

    override def toString: String = {
      s"${contig}_${start}_${ref}_$alt"
    }
  }

}

class ExportVariantsEvaluator(list: String, vas: AnnotationSignatures)
  extends Evaluator[(Variant, AnnotationData) => String]({
    s"""(__v: org.broadinstitute.hail.variant.Variant,
        |  __va: org.broadinstitute.hail.annotations.AnnotationData) => {
        |  import org.broadinstitute.hail.methods.FilterUtils._
        |  import org.broadinstitute.hail.methods.UserExportUtils._
        |
        |
        |  val v: ExportVariant = new ExportVariant(__v)
        |  ${signatures(vas, "__va", makeToString = true)}
        |  ${instantiate("va", "__va")}
          |  Array($list).map(toTSVString).mkString("\t")
        |}: String
    """.stripMargin}) {
  def apply(v: Variant, va: AnnotationData): String = eval()(v, va)
}

class ExportSamplesEvaluator(list: String, sas: AnnotationSignatures)
  extends Evaluator[(Sample, AnnotationData) => String](
    {val s = s"""(s: org.broadinstitute.hail.variant.Sample,
        |  __sa: org.broadinstitute.hail.annotations.AnnotationData) => {
        |  import org.broadinstitute.hail.methods.FilterUtils._
        |  import org.broadinstitute.hail.methods.UserExportUtils._
        |  import org.broadinstitute.hail.Utils.toTSVString
        |  ${signatures(sas, "__sa", makeToString = true)}
        |  ${instantiate("sa", "__sa")}
        |  Array($list).map(toTSVString).mkString("\t")
        |}: String
    """.stripMargin;println(s);s}) {
  def apply(s: Sample, sa: AnnotationData): String = eval()(s, sa)
}

object ExportGenotypeEvaluator {
  type ExportGenotypeWithSA = ((IndexedSeq[AnnotationData], IndexedSeq[String]) =>
    ((Variant, AnnotationData) => ((Int, Genotype) => String)))
  type ExportGenotypePostSA = (Variant, AnnotationData) => ((Int, Genotype) => String)
}

class ExportGenotypeEvaluator(list: String, vas: AnnotationSignatures, sas: AnnotationSignatures,
  sad: IndexedSeq[AnnotationData], ids: IndexedSeq[String])
  extends EvaluatorWithTransformation[ExportGenotypeEvaluator.ExportGenotypeWithSA,
    ExportGenotypeEvaluator.ExportGenotypePostSA](
      s"""(__sa: IndexedSeq[org.broadinstitute.hail.annotations.AnnotationData],
          |  __ids: IndexedSeq[String]) => {
          |  import org.broadinstitute.hail.methods.FilterUtils._
          |  import org.broadinstitute.hail.methods.UserExportUtils._
          |  import org.broadinstitute.hail.Utils.toTSVString
          |  ${signatures(sas, "__sa", makeToString = true)}
          |  ${makeIndexedSeq("__saArray", "__sa", "__sa")}
          |  (__v: org.broadinstitute.hail.variant.Variant,
          |    __va: org.broadinstitute.hail.annotations.AnnotationData) => {
          |    val v = new ExportVariant(__v)
          |    ${signatures(vas, "__va")}
          |    ${instantiate("va", "__va")}
          |    (__sIndex: Int,
          |      g: org.broadinstitute.hail.variant.Genotype) => {
          |        val sa = __saArray(__sIndex)
          |        val s = org.broadinstitute.hail.variant.Sample(__ids(__sIndex))
          |        Array($list).map(toTSVString).mkString("\t")
          |      }: String
          |   }
          | }
      """.stripMargin,
    t => t(sad, ids)) {

  def apply(v: Variant, va: AnnotationData)(sIndex: Int, g: Genotype): String =
    eval()(v, va)(sIndex, g)
}