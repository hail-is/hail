package org.broadinstitute.hail.methods

import org.broadinstitute.hail.annotations.AnnotationClassBuilder._
import org.broadinstitute.hail.annotations._
import org.broadinstitute.hail.variant.{VariantMetadata, Sample, Variant, Genotype}
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

  // FIXME move to Utils after we figure out what is calling the illegal cyclic operation bug
  def toTSVString(a: Any): String = {
    a match {
      case fo: FilterOption[_] => toTSVString(fo.o)
      case Some(x) => toTSVString(x)
      case None => "NA"
      case d: Double => d.formatted("%.4e")
      case i: Iterable[_] => i.map(toTSVString).mkString(",")
      case arr: Array[_] => arr.map(toTSVString).mkString(",")
      case _ => a.toString
    }
  }
}

class ExportVariantsEvaluator(list: String, vas: AnnotationSignatures)
  extends Evaluator[(Variant, AnnotationData) => String](
    s"""(__v: org.broadinstitute.hail.variant.Variant,
        |  __va: org.broadinstitute.hail.annotations.AnnotationData) => {
        |  import org.broadinstitute.hail.methods.FilterUtils._
        |  import org.broadinstitute.hail.methods.FilterOption
        |  import org.broadinstitute.hail.methods.UserExportUtils._
        |
        |  val v: ExportVariant = new ExportVariant(__v)
        |  ${signatures(vas, "__vaClass", makeToString = true)}
        |  ${instantiate("va", "__vaClass", "__va")}
        |  Array[Any]($list).map(toTSVString).mkRealString("\t")
        |}: String
    """.stripMargin,
    Filter.renameSymbols) {
  def apply(v: Variant, va: AnnotationData): String = eval()(v, va)
}

class ExportSamplesEvaluator(list: String, sas: AnnotationSignatures)
  extends Evaluator[(Sample, AnnotationData) => String](
    s"""(s: org.broadinstitute.hail.variant.Sample,
        |  __sa: org.broadinstitute.hail.annotations.AnnotationData) => {
        |  import org.broadinstitute.hail.methods.FilterUtils._
        |  import org.broadinstitute.hail.methods.FilterOption
        |  import org.broadinstitute.hail.methods.UserExportUtils._
        |
        |  ${signatures(sas, "__saClass", makeToString = true)}
        |  ${instantiate("sa", "__saClass", "__sa")}
        |  Array[Any]($list).map(toTSVString).mkRealString("\t")
        |}: String
    """.stripMargin,
    Filter.renameSymbols) {
  def apply(s: Sample, sa: AnnotationData): String = eval()(s, sa)
}

object ExportGenotypeEvaluator {
  type ExportGenotypeWithSA = ((IndexedSeq[AnnotationData], IndexedSeq[String]) =>
    ((Variant, AnnotationData) => ((Int, Genotype) => String)))
  type ExportGenotypePostSA = (Variant, AnnotationData) => ((Int, Genotype) => String)
}

class ExportGenotypeEvaluator(list: String, metadata: VariantMetadata)
  extends EvaluatorWithValueTransform[ExportGenotypeEvaluator.ExportGenotypeWithSA,
    ExportGenotypeEvaluator.ExportGenotypePostSA](
    s"""(__sa: IndexedSeq[org.broadinstitute.hail.annotations.AnnotationData],
        |  __ids: IndexedSeq[String]) => {
        |  import org.broadinstitute.hail.methods.FilterUtils._
        |  import org.broadinstitute.hail.methods.FilterOption
        |  import org.broadinstitute.hail.methods.FilterGenotype
        |  import org.broadinstitute.hail.methods.UserExportUtils._
        |
        |  ${signatures(metadata.sampleAnnotationSignatures, "__saClass", makeToString = true)}
        |  ${instantiateIndexedSeq("__saArray", "__saClass", "__sa")}
        |  (__v: org.broadinstitute.hail.variant.Variant,
        |    __va: org.broadinstitute.hail.annotations.AnnotationData) => {
        |    val v = new ExportVariant(__v)
        |    ${signatures(metadata.variantAnnotationSignatures, "__vaClass")}
        |    ${instantiate("va", "__vaClass", "__va")}
        |    (__sIndex: Int,
        |     __g: org.broadinstitute.hail.variant.Genotype) => {
        |        val sa = __saArray(__sIndex)
        |        val s = org.broadinstitute.hail.variant.Sample(__ids(__sIndex))
        |        val g = new FilterGenotype(__g)
        |        Array[Any]($list).map(toTSVString).mkRealString("\t")
        |      }: String
        |   }
        | }
      """.stripMargin,
    t => t(metadata.sampleAnnotations, metadata.sampleIds),
    Filter.renameSymbols) {

  def apply(v: Variant, va: AnnotationData)(sIndex: Int, g: Genotype): String =
    eval()(v, va)(sIndex, g)
}
