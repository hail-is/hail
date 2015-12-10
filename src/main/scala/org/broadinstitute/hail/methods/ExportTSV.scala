package org.broadinstitute.hail.methods

import org.broadinstitute.hail.annotations.AnnotationClassBuilder._
import org.broadinstitute.hail.annotations._
import org.broadinstitute.hail.variant.{Sample, Variant, Genotype}
import org.broadinstitute.hail.Utils.stringFormatDouble
import scala.language.implicitConversions

object Formatter {
  def writeOption(o: Option[Any], missingValue: String): String = o match {
    case Some(d: Double) => stringFormatDouble(d)
    case Some(x) => x.toString
    case None => missingValue
  }
}

class Formatter[T](val t: T) extends AnyVal {
  def formatString(missingValue: String): String = t match {
    case x: Option[Any] => Formatter.writeOption(x, missingValue)
    case d: Double => stringFormatDouble(d)
    case x: Iterable[Double] => if (x.isEmpty) "" else x.map(stringFormatDouble).reduceRight(_ + "," + _)
    case x: Iterable[Any] => if (x.isEmpty) "" else x.map(_.toString).reduceRight(_ + "," + _)
    case _ => t.toString
  }
}

object ExportUtils {
  type ExportGenotypeWithSA = (IndexedSeq[AnnotationData] => ((Variant, AnnotationData) => ((Int, Sample, Genotype) => String)))
  type ExportGenotypePostSA = (Variant, AnnotationData) => ((Int, Sample, Genotype) => String)
}

object UserExportUtils {
  implicit def toFormatter[T](t: T): Formatter[T] = new Formatter(t)

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
    def isCopmlex = v.isComplex
    def isTransition = v.isTransition
    def isTransversion = v.isTransversion
    def nMismatch = v.nMismatch
    override def toString: String = {
      s"${contig}_${start}_${ref}_$alt"
    }
  }

}

class ExportVariantsEvaluator(list: String, vas: AnnotationSignatures, missingValue: String)
  extends Evaluator[(Variant, AnnotationData) => String]({
    val a = "(__v: org.broadinstitute.hail.variant.Variant, \n" +
      "__va: org.broadinstitute.hail.annotations.AnnotationData) => { \n" +
      "import org.broadinstitute.hail.methods.FilterUtils._\n" +
      "import org.broadinstitute.hail.methods.UserExportUtils._\n" +
      "val v: org.broadinstitute.hail.methods.UserExportUtils.ExportVariant = new ExportVariant(__v)\n" +
      signatures(vas, "__va", makeToString = true, missing = missingValue) +
      instantiate("va", "__va") +
      s"""Array($list).map(_.formatString("$missingValue")).reduceRight(_ + "\t" + _)}: String""";println(a); a}) {
  def apply(v: Variant, va: AnnotationData): String = eval()(v, va)
}

class ExportSamplesEvaluator(list: String, sas: AnnotationSignatures, missingValue: String)
  extends Evaluator[(Sample, AnnotationData) => String]({
    val a = "(s: org.broadinstitute.hail.variant.Sample, \n" +
      "__sa: org.broadinstitute.hail.annotations.AnnotationData) => { \n" +
      "import org.broadinstitute.hail.methods.FilterUtils._\n" +
      "import org.broadinstitute.hail.methods.UserExportUtils._\n" +
      signatures(sas, "__sa", makeToString = true, missing = missingValue) +
      instantiate("sa", "__sa") +
      s"""Array($list).map(_.formatString("$missingValue")).reduceRight(_ + "\t" + _)}: String"""; println(a); a}) {
  def apply(s: Sample, sa: AnnotationData): String = eval()(s, sa)
}

class ExportGenotypeEvaluator(list: String, vas: AnnotationSignatures, sas: AnnotationSignatures,
  sad: IndexedSeq[AnnotationData], missingValue: String)
  extends EvaluatorWithTransformation[ExportUtils.ExportGenotypeWithSA, ExportUtils.ExportGenotypePostSA](
    {"(__sa: IndexedSeq[org.broadinstitute.hail.annotations.AnnotationData]) => {\n" +
      "import org.broadinstitute.hail.methods.FilterUtils._\n" +
      "import org.broadinstitute.hail.methods.UserExportUtils._\n" +
      signatures(sas, "__sa") +
      makeIndexedSeq("__saArray", "__sa", "__sa") +
      "(__v: org.broadinstitute.hail.variant.Variant, " +
      "__va: org.broadinstitute.hail.annotations.AnnotationData) => {\n" +
      "val v: org.broadinstitute.hail.methods.UserExportUtils.ExportVariant = new ExportVariant(__v)\n" +
      signatures(vas, "__va", makeToString = true, missing = missingValue) +
      instantiate("va", "__va") +
      "(__sIndex: Int, " +
      "s: org.broadinstitute.hail.variant.Sample, " +
      "g: org.broadinstitute.hail.variant.Genotype) => {\n" +
      "val sa = __saArray(__sIndex)\n" +
      s"""Array($list).map(_.formatString("$missingValue")).reduceRight(_ + "\t" + _)}: String}}"""}, t => t(sad)) {
  def apply(sa: IndexedSeq[AnnotationData])
    (v: Variant, va: AnnotationData)(sIndex: Int, s: Sample, g: Genotype): String =
    eval()(v, va)(sIndex, s, g)
}