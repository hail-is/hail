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
  def flattenOptions(missingValue: String): String = t match {
    case x: Option[Any] => Formatter.writeOption(x, missingValue)
    case x: Iterable[Any] => x.map(_.toString).reduceRight(_ + "," + _)
    case d: Double => stringFormatDouble(d)
    case _ => t.toString
  }
}

object ExportUtils {
  type ExportGenotypeWithSA = (IndexedSeq[AnnotationData] => ((Variant, AnnotationData) => ((Int, Sample, Genotype) => String)))
  type ExportGenotypePostSA = (Variant, AnnotationData) => ((Int, Sample, Genotype) => String)
}

object UserExportUtils {
  implicit def toFormatter[T](t: T): Formatter[T] = new Formatter(t)
}


class ExportVariantsEvaluator(list: String, vas: AnnotationSignatures, missingValue: String)
  extends Evaluator[(Variant, AnnotationData) => String]({
    "(v: org.broadinstitute.hail.variant.Variant, \n" +
      "__va: org.broadinstitute.hail.annotations.AnnotationData) => { \n" +
      "import org.broadinstitute.hail.methods.FilterUtils._\n" +
      "import org.broadinstitute.hail.methods.UserExportUtils._\n" +
      signatures(vas, "__va") +
      instantiate("va", "__va") +
      s"""Array($list).map(_.flattenOptions("$missingValue")).reduceRight(_ + "\t" + _)}: String"""}) {
  def apply(v: Variant, va: AnnotationData): String = eval()(v, va)
}

class ExportSamplesEvaluator(list: String, sas: AnnotationSignatures, missingValue: String)
  extends Evaluator[(Sample, AnnotationData) => String]({
    "(s: org.broadinstitute.hail.variant.Sample, \n" +
      "__sa: org.broadinstitute.hail.annotations.AnnotationData) => { \n" +
      "import org.broadinstitute.hail.methods.FilterUtils._\n" +
      "import org.broadinstitute.hail.methods.UserExportUtils._\n" +
      signatures(sas, "__sa") +
      instantiate("sa", "__sa") +
      s"""Array($list).map(_.flattenOptions("$missingValue")).reduceRight(_ + "\t" + _)}: String"""}) {
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
      "(v: org.broadinstitute.hail.variant.Variant, " +
      "__va: org.broadinstitute.hail.annotations.AnnotationData) => {\n" +
      signatures(vas, "__va") +
      instantiate("va", "__va") +
      "(__sIndex: Int, " +
      "s: org.broadinstitute.hail.variant.Sample, " +
      "g: org.broadinstitute.hail.variant.Genotype) => {\n" +
      "val sa = __saArray(__sIndex)\n" +
      s"""Array($list).map(_.flattenOptions("$missingValue")).reduceRight(_ + "\t" + _)}: String}}"""}, t => t(sad)) {
  def apply(sa: IndexedSeq[AnnotationData])
    (v: Variant, va: AnnotationData)(sIndex: Int, s: Sample, g: Genotype): String =
    eval()(v, va)(sIndex, s, g)
}