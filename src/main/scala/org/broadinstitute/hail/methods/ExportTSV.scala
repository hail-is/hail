package org.broadinstitute.hail.methods

import org.broadinstitute.hail.annotations.AnnotationClassBuilder._
import org.broadinstitute.hail.annotations._
import org.broadinstitute.hail.variant.{Sample, Variant}
import scala.language.implicitConversions

object Formatter {
  def writeOption(o: Option[Any], missingValue: String): String = o match {
    case Some(x) => x.toString
    case None => missingValue
  }
}

class Formatter[T](val t: T) extends AnyVal {
  def flattenOptions(missingValue: String): String = t match {
    case x: Option[Any] => Formatter.writeOption(x, missingValue)
    case _ => t.toString
  }
}

object ExportUtils {
  implicit def toFormatter[T](t: T): Formatter[T] = new Formatter(t)
}


class ExportVariantsEvaluator(list: String, vas: AnnotationSignatures, missingValue: String)
  extends Evaluator[(Variant, AnnotationData) => String]({
    "(v: org.broadinstitute.hail.variant.Variant, \n" +
      "__va: org.broadinstitute.hail.annotations.AnnotationData) => { \n" +
      "import org.broadinstitute.hail.methods.FilterUtils._; \n" +
      "import org.broadinstitute.hail.methods.ExportUtils._; \n" +
      signatures(vas, "__va") +
      instantiate("va", "__va") +
      s"""Array($list).map(_.flattenOptions("$missingValue")).reduceRight(_ + "\t" + _)}: String"""}) {
  def apply(v: Variant, va: AnnotationData): String = eval()(v, va)
}

class ExportSamplesEvaluator(list: String, sas: AnnotationSignatures, missingValue: String)
  extends Evaluator[(Sample, AnnotationData) => String]({
    "(s: org.broadinstitute.hail.variant.Sample, \n" +
      "__sa: org.broadinstitute.hail.annotations.AnnotationData) => { \n" +
      "import org.broadinstitute.hail.methods.FilterUtils._; \n" +
      "import org.broadinstitute.hail.methods.ExportUtils._; \n" +
      signatures(sas, "__sa") +
      instantiate("sa", "__sa") +
      s"""Array($list).map(_.flattenOptions("$missingValue")).reduceRight(_ + "\t" + _)}: String"""}) {
  def apply(s: Sample, sa: AnnotationData): String = eval()(s, sa)
}