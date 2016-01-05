package org.broadinstitute.hail.methods

import org.apache.spark.SparkContext
import org.broadinstitute.hail.Utils._
import org.broadinstitute.hail.annotations.AnnotationClassBuilder._
import org.broadinstitute.hail.annotations._
import org.broadinstitute.hail.variant.{VariantMetadata, Sample, Variant, Genotype}
import scala.io.Source
import scala.language.implicitConversions

object ExportTSV {
  def parseExpression(cond: String): (String, String) = {
    import scala.tools.reflect.ToolBox
    import scala.reflect.runtime.currentMirror
    import scala.reflect.runtime.universe._
    val toolbox = currentMirror.mkToolBox()
    val tree = toolbox.parse(s"dummy($cond)")
    val (headers, expressions) = tree match {
      case Apply(_, args: List[_]) =>
        args.map(t => t match {
          case (AssignOrNamedArg(Ident(name), expr)) => (name.toString, expr.toString)
          case _ => println(t); fatal("invalid export expression")
        })
        .unzip
    }
    println(s"headers = ${headers.mkString("\t")}")
    println(s"exprs = ${expressions.mkString(",")}")
    (headers.mkString("\t"), expressions.mkString(","))
  }
//
//
//  def parseExpression(cond: String, sc: SparkContext,
//    vas: Option[AnnotationSignatures] = None,
//    sas: Option[AnnotationSignatures] = None): (String, String) = {
//    if (cond.endsWith(".columns")) {
//      val lines = Source
//        .fromInputStream(hadoopOpen(cond, sc.hadoopConfiguration))
//        .getLines()
//        .map(_.split("\t"))
//        .toList
//      //        println(lines.map(_.mkString("; ")).mkString("\t"))
//      /* Check errors in user input format here.  Bad input that satisfies this check will throw
//        errors in makeString */
//      if (lines.isEmpty) {
//        fatal("parse error in .columns file: empty file")
//      }
//      if (!lines.forall(_.length == 2))
//        fatal("parse error in .columns file: expect 2 tab-separated fields per line")
//      (lines.map(_.apply(0)).mkString("\t"), lines.map(_.apply(1)).mkString(","))
//    } else
//      (cond.split(",")
//        .map(mapColumnNames(_, vas, sas))
//        .mkString("\t"), cond)
//  }
//
//  val topLevelVariantAnnoRegex = """va\.(\w+)""".r
//  val topLevelSampleAnnoRegex = """sa\.(\w+)""".r
//  val variantAllRegex = """va\.(.+)\.all""".r
//  val sampleAllRegex = """sa\.(.+)\.all""".r
//
//  def getSortedKeys[T](a: Option[Annotations[T]], map: String): Array[String] = {
//    if (a.isDefined) {
//      assert(a.get.hasMap(map))
//      a
//        .get
//        .maps(map)
//        .keys
//        .toArray
//        .sorted
//    }
//    else
//      Array.empty[String]
//  }
//  def mapColumnNames(input: String, vas: Option[AnnotationSignatures], sas: Option[AnnotationSignatures]): String = {
//    input match {
//      case "s" => "Sample"
//      case "v" => "Variant"
//      case "g" => "Genotype"
//      case "va" =>
//        fatal("parse error in condition: cannot print 'va', choose a group or value in annotations")
//      case "sa" =>
//        fatal("parse error in condition: cannot print 'sa', choose a group or value in annotations")
//      case variantAllRegex(x) => getSortedKeys(vas, x).map(field => s"va.$x.$field").mkString("\t")
//      case sampleAllRegex(x) => getSortedKeys(sas, x).map(field => s"sa.$x.$field").mkString("\t")
//      case _ => input
//    }
//  }
}

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
      case Some(o) => toTSVString(o)
      case None => "NA"
      case d: Double => d.formatted("%.4e")
      case i: Iterable[_] => i.map(toTSVString).mkString(",")
      case arr: Array[_] => arr.map(toTSVString).mkString(",")
      case _ => a.toString
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
        |  val v: ExportVariant = new ExportVariant(__v)
        |  ${signatures(vas, "__vaClass", makeToString = true)}
        |  ${instantiate("va", "__vaClass", "__va")}
        |  Array($list).map(toTSVString).mkString("\t")
        |}: String
    """.stripMargin
  }) {
  def apply(v: Variant, va: AnnotationData): String = eval()(v, va)
}

class ExportSamplesEvaluator(list: String, sas: AnnotationSignatures)
  extends Evaluator[(Sample, AnnotationData) => String](
    s"""(s: org.broadinstitute.hail.variant.Sample,
        |  __sa: org.broadinstitute.hail.annotations.AnnotationData) => {
        |  import org.broadinstitute.hail.methods.FilterUtils._
        |  import org.broadinstitute.hail.methods.UserExportUtils._
        |
        |  ${signatures(sas, "__saClass", makeToString = true)}
        |  ${instantiate("sa", "__saClass", "__sa")}
        |  Array($list).map(toTSVString).mkString("\t")
        |}: String
    """.stripMargin) {
  def apply(s: Sample, sa: AnnotationData): String = eval()(s, sa)
}

object ExportGenotypeEvaluator {
  type ExportGenotypeWithSA = ((IndexedSeq[AnnotationData], IndexedSeq[String]) =>
    ((Variant, AnnotationData) => ((Int, Genotype) => String)))
  type ExportGenotypePostSA = (Variant, AnnotationData) => ((Int, Genotype) => String)
}

class ExportGenotypeEvaluator(list: String, metadata: VariantMetadata)
  extends EvaluatorWithTransformation[ExportGenotypeEvaluator.ExportGenotypeWithSA,
    ExportGenotypeEvaluator.ExportGenotypePostSA](
    s"""(__sa: IndexedSeq[org.broadinstitute.hail.annotations.AnnotationData],
        |  __ids: IndexedSeq[String]) => {
        |  import org.broadinstitute.hail.methods.FilterUtils._
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
        |      g: org.broadinstitute.hail.variant.Genotype) => {
        |        val sa = __saArray(__sIndex)
        |        val s = org.broadinstitute.hail.variant.Sample(__ids(__sIndex))
        |        Array($list).map(toTSVString).mkString("\t")
        |      }: String
        |   }
        | }
      """.stripMargin,
    t => t(metadata.sampleAnnotations, metadata.sampleIds)) {

  def apply(v: Variant, va: AnnotationData)(sIndex: Int, g: Genotype): String =
    eval()(v, va)(sIndex, g)
}
