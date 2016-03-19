package org.broadinstitute.hail.driver

import org.broadinstitute.hail.Utils._
import org.broadinstitute.hail.annotations.{Annotation, Inserter}
import org.broadinstitute.hail.io.annotators.SampleTSVAnnotator
import org.broadinstitute.hail.expr
import org.broadinstitute.hail.methods.ProgrammaticAnnotation
import org.broadinstitute.hail.variant.Sample
import org.kohsuke.args4j.{Option => Args4jOption}

import scala.collection.mutable

object AnnotateSamples extends Command {

  class Options extends BaseOptions {
    @Args4jOption(required = true, name = "-c", aliases = Array("--condition"),
      usage = "Annotation file path")
    var condition: String = _

    @Args4jOption(name = "-s", aliases = Array("--sampleheader"),
      usage = "Identify the name of the column containing the sample IDs (default: 'Sample')")
    var sampleCol: String = _

    @Args4jOption(required = false, name = "-t", aliases = Array("--types"),
      usage = "Define types of fields in annotations files")
    var types: String = _

    @Args4jOption(required = false, name = "-r", aliases = Array("--root"),
      usage = "Place annotations in the path 'sa.<root>.<field>, or sa.<field> if unspecified'")
    var root: String = _

    @Args4jOption(required = false, name = "-m", aliases = Array("--missing"),
      usage = "Specify identifiers to be treated as missing (default: 'NA')")
    var missingIdentifiers: String = _
  }

  def newOptions = new Options

  def name = "annotatesamples"

  def description = "Annotate samples in current dataset"

  override def supportsMultiallelic = true

  def parseRoot(s: String): List[String] = s match {
    case r if r.startsWith("sa.") => r.substring(3).split("""\.""").toList
    case "sa" => List[String]()
    case error => fatal(s"invalid root '$error': expect 'sa.<path[.path2...]>'")
  }

  def run(state: State, options: Options): State = {
    val vds = state.vds

    val cond = options.condition

    val stripped = hadoopStripCodec(cond, state.sc.hadoopConfiguration)
    val annotated = stripped match {
      case tsv if tsv.endsWith(".tsv") =>
        if (options.root == null)
          fatal("argument 'root' is required for '.tsv' annotation")
        val (m, signature) = SampleTSVAnnotator(cond,
          Option(options.sampleCol).getOrElse("Sample"),
          AnnotateVariants.parseTypeMap(Option(options.types).getOrElse("")),
          AnnotateVariants.parseMissing(Option(options.missingIdentifiers).getOrElse("NA")),
          vds.sparkContext.hadoopConfiguration)
        vds.annotateSamples(m, signature, parseRoot(options.root))
      case programmatic =>
        if (options.root != null)
          warn("argument 'root' is unnecessary for programmatic annotation, ignoring it")
        if (options.types != null)
          warn("argument 'types' is unnecessary for programmatic annotation, ignoring it")
        if (options.missingIdentifiers != null)
          warn("argument 'missing' is unnecessary for programmatic annotation, ignoring it")
        if (options.sampleCol != null)
          warn("argument 'sampleheader' is unnecessary for programmatic annotation, ignoring it")

        val symTab = Map(
          "s" ->(0, expr.TSample),
          "sa" ->(1, vds.saSignature))
        val a = new mutable.ArrayBuffer[Any](2)
        val parsed = expr.Parser.parseAnnotationArgs(symTab, a, cond)
        val keyedSignatures = parsed.map { case (ids, t, f) =>
          if (ids.head != "sa")
            fatal(s"expect 'sa[.identifier]+', got ${ids.mkString(".")}")
          ProgrammaticAnnotation.checkType(ids.mkString("."), t)
          (ids.tail, t)
        }
        val inserterBuilder = mutable.ArrayBuilder.make[Inserter]
        val vdsAddedSigs = keyedSignatures.foldLeft(vds) { case (v, (ids, signature)) =>
          val (s, i) = v.insertSA(signature, ids)
          inserterBuilder += i
          v.copy(saSignature = s)
        }

        val computations = parsed.map(_._3)
        val inserters = inserterBuilder.result()

        for (_ <- computations)
          a += null

        val newAnnotations = vdsAddedSigs.sampleAnnotations.zipWithIndex.map { case (sa, i) =>
          a(0) = Sample(vds.sampleIds(i))
          a(1) = sa

          val queries = computations.map(_())
          queries.indices.foreach { i =>
            a(1) = inserters(i).apply(
              a(1),
              Option(queries(i)))
          }
          a(1): Annotation
        }
        vdsAddedSigs.copy(sampleAnnotations = newAnnotations)
    }
    state.copy(vds = annotated)
  }
}
