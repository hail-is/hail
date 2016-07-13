package org.broadinstitute.hail.driver

import org.broadinstitute.hail.Utils._
import org.broadinstitute.hail.annotations.Annotation
import org.broadinstitute.hail.expr._
import org.broadinstitute.hail.io.annotators._
import org.broadinstitute.hail.variant.{Genotype, Variant}
import org.json4s.jackson.JsonMethods._
import org.kohsuke.args4j.{Argument, Option => Args4jOption}

import scala.collection.JavaConverters._

object AnnotateVariantsJSON extends Command {

  class Options extends BaseOptions {
    @Args4jOption(required = false, name = "-t", aliases = Array("--types"),
      usage = "Type of JSON")
    var `type`: String = ""

    @Args4jOption(required = true, name = "-r", aliases = Array("--root"),
      usage = "Period-delimited path starting with `va'")
    var root: String = _

    @Args4jOption(required = true, name = "-v", aliases = Array("--vfields"),
      usage = "Expressions for chromosome, position, ref and alt in terms of `root'")
    var variantFields: String = _

    @Argument(usage = "<files...>")
    var arguments: java.util.ArrayList[String] = new java.util.ArrayList[String]()

  }

  def newOptions = new Options

  def name = "annotatevariants json"

  def description = "Annotate variants with JSON file"

  def supportsMultiallelic = true

  def requiresVDS = true

  def run(state: State, options: Options): State = {

    val files = hadoopGlobAll(options.arguments.asScala, state.hadoopConf)

    val sc = state.sc
    val vds = state.vds

    val t = Parser.parseType(options.`type`)

    val extractVariant = JSONAnnotationImpex.jsonExtractVariant(t, options.variantFields)

    val jsonRDD =
      sc.union(files.map { f =>
        sc.textFile(f)
          .map { line =>
            JSONAnnotationImpex.importAnnotation(parse(line), t, "<root>")
          }
      })
        .flatMap { va =>
          extractVariant(va)
            .map { v => (v, va) }
        }

    val annotated = vds
      .withGenotypeStream()
      .annotateVariants(jsonRDD, t,
        Parser.parseAnnotationRoot(options.root, Annotation.VARIANT_HEAD))

    state.copy(vds = annotated)
  }
}
