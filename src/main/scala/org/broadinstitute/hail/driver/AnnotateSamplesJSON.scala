package org.broadinstitute.hail.driver

import org.broadinstitute.hail.Utils._
import org.broadinstitute.hail.annotations.Annotation
import org.broadinstitute.hail.expr._
import org.json4s.jackson.JsonMethods._
import org.kohsuke.args4j.{Argument, Option => Args4jOption}

import scala.collection.JavaConverters._

object AnnotateSamplesJSON extends Command {

  class Options extends BaseOptions {
    @Args4jOption(required = false, name = "-t", aliases = Array("--types"),
      usage = "Type of JSON")
    var `type`: String = ""

    @Args4jOption(required = true, name = "-r", aliases = Array("--root"),
      usage = "Period-delimited path starting with `sa'")
    var root: String = _

    @Args4jOption(required = true, name = "-v", aliases = Array("--sample"),
      usage = "Expression for sample in terms of `root'")
    var sample: String = _

    @Argument(usage = "<files...>")
    var arguments: java.util.ArrayList[String] = new java.util.ArrayList[String]()

  }

  def newOptions = new Options

  def name = "annotatesamples json"

  def description = "Annotate samples with JSON file"

  def supportsMultiallelic = true

  def requiresVDS = true

  def run(state: State, options: Options): State = {

    val files = hadoopGlobAll(options.arguments.asScala, state.hadoopConf)

    val sc = state.sc
    val vds = state.vds

    val t = Parser.parseType(options.`type`)

    val extractSample = JSONAnnotationImpex.jsonExtractSample(t, options.sample)

    val sampleAnnot =
      sc.union(files.map { f =>
        sc.textFile(f)
          .map { line =>
            JSONAnnotationImpex.importAnnotation(parse(line), t)
          }
      })
        .flatMap { sa =>
          extractSample(sa)
            .map { s => (s, sa) }
        }
        .collectAsMap()
        .toMap

    state.copy(vds = vds
      .annotateSamples(sampleAnnot, t, Parser.parseAnnotationRoot(options.root, Annotation.SAMPLE_HEAD)))
  }
}
