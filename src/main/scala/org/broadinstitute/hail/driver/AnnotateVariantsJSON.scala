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

    val ec = EvalContext(Map(
      "va" ->(0, t)))

    val fs: Array[(BaseType, () => Option[Any])] = Parser.parseExprs(options.variantFields, ec)

    if (fs.length != 4)
      fatal(s"wrong number of variant field expressions: expected 4, got ${fs.length}")

    if (fs(0)._1 != TString)
      fatal(s"wrong type for chromosome field: expected String, got ${fs(0)._1}")
    if (fs(1)._1 != TInt)
      fatal(s"wrong type for pos field: expected Int, got ${fs(1)._1}")
    if (fs(2)._1 != TString)
      fatal(s"wrong type for ref field: expected String, got ${fs(2)._1}")
    if (fs(3)._1 != TArray(TString))
      fatal(s"wrong type for alt field: expected Array[String], got ${fs(3)._1}")

    val jsonRDD =
      sc.union(files.map { f =>
        sc.textFile(f)
          .map { line =>
            Annotation.fromJson(parse(line), t, "va")
          }
      })
        .flatMap { va =>
          ec.setAll(va)

          val vfs = fs.map(_._2())

          vfs(0).flatMap { chr =>
            vfs(1).flatMap { pos =>
              vfs(2).flatMap { ref =>
                vfs(3).map { alt =>
                  (Variant(chr.asInstanceOf[String],
                    pos.asInstanceOf[Int],
                    ref.asInstanceOf[String],
                    alt.asInstanceOf[IndexedSeq[String]].toArray),
                    va)
                }
              }
            }
          }
        }

    val annotated = vds
      .withGenotypeStream()
      .annotateVariants(jsonRDD, t,
        Parser.parseAnnotationRoot(options.root, Annotation.VARIANT_HEAD))

    state.copy(vds = annotated)
  }
}
