package org.broadinstitute.hail.driver

import org.broadinstitute.hail.Utils._
import org.broadinstitute.hail.methods._
import org.broadinstitute.hail.variant._
import org.broadinstitute.hail.annotations._
import org.kohsuke.args4j.{Option => Args4jOption}

object ExportVariants extends Command {

  class Options extends BaseOptions {

    @Args4jOption(required = true, name = "-o", aliases = Array("--output"),
      usage = "path of output tsv")
    var output: String = _

    @Args4jOption(required = true, name = "-c", aliases = Array("--condition"),
      usage = "Comma-separated list of fields to be printed to tsv")
    var condition: String = _
  }

  def newOptions = new Options

  def name = "exportvariants"

  def description = "Export list of variant information to tsv"

  def run(state: State, options: Options): State = {
    val vds = state.vds

    val cond = options.condition

    val output = options.output

    val vas = vds.metadata.variantAnnotationSignatures
    val makeString: (Variant, Annotations[String]) => String = {
      try {
          val eve = new ExportVariantsEvaluator(cond, vas)
          eve.typeCheck()
          eve.apply
        } catch {
          case e: scala.tools.reflect.ToolBoxError =>
            /* e.message looks like:
               reflective compilation has failed:

               ';' expected but '.' found. */
            fatal("parse error in condition: " + e.message.split("\n").last)
        }
    }

    // FIXME add additional command parsing functionality
    val variantRegex = """v\.(\w+)""".r
    val topLevelAnnoRegex = """va\.(\w+)""".r
    val printMapRegex = """va\.(\w+)\.all""".r
    val annoRegex = """va\.(.+)""".r
    def mapColumnNames(input: String): String = {
      input match {
        case "v" => "Variant"
        case "va" =>
          fatal("parse error in condition: cannot print 'va', choose a group or value in annotations")
        case variantRegex(x) => x
        case topLevelAnnoRegex(x) =>
          if (vas.maps.contains(x)) {
            val keys = vas.maps(x).keys.toArray.sorted
            if (keys.isEmpty) x else s"$x:" + keys.mkString("\t")
          }
          else x
        case printMapRegex(x) =>
          val keys = vas.maps(x).keys
          if (keys.isEmpty) x else keys.mkString("\t")
        case annoRegex(x) => x
        case _ => input
      }
    }

    writeTextFile(output + ".header", state.hadoopConf) { s =>
      s.write(cond.split(",").map(mapColumnNames).mkString("\t"))
      s.write("\n")
    }

    hadoopDelete(output, state.hadoopConf, recursive = true)

    vds.variantsAndAnnotations
      .map { case (v, va) => makeString(v, va) }
      .saveAsTextFile(output)

    state
  }
}
