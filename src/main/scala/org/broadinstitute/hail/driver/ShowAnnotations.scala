package org.broadinstitute.hail.driver

import org.broadinstitute.hail.Utils._
import org.broadinstitute.hail.annotations._
import org.kohsuke.args4j.{Option => Args4jOption}
import scala.collection.mutable

object ShowAnnotations extends Command {

  class Options extends BaseOptions {
    @Args4jOption(name = "-o", aliases = Array("--output"), usage = "Output file")
    var output: String = _
  }

  def newOptions = new Options

  def name = "showannotations"

  def description = "Shows the signatures for all annotations currently stored in the dataset"

  override def supportsMultiallelic = true

  def printSignatures(sb: StringBuilder, a: StructSignature, spaces: Int, path: String) {
    val spacing = (0 until spaces).map(i => " ").fold("")(_ + _)
    val values = new mutable.ArrayBuilder.ofRef[(String, Signature)]()
    val subAnnotations = new mutable.ArrayBuilder.ofRef[(String, StructSignature)]()

    a.m.toArray
      .sortBy { case (k, (i, v)) => i }
      .map {
        case (k, (i, v)) =>
          v match {
            case anno: StructSignature =>
              sb.append(s"""$spacing$k: $path.$k.<identifier> [$i]""")
              sb.append("\n")
              printSignatures(sb, anno, spaces + 2, path + "." + k)
            case sig =>
              sb.append(s"""$spacing$k: ${sig.dType} [$i]""")
              sb.append("\n")
          }
      }
  }

  def run(state: State, options: Options): State = {
    val vds = state.vds

    if (vds == null)
      fatal("showannotations requires a non-null variant dataset, import or read one first")

    val sampleSB = new StringBuilder()
    printSignatures(sampleSB, vds.metadata.sampleAnnotationSignatures, 4, "sa")

    val variantSB = new StringBuilder()
    printSignatures(variantSB, vds.metadata.variantAnnotationSignatures, 4, "va")

    val combinedSB = new StringBuilder()
    combinedSB.append("  Sample annotations: sa.<identifier>")
    combinedSB.append("\n")
    combinedSB.append(sampleSB.result())
    combinedSB.append("\n")
    combinedSB.append("  Variant annotations: va.<identifier>")
    combinedSB.append("\n")
    combinedSB.append(variantSB.result())
    val result = combinedSB.result()

    options.output match {
      case null => println(result)
      case out => writeTextFile(out, vds.sparkContext.hadoopConfiguration) { dos =>
        dos.write(result)
      }
    }

    state
  }
}
