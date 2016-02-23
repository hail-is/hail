package org.broadinstitute.hail.driver

import org.broadinstitute.hail.Utils._
import org.broadinstitute.hail.annotations.{AnnotationSignatures, AnnotationSignature, Annotations}
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

  def printSignatures(sb: StringBuilder, a: Annotations, spaces: Int, path: String) {
    val spacing = (0 until spaces).map(i => " ").fold("")(_ + _)
    val values = new mutable.ArrayBuilder.ofRef[(String, AnnotationSignature)]()
    val subAnnotations = new mutable.ArrayBuilder.ofRef[(String, Annotations)]()

    a.attrs.foreach {
      case (k, v) =>
        v match {
          case sig: AnnotationSignature => values += ((k, sig))
          case anno: Annotations => subAnnotations += ((k, anno))
          case _ => fatal("corrupt annotation signatures")
        }
    }

    values.result().sortBy {
      case (key, sig) => key
    }
      .foreach {
        case (key, sig) =>
          sb.append(s"""$spacing$key: ${sig.typeOf}""")
          sb.append("\n")
      }

    subAnnotations.result().sortBy {
      case (key, anno) => key
    }
      .foreach {
        case (key, anno) =>
          sb.append(s"""$spacing$key: $path.$key.<identifier>""")
          sb.append("\n")
          printSignatures(sb, anno, spaces + 2, path + "." + key)
      }
  }

  def printSignatures(sb: StringBuilder, a: AnnotationSignatures, spaces: Int, path: String) {
    val spacing = (0 until spaces).map(i => " ").fold("")(_ + _)
    val values = new mutable.ArrayBuilder.ofRef[(String, AnnotationSignature)]()
    val subAnnotations = new mutable.ArrayBuilder.ofRef[(String, AnnotationSignatures)]()

    a.attrs.foreach {
      case (k, v) =>
        v match {
          case anno: AnnotationSignatures => subAnnotations += ((k, anno))
          case sig: AnnotationSignature => values += ((k, sig))
          case _ => fatal("corrupt annotation signatures")
        }
    }

    values.result().sortBy {
      case (key, sig) => key
    }
      .foreach {
        case (key, sig) =>
          sb.append(s"""$spacing$key: ${sig.typeOf} [${sig.index.path.mkString(",")}, ${sig.index.last}]""")
          sb.append("\n")
      }

    subAnnotations.result().sortBy {
      case (key, anno) => key
    }
      .foreach {
        case (key, anno) =>
          sb.append(s"""$spacing$key: $path.$key.<identifier> [${anno.index.path.mkString(",")}, ${anno.index.last}]""")
          sb.append("\n")
          printSignatures(sb, anno, spaces + 2, path + "." + key)
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
