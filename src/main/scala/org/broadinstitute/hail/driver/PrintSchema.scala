package org.broadinstitute.hail.driver

import org.broadinstitute.hail.Utils._
import org.broadinstitute.hail.annotations.Annotation
import org.kohsuke.args4j.{Option => Args4jOption}

object PrintSchema extends Command {

  class Options extends BaseOptions {
    @Args4jOption(name = "-o", aliases = Array("--output"), usage = "Output file")
    var output: String = _

    @Args4jOption(name = "--attributes", usage = "Print attributes on all fields")
    var attributes: Boolean = _
  }

  def newOptions = new Options

  def name = "printschema"

  def description = "Shows the schema for global, sample, and variant annotations"

  override def supportsMultiallelic = true

  def run(state: State, options: Options): State = {
    val vds = state.vds

    if (vds == null)
      fatal("printschema requires a non-null variant dataset, import or read one first")

    val sb = new StringBuilder()

    sb.append("Global annotation schema:\n")
    sb.append("global: ")
    vds.metadata.globalSignature.pretty(sb, 0, printAttrs = options.attributes)

    sb += '\n'
    sb += '\n'

    sb.append("Sample annotation schema:\n")
    sb.append("sa: ")
    vds.metadata.saSignature.pretty(sb, 0, printAttrs = options.attributes)

    sb += '\n'
    sb += '\n'

    sb.append("Variant annotation schema:\n")
    sb.append("va: ")
    vds.metadata.vaSignature.pretty(sb, 0, printAttrs = options.attributes)
    sb += '\n'

    val result = sb.result()
    options.output match {
      case null => print(result)
      case out => writeTextFile(out, vds.sparkContext.hadoopConfiguration) { dos =>
        dos.write(result)
      }
    }

    state
  }
}
