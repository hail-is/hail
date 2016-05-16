package org.broadinstitute.hail.driver

import org.broadinstitute.hail.Utils._
import org.broadinstitute.hail.annotations.Annotation
import org.kohsuke.args4j.{Option => Args4jOption}

object PrintSchema extends Command {

  class Options extends BaseOptions {
    @Args4jOption(name = "-o", aliases = Array("--output"), usage = "Output file")
    var output: String = _

    @Args4jOption(name = "--attributes", usage = "Print attributes on all fields")
    var attributes: Boolean = false

    @Args4jOption(name = "--va", usage = "Print the variant annotation schema")
    var va: Boolean = false

    @Args4jOption(name = "--sa", usage = "Print the sample annotation schema")
    var sa: Boolean = false

    @Args4jOption(name = "--global", usage = "Print the global annotation schema")
    var global: Boolean = false

  }

  def newOptions = new Options

  def name = "printschema"

  def description = "Shows the schema for global, sample, and variant annotations"

  def supportsMultiallelic = true

  def requiresVDS = true

  def run(state: State, options: Options): State = {
    val vds = state.vds

    if (vds == null)
      fatal("printschema requires a non-null variant dataset, import or read one first")

    val sb = new StringBuilder()

    val printAll = !options.va && !options.sa && !options.global

    if (printAll || options.global) {
      sb.append("Global annotation schema:\n")
      sb.append("global: ")
      vds.metadata.globalSignature.pretty(sb, 0, printAttrs = options.attributes)

      sb += '\n'
      sb += '\n'
    }

    if (printAll || options.sa) {
      sb.append("Sample annotation schema:\n")
      sb.append("sa: ")
      vds.metadata.saSignature.pretty(sb, 0, printAttrs = options.attributes)

      sb += '\n'
      sb += '\n'
    }

    if (printAll || options.va) {
      sb.append("Variant annotation schema:\n")
      sb.append("va: ")
      vds.metadata.vaSignature.pretty(sb, 0, printAttrs = options.attributes)
      sb += '\n'
    }

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
