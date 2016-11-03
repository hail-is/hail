package org.broadinstitute.hail.driver

import org.broadinstitute.hail.expr._
import org.broadinstitute.hail.utils._
import org.json4s.jackson.JsonMethods._
import org.kohsuke.args4j.{Option => Args4jOption}

object ExportGlobals extends Command {

  class Options extends BaseOptions {
    @Args4jOption(required = true, name = "-o", aliases = Array("--output"), usage = "Output file")
    var output: String = _
  }

  def newOptions = new Options

  def name = "exportglobals"

  def description = "Export all global annotations as JSON"

  def supportsMultiallelic = true

  def requiresVDS = true

  def run(state: State, options: Options): State = {
    val vds = state.vds

    val jsonString =
      if (vds.globalSignature == TStruct.empty)
        "{}"
      else
        pretty(render(vds.globalSignature.toJSON(vds.globalAnnotation)))

    state.hadoopConf.writeTextFile(options.output) { out => out.write(jsonString) }

    state
  }
}
