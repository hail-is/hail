package is.hail.driver

import is.hail.utils._
import is.hail.expr._
import org.json4s.jackson.JsonMethods._
import org.kohsuke.args4j.{Option => Args4jOption}

object ShowGlobalAnnotations extends Command {

  class Options extends BaseOptions {
    @Args4jOption(name = "-o", aliases = Array("--output"), usage = "Output file")
    var output: String = _
  }

  def newOptions = new Options

  def name = "showglobals"

  def description = "Print or export all global annotations as JSON"

  def supportsMultiallelic = true

  def requiresVDS = true

  def run(state: State, options: Options): State = {
    val vds = state.vds

    val jsonString =
      if (vds.globalSignature == TStruct.empty)
        "{}"
      else
        pretty(render(vds.globalSignature.toJSON(vds.globalAnnotation)))

    options.output match {
      case null => info(s"Global annotations: `global' = \n$jsonString")
      case path => state.hadoopConf.writeTextFile(path) { out => out.write(jsonString) }
    }

    state
  }
}
