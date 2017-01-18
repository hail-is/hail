package is.hail.driver

import is.hail.variant.VariantSampleMatrix
import org.kohsuke.args4j.{Option => Args4jOption}

object ReadKudu extends Command {
  def name = "readkudu"

  def description = "Load from Kudu as the current dataset"
  override def supportsMultiallelic = true
  override def requiresVDS = false

  class Options extends BaseOptions {
    @Args4jOption(required = true, name = "-i", aliases = Array("--input"), usage = "Input .vds file")
    var input: String = _

    @Args4jOption(required = true, name = "-t", aliases = Array("--table"), usage = "Table name")
    var table: String = _

    @Args4jOption(required = true, name = "-m", aliases = Array("--master"), usage = "Kudu master address")
    var master: String = _
  }

  def newOptions = new Options

  def run(state: State, options: Options): State = {
    val input = options.input

    val newVDS = VariantSampleMatrix.readKudu(state.sqlContext, input, options.table,
      options.master)
    state.copy(vds = newVDS)
  }
}
