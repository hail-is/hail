package org.broadinstitute.hail.driver

import org.broadinstitute.hail.methods.{Pedigree, TDT}
import org.kohsuke.args4j.{Option => Args4jOption}


object TDTCommand extends Command {

  def name = "tdt"

  def description = "Perform the family-based transmission disequilibrium test per variant"

  class Options extends BaseOptions {
    @Args4jOption(required = true, name = "-o", aliases = Array("--output"), usage = "Output root filename")
    var output: String = _

    @Args4jOption(required = true, name = "-f", aliases = Array("--fam"), usage = ".fam file")
    var famFilename: String = _
  }
  def newOptions = new Options

  def run(state: State, options: Options): State = {
    val ped = Pedigree.read(options.famFilename, state.hadoopConf, state.vds.sampleIds)
    val tdt = TDT(state.vds, ped.completeTrios)
    TDT.write(tdt, options.output + ".tdt")
    state
  }

}
