package org.broadinstitute.hail.driver

import org.broadinstitute.hail.expr._
import org.broadinstitute.hail.methods._
import org.kohsuke.args4j.{Option => Args4jOption}

import scala.language.postfixOps

object TDTCommand extends Command {

  def name = "tdt"
  def description = "Find Mendel errors; count per variant, individual, nuclear family"

  class Options extends BaseOptions {
    @Args4jOption(required = true, name = "-o", aliases = Array("--output"), usage = "Output root filename")
    var output: String = _

    @Args4jOption(required = true, name = "-f", aliases = Array("--fam"), usage = ".fam file")
    var famFilename: String = _
  }
  def newOptions = new Options

  def supportsMultiallelic = false

  def requiresVDS = true

  def run(state: State, options: Options): State = {
    val ped = Pedigree.read(options.famFilename, state.hadoopConf, state.vds.sampleIds)
    val results = TDT(state.vds, ped.completeTrios)

    val signature = TStruct("nT" -> TInt, "nU" -> TInt, "chi" -> TDouble)

    state.copy(vds = state.vds.annotateVariants(results, signature, List("tdt")))
  }
}
