package org.broadinstitute.hail.driver

import org.broadinstitute.hail.methods.{Pedigree, MendelErrors}
import org.kohsuke.args4j.{Option => Args4jOption}

import scala.language.postfixOps
import scala.sys.process._

object MendelErrorsCommand extends Command {

  def name = "mendelerrors"
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
    val men = MendelErrors(state.vds, ped.completeTrios)

    men.writeMendel(options.output + ".mendel")
    men.writeMendelL(options.output + ".lmendel")
    men.writeMendelF(options.output + ".fmendel")
    men.writeMendelI(options.output + ".imendel")

    state
  }
}
