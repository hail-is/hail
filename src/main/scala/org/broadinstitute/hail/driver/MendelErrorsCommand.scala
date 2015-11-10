package org.broadinstitute.hail.driver

import org.broadinstitute.hail.methods.{Pedigree, MendelErrors}
import org.kohsuke.args4j.{Option => Args4jOption}

import scala.language.postfixOps
import scala.sys.process._

object MendelErrorsCommand extends Command {

  def name = "mendelerrors"
  def description = "Compute Mendel errors and count per variant, nuclear family, and individual"

  class Options extends BaseOptions {
    @Args4jOption(required = true, name = "-o", aliases = Array("--output"), usage = "Output root filename")
    var output: String = _

    @Args4jOption(required = true, name = "-f", aliases = Array("--fam"), usage = ".fam file")
    var famFilename: String = _
  }
  def newOptions = new Options

  def run(state: State, options: Options): State = {
    val vds = state.vds
    val ped = Pedigree.read(options.famFilename, vds.sampleIds)
    val men = MendelErrors(vds, ped.completeTrios)

    val result1 = "rm -rf " + options.output + ".mendel" !;
    val result2 = "rm -rf " + options.output + ".lmendel" !;

    men.writeMendel(options.output + ".mendel")
    men.writeMendelL(options.output + ".lmendel")
    men.writeMendelF(options.output + ".fmendel")
    men.writeMendelI(options.output + ".imendel")

    state
  }
}
