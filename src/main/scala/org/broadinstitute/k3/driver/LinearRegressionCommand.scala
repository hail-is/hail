package org.broadinstitute.k3.driver

import org.broadinstitute.k3.methods.{LinearRegression, CovariateData, Pedigree}
import org.kohsuke.args4j.{Option => Args4jOption}

import scala.language.postfixOps
import scala.sys.process._

object LinearRegressionCommand extends Command {

  def name = "linreg"
  def description = "Compute beta, t-stat, and p-val for each SNP with additional sample covariates"

  class Options extends BaseOptions {
    @Args4jOption(required = true, name = "-o", aliases = Array("--output"), usage = "Output root filename")
    var output: String = _

    @Args4jOption(required = true, name = "-f", aliases = Array("--fam"), usage = ".fam file")
    var famFilename: String = _

    @Args4jOption(required = true, name = "-c", aliases = Array("--cov"), usage = ".cov file")
    var covFilename: String = _
  }
  def newOptions = new Options

  def run(state: State, options: Options): State = {
    val vds = state.vds
    val ped = Pedigree.read(options.famFilename, vds.sampleIds)
    val cov = CovariateData.read(options.covFilename, vds.sampleIds)
    val linreg = LinearRegression(vds, ped, cov)

    val result = "rm -rf " + options.output !;

    linreg.write(options.output)

    state
  }
}
