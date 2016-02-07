package org.broadinstitute.hail.driver

import org.broadinstitute.hail.Utils._
import org.broadinstitute.hail.methods.{LinearRegressionFromHardCallSet, CovariateData, Pedigree}
import org.broadinstitute.hail.variant.HardCallSet
import org.kohsuke.args4j.{Option => Args4jOption}

object WriteHardCallSet extends Command {
  class Options extends BaseOptions {
    @Args4jOption(required = true, name = "-o", aliases = Array("--output"), usage = "Output file")
    var output: String = _

    @Args4jOption(required = false, name = "-f", aliases = Array("--fam"), usage = ".fam file")
    var famFilename: String = null

    @Args4jOption(required = false, name = "-c", aliases = Array("--cov"), usage = ".cov file")
    var covFilename: String = null
  }

  def newOptions = new Options

  def name = "writehcs"
  def description = "Write current dataset as .hcs file, filtering samples to those phenotyped and/or with covariates"

  def run(state: State, options: Options): State = {
    val vds = state.vds

    val sampleFilter: Int => Boolean =
      (options.famFilename, options.covFilename) match {
        case (null, null) => s => true
        case (   _, null) =>
          Pedigree.read(options.famFilename, state.hadoopConf, vds.sampleIds).phenotypedSamples
        case (null,    _) =>
          CovariateData.read(options.covFilename, state.hadoopConf, vds.sampleIds).covRowSample.toSet
        case _            =>
          Pedigree.read(options.famFilename, state.hadoopConf, vds.sampleIds).phenotypedSamples intersect
          CovariateData.read(options.covFilename, state.hadoopConf, vds.sampleIds).covRowSample.toSet
      }

    val hcs = HardCallSet(vds.filterSamples{ case (s, sa) => sampleFilter(s) })

    hadoopDelete(options.output, state.hadoopConf, true)
    hcs.write(state.sqlContext, options.output)

    state
  }
}
