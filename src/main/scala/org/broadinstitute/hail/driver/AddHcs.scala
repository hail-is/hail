package org.broadinstitute.hail.driver

import org.broadinstitute.hail.methods.{CovariateData, Pedigree}
import org.broadinstitute.hail.variant.HardCallSet
import org.kohsuke.args4j.{Option => Args4jOption}

object AddHcs extends Command {

  class Options extends BaseOptions {
    @Args4jOption(required = false, name = "-f", aliases = Array("--fam"), usage = ".fam file")
    var famFilename: String = null

    @Args4jOption(required = false, name = "-c", aliases = Array("--cov"), usage = ".cov file")
    var covFilename: String = null

    @Args4jOption(required = false, name = "-s", aliases = Array("--sparse"), usage = "Sparse cut off, < s is sparse: s <= 0 is all dense, s > 1 is all sparse")
    var sparseCutOff: Double = .15
  }

  def newOptions = new Options

  def name = "addhcs"

  def description = "Add hard call set to state, filtering samples to those phenotyped and/or with covariates"

  def run(state: State, options: Options): State = {
    val vds = state.vds

    val sampleFilter: Int => Boolean =
      (options.famFilename, options.covFilename) match {
        case (null, null) => s => true
        case (_, null) =>
          Pedigree.read(options.famFilename, state.hadoopConf, vds.sampleIds).phenotypedSamples
        case (null, _) =>
          CovariateData.read(options.covFilename, state.hadoopConf, vds.sampleIds).covRowSample.toSet
        case _ =>
          Pedigree.read(options.famFilename, state.hadoopConf, vds.sampleIds).phenotypedSamples intersect
            CovariateData.read(options.covFilename, state.hadoopConf, vds.sampleIds).covRowSample.toSet
      }

    val hcs = HardCallSet(state.sqlContext, vds.filterSamples { case (s, sa) => sampleFilter(s) }, options.sparseCutOff)

    state.copy(hcs = hcs)
  }
}
