package org.broadinstitute.hail.driver

import org.broadinstitute.hail.variant.VariantSampleMatrix
import org.kohsuke.args4j.{Option => Args4jOption}

object CompareVDS extends Command {

  class Options extends BaseOptions {
    @Args4jOption(required = true, name = "-i", aliases = Array("--input"), usage = ".vds file to compare to")
    var input: String = _
  }

  def newOptions = new Options

  def name = "comparevds"

  def description = "Print number of samples, variants, and called genotypes in current dataset"

  def supportsMultiallelic = true

  def requiresVDS = true

  override def hidden = true

  def run(state: State, options: Options): State = {
    val vds1 = state.vds
    val vds2 = VariantSampleMatrix.read(state.sqlContext, options.input)

    println(vds1.same(vds2))

    state
  }
}
