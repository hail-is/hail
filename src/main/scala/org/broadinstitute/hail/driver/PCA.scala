package org.broadinstitute.hail.driver

import org.broadinstitute.hail.methods.SamplePCA
import org.kohsuke.args4j.{Option => Args4jOption}
import org.broadinstitute.hail.Utils._

object PCA extends Command {
  def name = "pca"
  def description = "Compute PCA on the matrix of genotypes"

  class Options extends BaseOptions {
    @Args4jOption(required = true, name = "-o", aliases = Array("--output"), usage = "Output file")
    var output: String = _

    @Args4jOption(required = false, name = "-k", aliases = Array("--components"), usage = "Number of principal components")
    var k: Int = 10

    @Args4jOption(required = false, name = "-l", aliases = Array("--loadings"), usage = "Compute loadings")
    var l: Boolean = _

    @Args4jOption(required = false, name = "-e", aliases = Array("--eigenvalues"), usage = "Compute eigenvalues")
    var e: Boolean = _

  }
  def newOptions = new Options

  def run(state: State, options: Options): State = {
    val vds = state.vds

    val (scores, loadings, eigenvalues) = (new SamplePCA(options.k, options.l, options.e))(vds)

    writeTextFile(options.output, state.hadoopConf) { s =>
      s.write("sample")
      for (i <- 0 until options.k)
        s.write("\t" + "PC" + (i + 1))
      s.write("\n")

      for (i <- 0 until vds.nSamples) {
        s.write(vds.sampleIds(i))
        for (j <- 0 until options.k)
          s.write("\t" + scores(i)(j))
        s.write("\n")
      }
    }

    state
  }
}
