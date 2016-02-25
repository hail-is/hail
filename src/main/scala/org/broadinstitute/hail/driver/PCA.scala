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
  }
  def newOptions = new Options

  def run(state: State, options: Options): State = {
    val vds = state.vds

    val samplePCs = (new SamplePCA(options.k))(vds)

    writeTextFile(options.output, state.hadoopConf) { s =>
      s.write("sample")
      for (i <- 0 until options.k)
        s.write("\t" + "PC" + i)
      s.write("\n")

      for (i <- 0 until vds.nSamples) {
        s.write(vds.sampleIds(i))
        for (j <- 0 until options.k)
          s.write("\t" + samplePCs(i)(j))
        s.write("\n")
      }
    }

    state
  }
}
