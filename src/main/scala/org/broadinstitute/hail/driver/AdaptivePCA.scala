package org.broadinstitute.hail.driver

import org.apache.spark.RangePartitioner
import org.apache.spark.storage.StorageLevel
import org.broadinstitute.hail.methods.AdaptivePCA
import org.broadinstitute.hail.variant.Variant
import org.apache.spark.mllib.linalg.{Vector => SVector}
import org.kohsuke.args4j.{Option => Args4jOption}
import org.broadinstitute.hail.Utils._

object AdaptivePCA extends Command {
  def name = "adaptivepca"

  def description = "Use Adaptive PCA to cluster samples into homogenous groups"

  class Options extends BaseOptions {
    @Args4jOption(required = true, name = "-o", aliases = Array("--output"), usage = "Output file")
    var output : String = _

    @Args4jOption(required = true, name = "-i", aliases = Array("--iterations"), usage = "Number of AdaptivePCA iterations to complete before returning")
    var iterations : Int = _

    @Args4jOption(required = false, name = "-k", aliases = Array("--components"), usage = "Number of principal components")
    var k : Int = 10

  }

  def newOptions = new Options

  def supportsMultiallelic = true

  def requiresVDS = true

  def run(state: State, options: Options): State = {

    val vds = state.vds
    val APCA = new AdaptivePCA(options.k)
    val T = APCA(vds,options.iterations)
    val clusts = APCA.leaves(T).toSeq

    writeTextFile(options.output, state.hadoopConf) { s =>
      s.write("Sample\tCluster\n")
      for (i <- 0 until clusts.size) {
        for (j <- clusts(i))
          s.write(j + "\t" + i + "\n")
      }
    }
    state
  }
}
