package org.broadinstitute.hail.driver

import org.apache.spark.RangePartitioner
import org.apache.spark.storage.StorageLevel
import org.broadinstitute.hail.methods.SamplePCA
import org.broadinstitute.hail.variant.Variant
import org.apache.spark.mllib.linalg.{Vector => SVector}
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

    //FIXME: Add gzip options with .gz (use Tim's utility function HadoopStripCodec)
    val filename = options.output.stripSuffix(".tsv")

    val (scores, loadings, eigenvalues) = (new SamplePCA(options.k, options.l, options.e))(vds)

    writeTextFile(filename + ".tsv", state.hadoopConf) { s =>
      s.write("sample\t" + (1 to options.k).map("PC" + _).mkString("\t") + "\n")
      for ((ls, i) <- vds.localSamples.zipWithIndex) {
        s.write(vds.sampleIds(ls))
        for (j <- 0 until options.k)
          s.write("\t" + scores(i, j))
        s.write("\n")
      }
    }

    if (options.l) {
      loadings.persist(StorageLevel.MEMORY_AND_DISK)
      val vls = loadings.repartitionAndSortWithinPartitions(new RangePartitioner[Variant, Array[Double]](loadings.partitions.length, loadings))
      vls.persist(StorageLevel.MEMORY_AND_DISK)
      vls
        .map{ case (v, l) => v.contig + "\t" + v.start + "\t" + v.ref + "\t" + v.alt + "\t" + l.mkString("\t")}
        .writeTable(filename + ".loadings.tsv", Some("chrom\tpos\tref\talt" + "\t" + (1 to options.k).map("PC" + _).mkString("\t")))
      loadings.unpersist()
      vls.unpersist()
    }

    if (options.e)
      writeTextFile(filename + ".eigen.tsv", state.hadoopConf) { s =>
        for (i <- 0 until options.k)
          s.write(eigenvalues(i) + "\n")
      }

    state
  }
}
