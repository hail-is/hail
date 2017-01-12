package is.hail.driver

import breeze.linalg.DenseVector
import org.apache.spark.SparkContext
import is.hail.utils.{fatal, plural}
import is.hail.stats.{BaldingNicholsModel, BaldingNicholsModelDist}
import is.hail.variant.VariantDataset

object BaldingNicholsModelCommand {

  /*def name = "balding_nichols"
  def description = "Generate a variant dataset using the Balding-Nichols model"
  class Options extends BaseOptions
  def newOptions = new Options
  def supportsMultiallelic = true
  def requiresVDS = false
  def run(state: State, options: Options): State = fatal("balding_nichols is only supported via Python")
  */

//  def balding_nichols(): Unit = {
//    print("yay")
//  }
  def balding_nichols(sc: SparkContext,
    nPops: Int,
    nSamples: Int,
    nVariants: Int,
    popDist: Array[Double],
    FstOfPop: Array[Double],
    seed: Int,
    nPartitions: Int): VariantDataset = {
    if (nPops <= 0)
      fatal(s"Number of populations must be positive, got ${ nPops }")

    if (nSamples <= 0)
      fatal(s"Number of samples must be positive, got ${ nSamples }")

    if (nVariants <= 0)
      fatal(s"Number of variants must be positive, got ${ nVariants }")


    if (popDist.size != nPops)
      fatal(s"Got ${ nPops } populations but ${ popDist.size } ${ plural(popDist.size, "probability", "probabilities") }")
    popDist.foreach(p =>
      if (p < 0d)
        fatal(s"Population probabilities must be non-negative, got $p"))


    if (FstOfPop.size != nPops)
      fatal(s"Got ${ nPops } populations but ${ FstOfPop.size } ${ plural(FstOfPop.size, "value") }")

    FstOfPop.foreach(f =>
      if (f <= 0d || f >= 1d)
        fatal(s"F_st values must satisfy 0.0 < F_st < 1.0, got $f"))



    if (nPartitions <= 0)
      fatal(s"Number of partitions must be positive, got $nPartitions")


    BaldingNicholsModelDist(sc,
        nPops,
        nSamples,
        nVariants,
        DenseVector(popDist),
        DenseVector(FstOfPop),
        seed)
  }
}
