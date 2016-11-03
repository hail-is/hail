package org.broadinstitute.hail.driver

import breeze.linalg.DenseVector
import org.apache.commons.math3.random.JDKRandomGenerator
import org.broadinstitute.hail.utils.{fatal, plural}
import org.broadinstitute.hail.expr.Parser
import org.broadinstitute.hail.stats.BaldingNicholsModel
import org.kohsuke.args4j.{Option => Args4jOption}

object BaldingNicholsModelCommand extends Command {

  class Options extends BaseOptions {
    @Args4jOption(required = true, name = "-k", aliases = Array("--populations"), usage = "Number of populations")
    var nPops: Int = 0

    @Args4jOption(required = true, name = "-n", aliases = Array("--samples"), usage = "Number of samples")
    var nSamples: Int = 0

    @Args4jOption(required = true, name = "-m", aliases = Array("--variants"), usage = "Number of variants")
    var nVariants: Int = 0

    @Args4jOption(required = false, name = "-d", aliases = Array("--popdist"), usage = "(Unnormalized) population distribution, comma-separated")
    var popDist: String = _

    @Args4jOption(required = false, name = "-f", aliases = Array("--fst"), usage = "F_st values, comma-separated")
    var FstOfPop: String = _

    @Args4jOption(required = false, name = "-r", aliases = Array("--root"), usage = "Period-delimited path to follow `global', `sa', and `va'")
    var root: String = "bn"

    @Args4jOption(required = false, name = "-s", aliases = Array("--seed"), usage = "Random seed")
    var seed: java.lang.Integer = _

    @Args4jOption(required = false, name = "-p", aliases = Array("--npartitions"), usage = "Number of partitions")
    var nPartitions: java.lang.Integer = _
  }

  def newOptions = new Options

  def name = "baldingnichols"

  def description = "Generate a variant dataset using the Balding-Nichols model"

  def supportsMultiallelic = true

  def requiresVDS = false

  override def hidden = true

  def run(state: State, options: Options): State = {
    if (options.nPops <= 0)
      fatal(s"Number of populations must be positive, got ${options.nPops}")

    if (options.nSamples <= 0)
      fatal(s"Number of samples must be positive, got ${options.nSamples}")

    if (options.nVariants <= 0)
      fatal(s"Number of variants must be positive, got ${options.nVariants}")

    val popDist = Option(options.popDist).map(Parser.parseCommaDelimitedDoubles)
    popDist.foreach { probs =>
      if (probs.size != options.nPops)
        fatal(s"Got ${options.nPops} populations but ${probs.size} ${plural(probs.size, "probability", "probabilities")}")
      probs.foreach(p =>
        if (p < 0d)
          fatal(s"Population probabilities must be non-negative, got $p"))
    }

    val FstOfPop = Option(options.FstOfPop).map(Parser.parseCommaDelimitedDoubles)
    FstOfPop.foreach { fs =>
      if (fs.length != options.nPops)
        fatal(s"Got ${options.nPops} populations but ${fs.size} ${plural(fs.size, "value")}")
      fs.foreach(f =>
        if (f <= 0d || f >= 1d)
          fatal(s"Fst values must be strictly between 0.0 and 1.0, got $f"))
    }

    val seed = Option(options.seed).map(_.intValue()).getOrElse(scala.util.Random.nextInt())

    val nPartitions = Option(options.nPartitions).map(_.intValue())
    nPartitions.foreach {n =>
      if (n <= 0)
        fatal(s"Number of partitions must be positive, got $n")
    }

    state.copy(vds =
      BaldingNicholsModel(
        options.nPops,
        options.nSamples,
        options.nVariants,
        popDist.map(DenseVector(_)),
        FstOfPop.map(DenseVector(_)),
        seed
      )
        .toVDS(state.sc, options.root, nPartitions))
  }
}
