package org.broadinstitute.hail.driver

import org.broadinstitute.hail.expr.{EvalContext, TSample}
import org.broadinstitute.hail.methods.{LinearRegression, LinRegUtils}
import org.kohsuke.args4j.{Option => Args4jOption}
import org.broadinstitute.hail.Utils._

object GroupTestLinReg extends Command {

  class Options extends BaseOptions {
    @Args4jOption(required = true, name = "-y", aliases = Array("--y"), usage = "Response sample annotation")
    var ySA: String = _

    @Args4jOption(required = false, name = "-c", aliases = Array("--covariates"), usage = "Covariate sample annotations, comma-separated")
    var covSA: String = ""

    @Args4jOption(required = true, name = "-o", aliases = Array("--output"), usage = "path of output tsv")
    var output: String = _
  }

  def newOptions = new Options

  def name = "grouptest linreg"

  def description = "Run linear regression on previously defined groups"

  def supportsMultiallelic = true

  def requiresVDS = true

  def run(state: State, options: Options): State = {
    val vds = state.vds
    val group = state.group

    if (group == null)
      fatal("No group has been created. Use the `creategroup` command.")

    val symTab = Map(
      "s" -> (0, TSample),
      "sa" -> (1, vds.saSignature))

    val ec = EvalContext(symTab)

    val (completeSampleSet, y, cov) = LinRegUtils.prepareCovMatrixAndY(vds.sampleIdsAndAnnotations, options.ySA, options.covSA, ec)

    val sampleIds = vds.sampleIds

    val inputData = group.map{ case (k, v) =>
      (k, v.zipWithIndex.filter{ case (d, i) =>
        completeSampleSet(sampleIds(i))
      }.map{case (d, i) => d}.toIterable)}

    val linreg = LinearRegression[IndexedSeq[Any]](state.sc, y, cov, inputData)

    hadoopDelete(options.output, state.hadoopConf, recursive = true)

    val header = "groupKey\tnMissing\tbeta\tse\tt\tp"

    linreg.rdd
      .mapPartitions { it =>
        val sb = new StringBuilder()
        it.map { case (k, v) =>
          sb.clear()
          sb.append(k.mkString(","))
          sb.append("\t")
          sb.append(v.getOrElse("NA\tNA\tNA\tNA\tNA\tNA\n").toString)
          sb.result()
        }
      }.writeTable(options.output, Some(header))

    state
  }
}