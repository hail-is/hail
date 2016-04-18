package org.broadinstitute.hail.rest

import org.broadinstitute.hail.Utils._
import org.broadinstitute.hail.driver.{State, Command}
import org.broadinstitute.hail.variant.HardCallSet
import org.http4s.server.blaze.BlazeBuilder
import org.kohsuke.args4j.{Option => Args4jOption}

object T2DServer extends Command {

  class Options extends BaseOptions {
    @Args4jOption(required = true, name = "-c", aliases= Array("--covariate-file"), usage = "Covariate file")
    var covFile: String = _

    @Args4jOption(required = false, name = "-p", aliases = Array("--port"), usage = "Service port")
    var port: Int = 6062

    @Args4jOption(required = true, name = "-b1", aliases = Array("--hcs1Mb"), usage = ".hcs with 1Mb block")
    var hcs1MbFile: String = _

    @Args4jOption(required = true, name = "-b10", aliases = Array("--hcs10Mb"), usage = ".hcs with 1Mb block")
    var hcs10MbFile: String = _
  }

  def newOptions = new Options

  def name = "t2dserver"

  def description = "Run T2D REST server"

  override def supportsMultiallelic = true

  def run(state: State, options: Options): State = {
    val hcs = state.hcs
    if (hcs == null)
      fatal("hard call set required")

    val hcs1Mb = HardCallSet.read(state.sqlContext, options.hcs1MbFile)

    val hcs10Mb = HardCallSet.read(state.sqlContext, options.hcs10MbFile).cache()

    val cov = CovariateData.read(options.covFile, state.hadoopConf, hcs.sampleIds)

    val service = new T2DService(hcs, hcs1Mb, hcs10Mb, cov)
    val task = BlazeBuilder.bindHttp(options.port, "0.0.0.0")
      .mountService(service.service, "/")
      .run
    task.awaitShutdown()

    state
  }
}
