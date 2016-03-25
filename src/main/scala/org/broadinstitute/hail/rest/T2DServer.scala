package org.broadinstitute.hail.rest

import org.broadinstitute.hail.Utils._
import org.broadinstitute.hail.driver.{State, Command}
import org.broadinstitute.hail.methods.CovariateData
import org.http4s.server.blaze.BlazeBuilder
import org.kohsuke.args4j.{Option => Args4jOption}

object T2DServer extends Command {

  class Options extends BaseOptions {
    @Args4jOption(name = "-p", aliases = Array("--port"), usage = "Service port")
    var port: Int = 6061

    @Args4jOption(required = true, name = "-c", aliases= Array("--covariate-file"), usage = "Covariate file")
    var covFile: String = _
  }

  def newOptions = new Options

  def name = "t2dserver"

  def description = "Run T2D REST server"

  override def supportsMultiallelic = true

  def run(state: State, options: Options): State = {
    val hcs = state.hcs
    if (hcs == null)
      fatal("hard call set required")

    val cov = CovariateData.read(options.covFile, state.hadoopConf, hcs.sampleIds)

    val service = new T2DService(hcs, cov)
    val task = BlazeBuilder.bindHttp(options.port)
      .mountService(service.service, "/")
      .run
    task.awaitShutdown()

    state
  }
}
