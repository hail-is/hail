package org.broadinstitute.hail.rest

import org.broadinstitute.hail.Utils._
import org.broadinstitute.hail.driver.{Command, State}
import org.broadinstitute.hail.variant.HardCallSet
import org.http4s.server.blaze.BlazeBuilder
import org.kohsuke.args4j.{Option => Args4jOption}

object T2DServer extends Command {

  class Options extends BaseOptions {
    @Args4jOption(required = true, name = "-c", aliases= Array("--covariate-file"), usage = "Covariate file")
    var covFile: String = _

    @Args4jOption(required = false, name = "-p", aliases = Array("--port"), usage = "Service port")
    var port: Int = 8080


    @Args4jOption(required = true, name = "-h1", aliases = Array("--hcs100Kb"), usage = ".hcs with 100Kb block")
    var hcsFile: String = _

    @Args4jOption(required = true, name = "-h2", aliases = Array("--hcs1Mb"), usage = ".hcs with 1Mb block")
    var hcs1MbFile: String = _

    @Args4jOption(required = true, name = "-h3", aliases = Array("--hcs10Mb"), usage = ".hcs with 10Mb block")
    var hcs10MbFile: String = _

    @Args4jOption(required = false, name = "-m", aliases = Array("--minmac"), usage = "default minimum MAC")
    var defaultMinMAC: Int = 0
  }

  def newOptions = new Options

  def name = "t2dserver"

  def description = "Run T2D REST server"

  def supportsMultiallelic = true

  def requiresVDS = false

  def readCovData(state: State, covFile: String, sampleIds: IndexedSeq[String]): Map[String, Array[Double]] = {
    val (covNames, sampleCovs): (Array[String], Map[String, Array[Double]]) =
      readLines(covFile, state.hadoopConf) { lines =>
        if (lines.isEmpty)
          fatal("empty TSV file")

        val fieldNames = lines.next().value.split("\\t")
        val nFields = fieldNames.size

        (fieldNames.drop(1),
          lines.map {
            _.transform { l =>
              val lineSplit = l.value.split("\\t")
              if (lineSplit.length != nFields)
                fatal(s"expected $nFields fields, but got ${lineSplit.length}")
              (lineSplit(0), lineSplit.drop(1).map(_.toDouble))
            }
          }.toMap
        )
      }

    if (! sampleIds.forall(sampleCovs.keySet(_)))
      throw new RESTFailure("Not all samples in the hard call set are listed in the phenotype data set")

    covNames
      .zipWithIndex
      .map{ case (name, j) => (name, sampleIds.map(s => sampleCovs(s)(j)).toArray) }.toMap
  }


  def run(state: State, options: Options): State = {

    val hcs = HardCallSet.read(state.sqlContext, options.hcsFile)
    val hcs1Mb = HardCallSet.read(state.sqlContext, options.hcs1MbFile)
    val hcs10Mb = HardCallSet.read(state.sqlContext, options.hcs10MbFile)

    val covMap = readCovData(state, options.covFile, hcs.sampleIds)

    assert(hcs.sampleIds == hcs1Mb.sampleIds)
    assert(hcs1Mb.sampleIds == hcs10Mb.sampleIds)

    val service = new T2DService(hcs, hcs1Mb, hcs10Mb, covMap, options.defaultMinMAC)
    val task = BlazeBuilder.bindHttp(options.port, "0.0.0.0")
      .mountService(service.service, "/")
      .run
    task.awaitShutdown()

    state
  }
}
