package org.broadinstitute.hail.rest

import org.broadinstitute.hail.utils._
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

    @Args4jOption(required = true, name = "-hcs", aliases = Array("--hcs"), usage = "hard call set")
    var hcsFile: String = _

    @Args4jOption(required = false, name = "-m", aliases = Array("--minmac"), usage = "default minimum MAC")
    var defaultMinMAC: Int = 0

    @Args4jOption(required = false, name = "-w", aliases = Array("--maxwidth"), usage = "maximum interval width")
    var maxWidth: Int = 600000

    @Args4jOption(required = false, name = "-l", aliases = Array("--limit"), usage = "maximum number of variants returned")
    var hardLimit: Int = 100000
  }

  def newOptions = new Options

  def name = "t2dserver"

  def description = "Run T2D REST server"

  def supportsMultiallelic = true

  def requiresVDS = false

  def readCovData(state: State, covFile: String, sampleIds: IndexedSeq[String]): Map[String, IndexedSeq[Option[Double]]] = {
    val (covNames, sampleCovs): (Array[String], Map[String, Array[Option[Double]]]) =
      state.hadoopConf.readLines(covFile) { lines =>
        if (lines.isEmpty)
          fatal("empty TSV file")

        val fieldNames = lines.next().value.split("\\t")
        val nFields = fieldNames.size

        (fieldNames.drop(1),
          lines.map{ l =>
            val lineSplit = l.value.split("\\t")
            if (lineSplit.length != nFields)
              fatal(s"expected $nFields fields, but got ${lineSplit.length}")
            (lineSplit(0), lineSplit.drop(1).map(x => if (x == "NA") None else Some(x.toDouble))) // FIXME: add error checking
          }.toMap)
      }

    if (! sampleIds.forall(sampleCovs.keySet(_)))
      throw new RESTFailure("Not all samples in the hard call set are listed in the phenotype data set")

    covNames
      .zipWithIndex
      .map{ case (name, j) => (name, sampleIds.map(s => sampleCovs(s)(j))) }.toMap
  }


  def run(state: State, options: Options): State = {

    val hcs = HardCallSet.read(state.sqlContext, options.hcsFile)
    val covMap = readCovData(state, options.covFile, hcs.sampleIds)

    val service = new T2DService(hcs, covMap, options.defaultMinMAC, options.maxWidth, options.hardLimit)
    val task = BlazeBuilder.bindHttp(options.port, "0.0.0.0")
      .mountService(service.service, "/")
      .run
    task.awaitShutdown()

    state
  }
}