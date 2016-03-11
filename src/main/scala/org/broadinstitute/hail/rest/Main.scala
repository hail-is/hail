package org.broadinstitute.hail.rest

import org.broadinstitute.hail.methods.CovariateData
import org.broadinstitute.hail.variant.HardCallSet
import org.http4s.server.blaze.BlazeBuilder

object Conf {
  var dataRoot: String = _

  def epactsFile = dataRoot + "/GoT2D.epacts.bgz"

  def pValuesFile = dataRoot + "/GoT2D.pvalues.parquet"

  def hardCallSetFile = dataRoot + "GoT2D.chr22.hcs"

  def covariateDataFile = dataRoot + "GoT2D.noNA.cov"
}

// extends App
object Main {

  def fatal(msg: String): Nothing = {
    println(msg)
    sys.exit(1)
  }

  def main(args: Array[String]) {
    if (args.length != 2)
      fatal("usage: hail-t2d-api <GoT2D path> [run/runhcs/import]")

    Conf.dataRoot = args(0)
    val command = args(1)

    if (command == "run") {
      // FIXME result gave unexpected end of file
      // val compressed = middleware.GZip(T2DService.service)

      println("loading p-values...")
      GoT2D.results.count()
      println("loading p-values done.")

      val task = BlazeBuilder.bindHttp(6060)
        .mountService(T2DService.service, "/")
        .run
      T2DService.task = task
      task.awaitShutdown()
    }
    else if (command == "runhcs") {

      println("loading hcs...")
      val hcs = HardCallSet.read(SparkStuff.sqlContext, Conf.hardCallSetFile).cache()
      println("loading hcs done.")

      hcs.nVariants

      println("loading covariate data...")
      val cov = CovariateData.read(Conf.covariateDataFile, SparkStuff.hadoopConf, hcs.sampleIds)
      println("loading covariate data done.")

      val task = BlazeBuilder.bindHttp(6060)
        .mountService(T2DService.serviceHcs(hcs, cov), "/")
        .run
      T2DService.task = task
      task.awaitShutdown()
    }
    else if (command == "import") {
      GoT2D.importPValues()
    }
    else
      fatal(s"unknown command `$command'")
  }
}
