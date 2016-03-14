package org.broadinstitute.hail.rest

import org.broadinstitute.hail.methods.CovariateData
import org.broadinstitute.hail.variant.HardCallSet
import org.http4s.server.blaze.BlazeBuilder

object Conf {
  var dataRoot: String = _

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
    if (args.length != 1)
      fatal("usage: hail-t2d-api <GoT2D path>")

    Conf.dataRoot = args(0)

    println("loading hcs...")
    val hcs = HardCallSet.read(SparkStuff.sqlContext, Conf.hardCallSetFile).cache()
    println("loading hcs done.")

    hcs.nVariants

    println("loading covariate data...")
    val cov = CovariateData.read(Conf.covariateDataFile, SparkStuff.hadoopConf, hcs.sampleIds)
    println("loading covariate data done.")

    val task = BlazeBuilder.bindHttp(6060)
      .mountService(T2DService.service(hcs, cov), "/")
      .run
    T2DService.task = task
    task.awaitShutdown()
  }


  {
    import breeze.linalg._
    import breeze.linalg.operators.{OpMulMatrix, BinaryRegistry}

    implicitly[BinaryRegistry[DenseMatrix[Double], Vector[Double], OpMulMatrix.type, DenseVector[Double]]].register(
      DenseMatrix.implOpMulMatrix_DMD_DVD_eq_DVD)
  }

}
