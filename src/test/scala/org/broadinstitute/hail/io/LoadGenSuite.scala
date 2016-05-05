package org.broadinstitute.hail.io

import org.apache.spark.rdd.RDD
import org.broadinstitute.hail.SparkSuite
import org.broadinstitute.hail.annotations.Annotation
import org.broadinstitute.hail.driver.{ImportGEN, State}
import org.testng.annotations.Test
import org.broadinstitute.hail.variant._

class LoadGenSuite extends SparkSuite {

  def makeRDD(genFile: String): RDD[(String, Array[Double])] = {
    sc.textFile(genFile).map{case line =>
      val arr = line.split("\\s+")
      val variant = Variant(arr(0), arr(3).toInt, arr(4), arr(5))
      val annotations = Annotation(arr(2), arr(1)) //rsid, varid
      val dosages = arr.drop(6).map {_.toDouble}

      (arr(1), dosages)
    }
  }

  @Test def testGavinExample() {
    val gen = "src/test/resources/example.gen"
    val sampleFile = "src/test/resources/example.sample"

    var s = State(sc, sqlContext, null)

    val genVDS = ImportGEN.run(s, Array("-s",sampleFile, gen)).vds
    val genOrigData = makeRDD(gen)

    val genQuery = genVDS.vaSignature.query("varid")
    val newRDD: RDD[(String, Iterable[Genotype])] = genVDS.rdd.map { case (v, va, gs) => (genQuery(va).get.toString, gs) }

    val res = newRDD.fullOuterJoin(genOrigData).map { case (v, (gs, dos)) =>
      val gs1 = gs.get
      val dosOld = dos.get

      require(gs1.size == dosOld.length / 3)

      var n = 0
      val result = for {g <- gs1} yield {
        val dosNew = g.dosage

        dosNew match {
          case Some(x) =>
            val agreeAA = math.abs(dosNew.get(0) - dosOld(n)) <= 3.0e-4
            val agreeAB = math.abs(dosNew.get(1) - dosOld(n + 1)) <= 3.0e-4
            val agreeBB = math.abs(dosNew.get(2) - dosOld(n + 2)) <= 3.0e-4

            n += 3

            agreeAA && agreeAB && agreeBB

          case None =>
            val res = if (dosOld(n) == 0.0 && dosOld(n+1) == 0.0 && dosOld(n+2) == 0.0) //FIXME: Is this correct?
              true
            else {
              println(s"${dosOld(n)} ${dosOld(n+1)} ${dosOld(n+2)}")
              false
            }
            n += 3
            res
        }
      }
      result.forall(p => p)
    }.fold(true)(_ && _)
    assert(res)
  }
}
