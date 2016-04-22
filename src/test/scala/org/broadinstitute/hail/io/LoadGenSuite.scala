package org.broadinstitute.hail.io

import org.apache.hadoop.conf.Configuration
import org.apache.spark.rdd.{PairRDDFunctions, RDD}
import org.broadinstitute.hail.SparkSuite
import org.broadinstitute.hail.Utils._
import org.broadinstitute.hail.annotations.Annotation
import org.broadinstitute.hail.driver.{ImportBGEN, IndexBGEN, State}
import org.testng.annotations.Test
import org.broadinstitute.hail.io.gen._
import org.broadinstitute.hail.variant._

class LoadGenSuite extends SparkSuite {

  def makeRDD(genFile: String): RDD[(Variant, Array[Double])] = {
    sc.textFile(genFile).map{case line =>
      val arr = line.split("\\s+")
      val variant = Variant(arr(0), arr(3).toInt, arr(4), arr(5))
      val annotations = Annotation(arr(2), arr(1)) //rsid, varid
      val dosages = arr.drop(6).map {_.toDouble}

      (variant, dosages)
    }
  }
/*
  @Test def hardCodedTest() {
    val probs: Map[Array[Double],Array[Double]] = (Array(0.984009, 3.35693E-4, 0.015686) -> Array())
  }*/

  @Test def testGavinExample() {
    val gen = "src/test/resources/example.gen"
    val sampleFile = "src/test/resources/example.sample"

    var s = State(sc, sqlContext, null)
    val genVDS = GenLoader2(gen, sampleFile, sc)
    val genOrigData = makeRDD(gen)

    val newRDD: RDD[(Variant, Iterable[Genotype])] = genVDS.rdd.map { case (v, va, gs) => (v, gs) }
    //val oldRDD: RDD[((Variant, Annotation), Array[Double])] = genOrigData.map{case (v, va, dos) => ((v,va),dos)}


    val res = newRDD.fullOuterJoin(genOrigData).map { case (v, (gs, dos)) =>
      val gs1 = gs.get
      val dosOld = dos.get
      var n = 0
      val result = for {g <- gs1} yield {

        val dosNew = g.dosage

        dosNew match {
          case Some(x) =>
            val agreeAA = D_==(dosNew.get(0), dosOld(n))
            val agreeAB = D_==(dosNew.get(1), dosOld(n + 1))
            val agreeBB = D_==(dosNew.get(2), dosOld(n + 2))
            n += 3
            if (!(agreeAA && agreeAB && agreeBB)) {
              println(s"v=$v n=${n/3} origAA=${dosOld(n)} origAB=${dosOld(n + 1)} origBB=${dosOld(n + 2)}")
              println(s"v=$v n=${n/3} newAA=${dosNew.get(0)} newAB=${dosNew.get(1)} newBB=${dosNew.get(2)}")
              println(s"v=$v n=${n/3} newAA=${g.pl.get(0)} newAB=${g.pl.get(1)} newBB=${g.pl.get(2)}")
            }
            agreeAA && agreeAB && agreeBB
          case None => true
        }

      }
      result.fold(true)(_ && _)
    }.fold(true)(_ && _)
    assert(res)
  }
}
