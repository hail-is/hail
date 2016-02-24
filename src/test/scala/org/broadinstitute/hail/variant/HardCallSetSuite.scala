package org.broadinstitute.hail.variant

import breeze.linalg.DenseVector
import org.broadinstitute.hail.SparkSuite
import org.broadinstitute.hail.methods.{CovariateData, Pedigree, LoadVCF}
import org.testng.annotations.Test

import scala.util.Random

class HardCallSetSuite extends SparkSuite {
  @Test def test() {

    val gtsList = List(
      //Iterable(),
      Iterable(0),
      Iterable(0,1),
      Iterable(0,1,2),
      Iterable(0,1,2,3),
      Iterable(0,1,2,3,0),
      Iterable(0,1,2,3,1),
      Iterable(0,1,2,3,1,2),
      Iterable(0,1,2,3,0,0,0,0,0,0,0,0,0,0,1,2,3),
      Iterable.fill[Int](1001)(1)
    )



    for (gts <- gtsList) {
      val n = gts.size
      val y = DenseVector.fill[Double](n)(Random.nextInt(2))

      val dcs = DenseCallStream.DenseCallStreamFromGtStream(gts, n)

      println(dcs)
      // dcs.showBinary()

      val scs = SparseCallStream.SparseCallStreamFromGtStream(gts, n)
      println(scs)

      val ds = dcs.hardStats(y,n)

      val ss = scs.hardStats(y,n)

      println(ds)

      println(ss)

      assert(ds == ss)

      println()
    }

    /*
    val vds = LoadVCF(sc, "src/test/resources/linearRegression.vcf")
    //val ped = Pedigree.read("src/test/resources/linearRegression.fam", sc.hadoopConfiguration, vds.sampleIds)
    //val cov = CovariateData.read("src/test/resources/linearRegression.cov", sc.hadoopConfiguration, vds.sampleIds)

    val hcs = HardCallSet(vds)
    println(hcs.sampleIds)
    hcs.rdd.foreach(println)

    hcs.write(sqlContext, "/tmp/hardcalls.hcs")
    val hcs2 = HardCallSet.read(sqlContext, "/tmp/hardcalls.hcs")

    println(hcs2.sampleIds)
    def toComp(hcs: HardCallSet) = hcs.rdd.mapValues(_.a.toList).collect().toSet
    assert(toComp(hcs) == toComp(hcs2))

   */
  }


}
