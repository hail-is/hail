package is.hail.io

import is.hail.SparkSuite
import is.hail.annotations.Annotation
import is.hail.utils._
import is.hail.testUtils._
import is.hail.variant._
import org.apache.spark.rdd.RDD
import org.testng.annotations.Test

class ImportGenSuite extends SparkSuite {

  def makeRDD(genFile: String): RDD[(String, Array[Double])] = {
    sc.textFile(genFile).map { line =>
      val arr = line.split("\\s+")
      val variant = Variant(arr(0), arr(3).toInt, arr(4), arr(5))
      val annotations = Annotation(arr(2), arr(1))
      //rsid, varid
      val gp = arr.drop(6).map {
        _.toDouble
      }

      (arr(1), gp)
    }
  }

  @Test def testGavinExample() {
    val gen = "src/test/resources/example.gen"
    val sampleFile = "src/test/resources/example.sample"

    val genVDS = hc.importGen(gen, sampleFile, contigRecoding = Some(Map("01" -> "1")))
    val genOrigData = makeRDD(gen)

    val genQuery = genVDS.rowType.query("varid")
    val newRDD: RDD[(String, Iterable[Annotation])] = genVDS.rdd.map { case (v, (va, gs)) => (genQuery(va).toString, gs) }

    val (_, qGP) = genVDS.queryGA("g.GP")
    val res = newRDD.fullOuterJoin(genOrigData).forall { case (v, (gs, dos)) =>
      val gs1 = gs.get
      val dosOld = dos.get

      require(gs1.size == dosOld.length / 3)

      var n = 0
      val result = for {g <- gs1} yield {
        val dosNew = Option(qGP(g).asInstanceOf[IndexedSeq[Double]])

        dosNew match {
          case Some(x) =>
            val agreeAA = math.abs(dosNew.get(0) - dosOld(n)) <= 3.0e-4
            val agreeAB = math.abs(dosNew.get(1) - dosOld(n + 1)) <= 3.0e-4
            val agreeBB = math.abs(dosNew.get(2) - dosOld(n + 2)) <= 3.0e-4

            n += 3

            agreeAA && agreeAB && agreeBB

          case None =>
            val res = if (dosOld(n) == 0.0 && dosOld(n + 1) == 0.0 && dosOld(n + 2) == 0.0) //FIXME: Is this correct?
              true
            else {
              println(s"${ dosOld(n) } ${ dosOld(n + 1) } ${ dosOld(n + 2) }")
              false
            }
            n += 3
            res
        }
      }
      result.forall(p => p)
    }
    assert(res)
  }
}
