package org.broadinstitute.hail.methods

import org.broadinstitute.hail.Utils._
import org.broadinstitute.hail.driver._
import org.testng.annotations.Test
import org.broadinstitute.hail.SparkSuite

class InfoScoreSuite extends SparkSuite {
  @Test def test() {
    val genFile = "src/test/resources/infoScoreTest.gen"
    val sampleFile = "src/test/resources/infoScoreTest.sample"
    val truthResultFile = "src/test/resources/infoScoreTest.result"

    var s = State(sc, sqlContext)
    s = ImportGEN.run(s, Array("-s", sampleFile, genFile))
    s = InfoScore.run(s, Array.empty[String])

    val truthResult = sc.parallelize(readLines(truthResultFile, sc.hadoopConfiguration)(_.map(_.map { line =>
      val Array(v, snpid, rsid, infoScore, nIncluded) = line.trim.split("\\s+")
      val info = infoScore match {
        case "None" => None
        case x => Some(x.toDouble)
      }

      (v, (info, Option(nIncluded.toInt)))
    }.value
    ).toIndexedSeq))

    val (_, infoQuerier) = s.vds.queryVA("va.infoscore.impute")
    val (_, nQuerier) = s.vds.queryVA("va.infoscore.nIncluded")

    val hailResult = s.vds.rdd.mapValues{ case (va, gs) =>
      (infoQuerier(va).map(_.asInstanceOf[Double]), nQuerier(va).map(_.asInstanceOf[Int]))}
      .map{case (v, (info, n)) => (v.toString, (info, n))}

    truthResult.fullOuterJoin(hailResult).forall { case (v, (t, h)) =>
      if (!(t.isDefined && h.isDefined))
        false
      else {
        val (tI, tN) = t.get
        val (hI, hN) = h.get

        val res =
          tN == hN && ((tI.isEmpty && hI.isEmpty) || D_==(tI.get, hI.get))

        if (!res)
          println(s"v=$v tI=$tI tN=$tN hI=$hI hN=$hN")

        res
      }

    }
  }
}
