package org.broadinstitute.hail.stats

import org.broadinstitute.hail.SparkSuite
import org.broadinstitute.hail.driver._
import org.broadinstitute.hail.utils._
import org.testng.annotations.Test

class InfoScoreSuite extends SparkSuite {
  @Test def test() {
    val genFile = "src/test/resources/infoScoreTest.gen"
    val sampleFile = "src/test/resources/infoScoreTest.sample"
    val truthResultFile = "src/test/resources/infoScoreTest.result"

    var s = State(sc, sqlContext)
    s = ImportGEN.run(s, Array("-s", sampleFile, genFile))
    s = AnnotateVariantsExpr.run(s, Array("-c", """va.infoScore = gs.infoScore()"""))

    val truthResult = hadoopConf.readLines(truthResultFile)(_.map(_.map { line =>
      val Array(v, snpid, rsid, infoScore, nIncluded) = line.trim.split("\\s+")
      val info = infoScore match {
        case "None" => None
        case x => Some(x.toDouble)
      }

      (v, (info, Option(nIncluded.toInt)))
    }.value
    ).toMap)

    val (_, infoQuerier) = s.vds.queryVA("va.infoScore.score")
    val (_, nQuerier) = s.vds.queryVA("va.infoScore.nIncluded")

    val hailResult = s.vds.rdd.mapValues { case (va, gs) =>
      (infoQuerier(va).map(_.asInstanceOf[Double]), nQuerier(va).map(_.asInstanceOf[Int]))
    }
      .map { case (v, (info, n)) => (v.toString, (info, n)) }.collectAsMap()

    (truthResult.keys ++ hailResult.keys).forall { v =>
      val (tI, tN) = truthResult.getOrElse(v, (None, None))
      val (hI, hN) = hailResult.getOrElse(v, (None, None))

      val res =
        tN == hN && ((tI.isEmpty && hI.isEmpty) || D_==(tI.get, hI.get))

      if (!res)
        println(s"v=$v tI=$tI tN=$tN hI=$hI hN=$hN")

      res
    }
  }
}
