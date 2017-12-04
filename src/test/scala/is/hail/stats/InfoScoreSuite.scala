package is.hail.stats

import is.hail.SparkSuite
import is.hail.utils._
import org.testng.annotations.Test

class InfoScoreSuite extends SparkSuite {
  @Test def test() {
    val genFile = "src/test/resources/infoScoreTest.gen"
    val sampleFile = "src/test/resources/infoScoreTest.sample"
    val truthResultFile = "src/test/resources/infoScoreTest.result"

    val vds = hc.importGen(genFile, sampleFile)
      .annotateVariantsExpr("""va.infoScore = gs.map(g => g.GP).infoScore()""")

    val truthResult = hadoopConf.readLines(truthResultFile)(_.map(_.map { line =>
      val Array(v, snpid, rsid, infoScore, nIncluded) = line.trim.split("\\s+")
      val info = infoScore match {
        case "None" => None
        case x => Some(x.toDouble)
      }

      (v, (info, Option(nIncluded.toInt)))
    }.value
    ).toMap)

    val (_, infoQuerier) = vds.queryVA("va.infoScore.score")
    val (_, nQuerier) = vds.queryVA("va.infoScore.nIncluded")

    val hailResult = vds.rdd.mapValues { case (va, gs) =>
      (Option(infoQuerier(va)).map(_.asInstanceOf[Double]), Option(nQuerier(va)).map(_.asInstanceOf[Int]))
    }
      .map { case (v, (info, n)) => (v.toString, (info, n)) }.collectAsMap()

    assert((truthResult.keys ++ hailResult.keys).forall { v =>
      val (tI, tN) = truthResult.getOrElse(v, (None, None))
      val (hI, hN) = hailResult.getOrElse(v, (None, None))

      val res =
        tN == hN && ((tI.isEmpty && hI.isEmpty) || math.abs(tI.get - hI.get) < 1e-3)

      if (!res)
        println(s"v=$v tI=$tI tN=$tN hI=$hI hN=$hN")

      res
    })
  }
}
