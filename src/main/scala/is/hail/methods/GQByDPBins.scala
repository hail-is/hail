package is.hail.methods

import is.hail.annotations.Annotation
import is.hail.variant.VariantDataset

object GQByDPBins {
  val nBins = 14
  val firstBinLow = 5
  val binStep = 5
  val gqThreshold = 20

  def dpBin(dp: Int): Option[Int] = {
    if (dp < firstBinLow)
      None
    else {
      val b = (dp - firstBinLow) / binStep
      if (b < nBins)
        Some(b)
      else
        None
    }
  }

  def binLow(b: Int): Int = {
    require(b >= 0 && b < nBins)
    firstBinLow + b * binStep
  }

  def binHigh(b: Int): Int = {
    firstBinLow + (b + 1) * binStep - 1
  }

  // ((sample, bin), %GQ)
  def apply(vds: VariantDataset): Map[(Annotation, Int), Double] = {
    vds
      .flatMapWithKeys((v, s, g) => {
        val bin = g.dp.flatMap(dpBin)
        if (bin.isDefined && g.isCalled)
          Some(((s, bin.get), (if (g.gq.exists(_ >= gqThreshold)) 1 else 0, 1)))
        else
          None
      })
      .foldByKey((0, 0))((a, b) => (a._1 + b._1, a._2 + b._2))
      .mapValues(a => {
        assert(a._2 != 0)
        a._1.toDouble / a._2
      })
      .collectAsMap().toMap
  }
}

