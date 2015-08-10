package org.broadinstitute.k3.methods

import org.broadinstitute.k3.variant._

import scala.collection.Map

object nTransversionPerSample { // should compute with transitions
  def apply(vds: VariantDataset): Map[Int, Int] = {
    vds
      .aggregateBySampleWithKeys(0)((n, v, s, g) => if ((g.isHet || g.isHomVar) && v.isTransversion) n+1 else n, _ + _)
  }
}



