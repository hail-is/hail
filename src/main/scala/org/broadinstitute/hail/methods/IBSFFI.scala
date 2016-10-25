package org.broadinstitute.hail.methods

import com.sun.jna._
import com.sun.jna.ptr._

trait IBSFFI extends Library {
  def ibsMat(result: Array[Long], nSamples: Long, nGenotypePacks: Long, genotypes1: Array[Long], genotypes2: Array[Long])
}

case class IBS (N0: Long, N1: Long, N2: Long) { }

object IBSFFI {

  System.setProperty("jna.library.path", "/Users/dking/projects/hail/src/main/c")
  val lib = Native.loadLibrary("ibs", classOf[IBSFFI]).asInstanceOf[IBSFFI]

  def ibs(nSamples: Int, nGenotypes: Int, gs1: Array[Byte], gs2: Array[Byte]): Array[Long] = {
    val packedLength = nGenotypes / 64
    val length = packedLength + (if (nGenotypes % 64 != 0) 1 else 0)

    def pack(gs: Array[Byte]): Array[Long] = {
      val sampleOrientedGenotypes = new Array[Long](nSamples * packedLength)
      var si = 0
      while (si != nSamples) {
        var vBlocki = 0
        while (vBlocki != packedLength) {
          val k = si + vBlocki*64*nSamples
          sampleOrientedGenotypes(si * length + vBlocki) =
              gs(k)                 << 62 | gs(k + 1 * nSamples)  << 60 | gs(k + 2 * nSamples)  << 58 | gs(k + 3 * nSamples)  << 56 |
              gs(k + 4 * nSamples)  << 54 | gs(k + 5 * nSamples)  << 52 | gs(k + 6 * nSamples)  << 50 | gs(k + 7 * nSamples)  << 48 |
              gs(k + 8 * nSamples)  << 46 | gs(k + 9 * nSamples)  << 44 | gs(k + 10 * nSamples) << 42 | gs(k + 11 * nSamples) << 40 |
              gs(k + 12 * nSamples) << 38 | gs(k + 13 * nSamples) << 36 | gs(k + 14 * nSamples) << 34 | gs(k + 15 * nSamples) << 32 |
              gs(k + 16 * nSamples) << 30 | gs(k + 17 * nSamples) << 28 | gs(k + 18 * nSamples) << 26 | gs(k + 19 * nSamples) << 24 |
              gs(k + 20 * nSamples) << 22 | gs(k + 21 * nSamples) << 20 | gs(k + 22 * nSamples) << 18 | gs(k + 23 * nSamples) << 16 |
              gs(k + 24 * nSamples) << 14 | gs(k + 25 * nSamples) << 12 | gs(k + 26 * nSamples) << 10 | gs(k + 27 * nSamples) << 8  |
              gs(k + 28 * nSamples) << 6  | gs(k + 29 * nSamples) << 4  | gs(k + 30 * nSamples) << 2  | gs(k + 31 * nSamples)

          vBlocki += 1
        }
        var vStragglersi = vBlocki*64
        var shift = 62
        while (vStragglersi != length*64) {
          val k = si + vStragglersi*nSamples
          val gt = if (vStragglersi < nGenotypes) gs(k) else 3L
          sampleOrientedGenotypes(si * length + packedLength) |= gt << shift
          vStragglersi += 1
          shift -= 2
        }
        si += 1
      }
      sampleOrientedGenotypes
    }

    val ibs = new Array[Long](nSamples * nGenotypes * 3)
    lib.ibsMat(ibs, nSamples, length, pack(gs1), pack(gs2))
    ibs
  }

}
