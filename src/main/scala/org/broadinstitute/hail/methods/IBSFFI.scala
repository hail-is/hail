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
    val packedLength = nGenotypes / 32
    val length = packedLength + (if (nGenotypes % 32 != 0) 1 else 0)

    def pack(gs: Array[Byte]): Array[Long] = {
      val sampleOrientedGenotypes = new Array[Long](nSamples * length)
      var si = 0
      while (si != nSamples) {
        var vBlocki = 0
        while (vBlocki != packedLength) {
          val k = si + vBlocki*32*nSamples
          sampleOrientedGenotypes(si * length + vBlocki) =
              gs(k).toLong                 << 62 | gs(k + 1 * nSamples).toLong  << 60 | gs(k + 2 * nSamples).toLong  << 58 | gs(k + 3 * nSamples).toLong  << 56 |
              gs(k + 4 * nSamples).toLong  << 54 | gs(k + 5 * nSamples).toLong  << 52 | gs(k + 6 * nSamples).toLong  << 50 | gs(k + 7 * nSamples).toLong  << 48 |
              gs(k + 8 * nSamples).toLong  << 46 | gs(k + 9 * nSamples).toLong  << 44 | gs(k + 10 * nSamples).toLong << 42 | gs(k + 11 * nSamples).toLong << 40 |
              gs(k + 12 * nSamples).toLong << 38 | gs(k + 13 * nSamples).toLong << 36 | gs(k + 14 * nSamples).toLong << 34 | gs(k + 15 * nSamples).toLong << 32 |
              gs(k + 16 * nSamples).toLong << 30 | gs(k + 17 * nSamples).toLong << 28 | gs(k + 18 * nSamples).toLong << 26 | gs(k + 19 * nSamples).toLong << 24 |
              gs(k + 20 * nSamples).toLong << 22 | gs(k + 21 * nSamples).toLong << 20 | gs(k + 22 * nSamples).toLong << 18 | gs(k + 23 * nSamples).toLong << 16 |
              gs(k + 24 * nSamples).toLong << 14 | gs(k + 25 * nSamples).toLong << 12 | gs(k + 26 * nSamples).toLong << 10 | gs(k + 27 * nSamples).toLong << 8  |
              gs(k + 28 * nSamples).toLong << 6  | gs(k + 29 * nSamples).toLong << 4  | gs(k + 30 * nSamples).toLong << 2  | gs(k + 31 * nSamples).toLong

          vBlocki += 1
        }
        var vStragglersi = vBlocki*32
        var shift = 62
        while (vStragglersi != length*32) {
          val k = si + vStragglersi*nSamples
          val gt = if (vStragglersi < nGenotypes) gs(k).toLong else 2L
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
