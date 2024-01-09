package is.hail.methods

import com.sun.jna._

case class IBS(N0: Long, N1: Long, N2: Long) {}

object IBSFFI {

  val gtToCRep = Array[Byte](0, 1, 3)
  val missingGTCRep: Byte = 2

  @native
  def ibsMat(
    result: Array[Long],
    nSamples: Long,
    nPacks: Long,
    genotypes1: Array[Long],
    genotypes2: Array[Long],
  )

  // NativeCode needs to control the initial loading of the libhail DLL, and
  // the call to getHailName() guarantees that.
  Native.register("hail")

  val genotypesPerPack = 32

  def pack(nSamples: Int, nGenotypes: Int, gs: Array[Byte]): Array[Long] = {
    require(nGenotypes % 32 == 0)

    val nPacks = nGenotypes / genotypesPerPack

    val sampleOrientedGenotypes = new Array[Long](nSamples * nPacks)
    var si = 0
    while (si != nSamples) {
      var pack = 0
      while (pack != nPacks) {
        val k = si + pack * genotypesPerPack * nSamples
        sampleOrientedGenotypes(si * nPacks + pack) =
          gs(k).toLong << 62 | gs(k + 1 * nSamples).toLong << 60 | gs(
            k + 2 * nSamples
          ).toLong << 58 | gs(k + 3 * nSamples).toLong << 56 |
            gs(k + 4 * nSamples).toLong << 54 | gs(k + 5 * nSamples).toLong << 52 | gs(
              k + 6 * nSamples
            ).toLong << 50 | gs(k + 7 * nSamples).toLong << 48 |
            gs(k + 8 * nSamples).toLong << 46 | gs(k + 9 * nSamples).toLong << 44 | gs(
              k + 10 * nSamples
            ).toLong << 42 | gs(k + 11 * nSamples).toLong << 40 |
            gs(k + 12 * nSamples).toLong << 38 | gs(k + 13 * nSamples).toLong << 36 | gs(
              k + 14 * nSamples
            ).toLong << 34 | gs(k + 15 * nSamples).toLong << 32 |
            gs(k + 16 * nSamples).toLong << 30 | gs(k + 17 * nSamples).toLong << 28 | gs(
              k + 18 * nSamples
            ).toLong << 26 | gs(k + 19 * nSamples).toLong << 24 |
            gs(k + 20 * nSamples).toLong << 22 | gs(k + 21 * nSamples).toLong << 20 | gs(
              k + 22 * nSamples
            ).toLong << 18 | gs(k + 23 * nSamples).toLong << 16 |
            gs(k + 24 * nSamples).toLong << 14 | gs(k + 25 * nSamples).toLong << 12 | gs(
              k + 26 * nSamples
            ).toLong << 10 | gs(k + 27 * nSamples).toLong << 8 |
            gs(k + 28 * nSamples).toLong << 6 | gs(k + 29 * nSamples).toLong << 4 | gs(
              k + 30 * nSamples
            ).toLong << 2 | gs(k + 31 * nSamples).toLong

        pack += 1
      }
      si += 1
    }
    sampleOrientedGenotypes
  }

  def ibs(nSamples: Int, nGenotypes: Int, gs1: Array[Long], gs2: Array[Long]): Array[Long] = {
    require(nGenotypes % 32 == 0)

    val nPacks = nGenotypes / genotypesPerPack
    val ibs = new Array[Long](nSamples * nSamples * 3)
    ibsMat(ibs, nSamples, nPacks, gs1, gs2)
    ibs
  }

}
