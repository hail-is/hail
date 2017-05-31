package is.hail.io.bgen

import is.hail.annotations._
import is.hail.utils._
import is.hail.variant.{Genotype, Variant, VariantDataset}

object BgenWriter {
  val ploidy: Byte = 2
  val phased: Byte = 0

  def shortToBytesLE(bb: ArrayBuilder[Byte], i: Int) {
    bb += (i & 0xff).toByte
    bb += ((i >>> 8) & 0xff).toByte
  }

  def intToBytesLE(bb: ArrayBuilder[Byte], i: Int) {
    bb += (i & 0xff).toByte
    bb += ((i >>> 8) & 0xff).toByte
    bb += ((i >>> 16) & 0xff).toByte
    bb += ((i >>> 24) & 0xff).toByte
  }

  def updateIntToBytesLE(bb: ArrayBuilder[Byte], i: Int, pos: Int) {
    bb(pos) = (i & 0xff).toByte
    bb(pos + 1) = ((i >>> 8) & 0xff).toByte
    bb(pos + 2) = ((i >>> 16) & 0xff).toByte
    bb(pos + 3) = ((i >>> 24) & 0xff).toByte
  }

  def stringToBytesWShortLength(bb: ArrayBuilder[Byte], s: String): Int = {
    val bytes = s.getBytes
    val l = bytes.length
    shortToBytesLE(bb, l)
    bb ++= bytes
    2 + l
  }

  def stringToBytesWIntLength(bb: ArrayBuilder[Byte], s: String): Int = {
    val bytes = s.getBytes
    val l = bytes.length
    intToBytesLE(bb, l)
    bb ++= bytes
    4 + l
  }

  def bitPack(bb: ArrayBuilder[Byte], input: Array[Int], nBitsPerProb: Int) {
    require(nBitsPerProb >= 1 && nBitsPerProb <= 32, "Number of bits per probability must be between 1 and 32 inclusive.")

    val bitMask = (1L << nBitsPerProb) - 1
    var byteIndex = 0
    var data = 0L
    var dataSize = 0

    var i = 0
    while (i < input.length) {
      data |= ((input(i).toUIntFromRep.toLong & bitMask) << dataSize)
      dataSize += nBitsPerProb

      while (dataSize >= 8) {
        bb += data.toByte
        data = data >>> 8
        dataSize -= 8
        byteIndex += 1
      }
      i += 1
    }

    if (dataSize > 0)
      bb += data.toByte
  }

  def sortedIndex(a: Array[Double], idx: Array[Int]) {
    val n = a.length
    if (n == 3) {
      if (a(0) >= a(1) && a(1) >= a(2)) {
        idx(0) = 0
        idx(1) = 1
        idx(2) = 2
      } else if (a(0) >= a(2) && a(2) >= a(1)) {
        idx(0) = 0
        idx(1) = 2
        idx(2) = 1
      } else if (a(1) >= a(0) && a(0) >= a(2)) {
        idx(0) = 1
        idx(1) = 0
        idx(2) = 2
      } else if (a(1) >= a(2) && a(2) >= a(0)) {
        idx(0) = 2
        idx(1) = 0
        idx(2) = 1
      } else if (a(2) >= a(0) && a(0) >= a(1)) {
        idx(0) = 1
        idx(1) = 2
        idx(2) = 0
      } else {
        idx(0) = 2
        idx(1) = 1
        idx(2) = 0
      }
    } else {
      throw new UnsupportedOperationException("sortedIndex is only supported for arrays of length 3.")
    }
  }

  private def resetIndex(index: Array[Int]) {
    var i = 0
    while (i < index.length) {
      index(i) = i
      i += 1
    }
  }

  def resizeWeights(input: Array[Double], output: ArrayBuilder[Int], resized: Array[Double], index: Array[Int], size: Long) {
    val n = input.length
    assert(resized.length == n && index.length == n)
    resetIndex(index)

    val conversionFactor = size.toDouble / input.sum

    var totalFractional = 0d
    var i = 0
    while (i < n) {
      val x = input(i) * conversionFactor
      resized(i) = x
      totalFractional += (x - x.floor)
      i += 1
    }

    val F = (totalFractional + 0.5).toInt
    assert(F >= 0 && F <= n)

    sortedIndex(resized, index)

    i = 0
    var newSize = 0d
    while (i < n) {
      val r = if (index(i) < F) resized(i).ceil else resized(i).floor
      if (i != n - 1)
        output += r.toUInt.intRep
      newSize += r
      i += 1
    }

    assert(newSize == size)
  }

  def resizeWeights(input: Array[Int], output: ArrayBuilder[Int], resized: Array[Double], index: Array[Int], size: Long) {
    val n = input.length
    assert(resized.length == n && index.length == n)
    resetIndex(index)

    val conversionFactor = size.toDouble / input.sum

    var totalFractional = 0d
    var i = 0
    while (i < n) {
      val x = input(i).toUIntFromRep * conversionFactor
      resized(i) = x
      totalFractional += (x - x.floor)
      i += 1
    }

    val F = (totalFractional + 0.5).toInt
    assert(F >= 0 && F <= n)

    sortedIndex(resized, index)

    i = 0
    var newSize = 0d
    while (i < n) {
      val r = if (index(i) < F) resized(i).ceil else resized(i).floor
      if (i != n - 1)
        output += r.toUInt.intRep
      newSize += r
      i += 1
    }

    assert(newSize == size)
  }

  def emitHeaderBlock(vds: VariantDataset): Array[Byte] = {
    val nSamples = vds.nSamples
    val sampleIds = vds.stringSampleIds.toArray
    val nVariants = vds.countVariants()
    assert(nVariants <= (1L << 32) - 1, s"Maximum number of variants can export is (2^32 - 1). Found $nVariants.")

    val magicNumbers = Array("b", "g", "e", "n").flatMap(_.getBytes)
    val flags = 0x01 | (0x02 << 2) | (0x01 << 31)
    val headerLength = 20

    val bb = new ArrayBuilder[Byte]

    intToBytesLE(bb, 0) // placeholder for offset
    intToBytesLE(bb, headerLength)
    intToBytesLE(bb, nVariants.toUInt.intRep)
    intToBytesLE(bb, nSamples)
    bb ++= magicNumbers
    intToBytesLE(bb, flags)

    intToBytesLE(bb, 0) // placeholder for length of sample ids
    intToBytesLE(bb, nSamples)

    var i = 0
    var sampleBlockLength = 8
    while (i < nSamples) {
      sampleBlockLength += stringToBytesWShortLength(bb, sampleIds(i))
      i += 1
    }

    val offset = headerLength + sampleBlockLength
    updateIntToBytesLE(bb, offset, 0)
    updateIntToBytesLE(bb, sampleBlockLength, 24)

    bb.result()
  }

  def emitVariantBlock(v: Variant, va: Annotation, gs: Iterable[Genotype],
    rsidQuery: Option[Querier], varidQuery: Option[Querier], nSamples: Int, nBitsPerProb: Int): Array[Byte] = {

    val bb = new ArrayBuilder[Byte]
    val nAlleles = v.nAlleles

    require(nAlleles <= 0xffff, s"Maximum number of alleles per variant is ${ 0xffff }. Found ${ v.nAlleles }.")

    val varid = varidQuery.flatMap(q => Option(q(va))).map(_.asInstanceOf[String]).getOrElse(v.toString)
    val rsid = rsidQuery.flatMap(q => Option(q(va))).map(_.asInstanceOf[String]).getOrElse(".")

    stringToBytesWShortLength(bb, varid)
    stringToBytesWShortLength(bb, rsid)
    stringToBytesWShortLength(bb, v.contig)
    intToBytesLE(bb, v.start)
    shortToBytesLE(bb, nAlleles)
    stringToBytesWIntLength(bb, v.ref)
    v.altAlleles.foreach(a => stringToBytesWIntLength(bb, a.alt))

    val gtDataBlockStart = bb.length
    intToBytesLE(bb, 0) // placeholder for length of compressed data
    intToBytesLE(bb, 0) // placeholder for length of uncompressed data

    val uncompressedData = emitGenotypeData(gs, nSamples, nAlleles, nBitsPerProb)

    val uncompressedLength = uncompressedData.length
    val compressedLength = compress(bb, uncompressedData)

    updateIntToBytesLE(bb, compressedLength + 4, gtDataBlockStart)
    updateIntToBytesLE(bb, uncompressedLength, gtDataBlockStart + 4)

    bb.result()
  }

  def emitGenotypeData(gs: Iterable[Genotype], nSamples: Int, nAlleles: Int, nBitsPerProb: Int): Array[Byte] = {
    val gtDataBuffer = new ArrayBuilder[Byte]
    val sampleProbs = new ArrayBuilder[Int]

    val nGenotypes = triangle(nAlleles)
    val newSize = (1L << nBitsPerProb) - 1

    intToBytesLE(gtDataBuffer, nSamples)
    shortToBytesLE(gtDataBuffer, nAlleles)
    gtDataBuffer += ploidy
    gtDataBuffer += ploidy

    val resized = new Array[Double](nGenotypes)
    val index = new Array[Int](nGenotypes)

    gs.foreach { gt =>
      val px = gt.unboxedPX
      if (px == null) {
        gtDataBuffer += 0x80.toByte
        var i = 0
        while (i < nGenotypes - 1) {
          sampleProbs += 0
          i += 1
        }
      } else {
        gtDataBuffer += ploidy
        if (gt.isLinearScale)
          resizeWeights(px, sampleProbs, resized, index, newSize)
        else
          resizeWeights(gt.unboxedGP, sampleProbs, resized, index, newSize)
      }
    }

    gtDataBuffer += phased
    gtDataBuffer += nBitsPerProb.toByte
    bitPack(gtDataBuffer, sampleProbs.result(), nBitsPerProb)

    gtDataBuffer.result()
  }
}
