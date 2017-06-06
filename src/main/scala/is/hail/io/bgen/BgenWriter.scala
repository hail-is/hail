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

  def sortedIndex(a: Array[Double], idx: Array[Int]) {
    val n = a.length
    if (n != 3)
      throw new UnsupportedOperationException("sortedIndex is only supported for arrays of length 3.")

    if (a(0) >= a(1)) {
      if (a(1) >= a(2)) {
        idx(0) = 0
        idx(1) = 1
        idx(2) = 2
      } else {
        if (a(0) >= a(2)) {
          idx(0) = 0
          idx(1) = 2
          idx(2) = 1
        } else {
          idx(0) = 1
          idx(1) = 2
          idx(2) = 0
        }
      }
    } else {
      if (a(0) >= a(2)) {
        idx(0) = 1
        idx(1) = 0
        idx(2) = 2
      } else {
        if (a(1) >= a(2)) {
          idx(0) = 2
          idx(1) = 0
          idx(2) = 1
        } else {
          idx(0) = 2
          idx(1) = 1
          idx(2) = 0
        }
      }
    }
  }

  private def resetIndex(index: Array[Int]) {
    var i = 0
    while (i < index.length) {
      index(i) = i
      i += 1
    }
  }

  def resizeAndComputeFractional(input: Array[Int], resized: Array[Double], fractional: Array[Double], conversionFactor: Double): Int = {
    val n = input.length
    assert(resized.length == n)

    var totalFractional = 0d
    var i = 0
    while (i < n) {
      val x = input(i).toUIntFromRep * conversionFactor
      val f = x - x.floor
      resized(i) = x
      fractional(i) = f
      totalFractional += f
      i += 1
    }

    val F = (totalFractional + 0.5).toInt
    assert(F >= 0 && F <= n)
    F
  }

  def resizeAndComputeFractional(input: Array[Double], resized: Array[Double], fractional: Array[Double], conversionFactor: Double): Int = {
    val n = input.length
    assert(resized.length == n)

    var totalFractional = 0d
    var i = 0
    while (i < n) {
      val x = input(i) * conversionFactor
      val f = x - x.floor
      resized(i) = x
      fractional(i) = f
      totalFractional += f
      i += 1
    }

    val F = (totalFractional + 0.5).toInt
    assert(F >= 0 && F <= n)
    F
  }

  def roundWithConstantSum(resized: Array[Double], fractional: Array[Double], index: Array[Int], output: IntConsumer, F: Int, expectedSize: Long, keepLast: Boolean = true) {
    val n = resized.length
    assert(index.length == n)
    resetIndex(index)
    sortedIndex(fractional, index)

    var i = 0
    var newSize = 0d
    while (i < n) {
      val r = if (index(i) < F) resized(i).ceil else resized(i).floor
      if (keepLast || i != n - 1)
        output += r.toUInt.intRep
      newSize += r
      i += 1
    }
    assert(newSize == expectedSize)
  }

  def headerBlock(sampleIds: Array[String], nVariants: Long): Array[Byte] = {
    val bb = new ArrayBuilder[Byte]
    emitHeaderBlock(bb, sampleIds, nVariants)
    bb.result()
  }

  def emitHeaderBlock(bb: ArrayBuilder[Byte], sampleIds: Array[String], nVariants: Long) {
    val nSamples = sampleIds.length
    assert(nVariants <= (1L << 32) - 1, s"Maximum number of variants can export is (2^32 - 1). Found $nVariants.")

    val magicNumbers = Array("b", "g", "e", "n").flatMap(_.getBytes)
    val flags = 0x01 | (0x02 << 2) | (0x01 << 31)
    val headerLength = 20

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
  }

  def emitVariant(bb: ArrayBuilder[Byte], v: Variant, va: Annotation, gs: Iterable[Genotype],
    rsidQuery: Option[Querier], varidQuery: Option[Querier], nSamples: Int, nBitsPerProb: Int) {
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
  }

  def emitGenotypeData(gs: Iterable[Genotype], nSamples: Int, nAlleles: Int, nBitsPerProb: Int): Array[Byte] = {
    val bb = new ArrayBuilder[Byte]

    val nGenotypes = triangle(nAlleles)
    val newSize = (1L << nBitsPerProb) - 1

    intToBytesLE(bb, nSamples)
    shortToBytesLE(bb, nAlleles)
    bb += ploidy
    bb += ploidy

    val resized = new Array[Double](nGenotypes)
    val index = new Array[Int](nGenotypes)
    val fractional = new Array[Double](nGenotypes)

    val samplePloidyStart = bb.length
    var i = 0
    while (i < nSamples) {
      bb += 0x82.toByte // placeholder for sample ploidy - default is missing
      i += 1
    }

    bb += phased
    bb += nBitsPerProb.toByte

    val bitPacker = new BitPacker(bb, nBitsPerProb)
    var idx = 0
    val gsIt = gs.iterator
    while (gsIt.hasNext) {
      val gt = gsIt.next()
      val px = gt.unboxedPX
      if (px == null) {
        var i = 0
        while (i < nGenotypes - 1) {
          bitPacker += 0
          i += 1
        }
      } else {
        bb(samplePloidyStart + idx) = ploidy

        val F = if (gt.isLinearScale) {
          val conversionFactor = newSize.toDouble / 32768
          resizeAndComputeFractional(px, resized, fractional, conversionFactor)
        } else {
          resizeAndComputeFractional(gt.unboxedGP, resized, fractional, newSize)
        }

        roundWithConstantSum(resized, fractional, index, bitPacker, F, newSize, keepLast = false)
      }
      idx += 1
    }

    bitPacker.flush()
    bb.result()
  }
}

trait IntConsumer {
  def +=(x: Int): Unit
}

class BitPacker(bb: ArrayBuilder[Byte], nBitsPerProb: Int) extends IntConsumer {
  require(nBitsPerProb >= 1 && nBitsPerProb <= 32, "Number of bits per probability must be between 1 and 32 inclusive.")

  val bitMask = (1L << nBitsPerProb) - 1
  private var data = 0L
  private var dataSize = 0

  def +=(i: Int) {
    data |= ((i.toUIntFromRep.toLong & bitMask) << dataSize)
    dataSize += nBitsPerProb
    write()
  }

  private def write() {
    while (dataSize >= 8) {
      bb += data.toByte
      data = data >>> 8
      dataSize -= 8
    }
  }

  def flush() {
    if (dataSize > 0)
      bb += data.toByte
  }
}
