package is.hail.io.bgen

import is.hail.annotations._
import is.hail.expr.TString
import is.hail.utils._
import is.hail.variant.{ArrayGenotypeView, Locus, Variant, VariantSampleMatrix}

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

  def stringToBytesWithShortLength(bb: ArrayBuilder[Byte], s: String): Int = {
    val bytes = s.getBytes
    val l = bytes.length
    shortToBytesLE(bb, l)
    bb ++= bytes
    2 + l
  }

  def stringToBytesWithIntLength(bb: ArrayBuilder[Byte], s: String): Int = {
    val bytes = s.getBytes
    val l = bytes.length
    intToBytesLE(bb, l)
    bb ++= bytes
    4 + l
  }

  private def resetIndex(index: Array[Int]) {
    var i = 0
    while (i < index.length) {
      index(i) = i
      i += 1
    }
  }

  private def quickSortWithIndex(a: Array[Double], idx: Array[Int], start: Int, n: Int) {
    def swap(i: Int, j: Int) {
      val tmp = idx(i)
      idx(i) = idx(j)
      idx(j) = tmp
    }

    if (n <= 1)
      return

    val pivotIdx = start + n / 2
    val pivot = a(idx(pivotIdx))
    swap(pivotIdx, start + n - 1)

    var left = start
    var right = start + n - 1

    while (left < right) {
      if (a(idx(left)) >= pivot)
        left += 1
      else if (a(idx(right - 1)) < pivot)
        right -= 1
      else {
        swap(left, right - 1)
        left += 1
        right -= 1
      }
    }

    swap(left, start + n - 1)

    quickSortWithIndex(a, idx, start, left - start)
    quickSortWithIndex(a, idx, left + 1, n - (left - start + 1))
  }

  def sortedIndex(a: Array[Double], idx: Array[Int]) {
    val n = a.length
    assert(idx.length == n)
    resetIndex(idx)
    quickSortWithIndex(a, idx, 0, n)
  }

  def computeFractional(input: Array[Double], fractional: Array[Double]): Int = {
    val n = input.length
    var totalFractional = 0d
    var i = 0
    while (i < n) {
      val x = input(i)
      val f = x - x.floor
      fractional(i) = f
      totalFractional += f
      i += 1
    }

    val F = (totalFractional + 0.5).toInt
    assert(F >= 0 && F <= n)
    F
  }

  def roundWithConstantSum(resized: Array[Double], fractional: Array[Double], index: Array[Int],
    indexInverse: Array[Int], output: IntConsumer, F: Int, expectedSize: Long, keepLast: Boolean = true) {
    val n = resized.length
    assert(fractional.length == n && index.length == n && indexInverse.length == n)
    sortedIndex(fractional, index)

    var i = 0
    while (i < n) {
      indexInverse(index(i)) = i
      i += 1
    }

    i = 0
    var newSize = 0d
    while (i < n) {
      val r = if (indexInverse(i) < F) resized(i).ceil else resized(i).floor
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
      sampleBlockLength += stringToBytesWithShortLength(bb, sampleIds(i))
      i += 1
    }

    val offset = headerLength + sampleBlockLength
    updateIntToBytesLE(bb, offset, 0)
    updateIntToBytesLE(bb, sampleBlockLength, 24)
  }

  def emitVariant(bb: ArrayBuilder[Byte], v: Variant, va: Annotation, view: ArrayGenotypeView,
    rsidQuery: Option[Querier], varidQuery: Option[Querier], nSamples: Int, nBitsPerProb: Int) {
    val nAlleles = v.nAlleles
    require(nAlleles <= 0xffff, s"Maximum number of alleles per variant is ${ 0xffff }. Found ${ v.nAlleles }.")

    val varid = varidQuery.flatMap(q => Option(q(va))).map(_.asInstanceOf[String]).getOrElse(v.toString)
    val rsid = rsidQuery.flatMap(q => Option(q(va))).map(_.asInstanceOf[String]).getOrElse(".")

    stringToBytesWithShortLength(bb, varid)
    stringToBytesWithShortLength(bb, rsid)
    stringToBytesWithShortLength(bb, v.contig)
    intToBytesLE(bb, v.start)
    shortToBytesLE(bb, nAlleles)
    stringToBytesWithIntLength(bb, v.ref)
    v.altAlleles.foreach(a => stringToBytesWithIntLength(bb, a.alt))

    val gtDataBlockStart = bb.length
    intToBytesLE(bb, 0) // placeholder for length of compressed data
    intToBytesLE(bb, 0) // placeholder for length of uncompressed data

    val uncompressedData = emitGPData(view, nSamples, nAlleles, nBitsPerProb)

    val uncompressedLength = uncompressedData.length
    val compressedLength = compress(bb, uncompressedData)

    updateIntToBytesLE(bb, compressedLength + 4, gtDataBlockStart)
    updateIntToBytesLE(bb, uncompressedLength, gtDataBlockStart + 4)
  }

  def emitGPData(view: ArrayGenotypeView, nSamples: Int, nAlleles: Int, nBitsPerProb: Int): Array[Byte] = {
    val bb = new ArrayBuilder[Byte]

    val nGenotypes = triangle(nAlleles)
    val newSize = (1L << nBitsPerProb) - 1

    intToBytesLE(bb, nSamples)
    shortToBytesLE(bb, nAlleles)
    bb += ploidy
    bb += ploidy

    val gpResized = new Array[Double](nGenotypes)
    val index = new Array[Int](nGenotypes)
    val indexInverse = new Array[Int](nGenotypes)
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
    i = 0
    while (i < nSamples) {
      view.setGenotype(i)

      if (view.hasGP) {
        var idx = 0
        while (idx < nGenotypes) {
          gpResized(idx) = view.getGP(idx) * newSize // Assuming sum(GP) == 1
          idx += 1
        }

        bb(samplePloidyStart + i) = ploidy
        val F = computeFractional(gpResized, fractional)
        roundWithConstantSum(gpResized, fractional, index, indexInverse, bitPacker, F, newSize, keepLast = false)
      } else {
        var gIdx = 0
        while (gIdx < nGenotypes - 1) {
          bitPacker += 0
          gIdx += 1
        }
      }

      i += 1
    }

    bitPacker.flush()
    bb.result()
  }

  def apply[T >: Null](vsm: VariantSampleMatrix[Locus, Variant, T], path: String, nBitsPerProb: Int = 8, parallel: Boolean = false) {
      val rsidQuery: Option[Querier] = vsm.vaSignature.getOption("rsid")
        .filter {
          case TString => true
          case t => warn(
            s"""found `rsid' field, but it was an unexpected type `$t'.  Emitting missing rsID.
             |  Expected ${ TString }""".stripMargin)
            false
        }.map(_ => vsm.queryVA("va.rsid")._2)

      val varidQuery: Option[Querier] = vsm.vaSignature.getOption("varid")
        .filter {
          case TString => true
          case t => warn(
            s"""found `varid' field, but it was an unexpected type `$t'.  Emitting missing variant ID.
             |  Expected ${ TString }""".stripMargin)
            false
        }.map(_ => vsm.queryVA("va.varid")._2)

      val sampleIds = vsm.stringSampleIds.toArray
      val partitionSizes = vsm.hc.sc.runJob(vsm.rdd, getIteratorSize _)
      val nVariants = partitionSizes.sum
      val nSamples = vsm.nSamples
      val localRowType = vsm.rowType

      val header = BgenWriter.headerBlock(sampleIds, nVariants)

      vsm.rdd2.mapPartitionsWithIndex { case (i: Int, it: Iterator[RegionValue]) =>
        val bb = new ArrayBuilder[Byte]
        val view = new ArrayGenotypeView(localRowType)

        val partHeader = if (parallel) {
          bb.clear()
          BgenWriter.emitHeaderBlock(bb, sampleIds, partitionSizes(i))
          bb.result()
        } else
          Array.empty[Byte]

        Iterator(partHeader) ++ it.map { rv =>
          view.setRegion(rv)
          val ur = new UnsafeRow(localRowType, rv)
          val v = ur.getAs[Variant](1)
          val va = ur.get(2)
          bb.clear()
          BgenWriter.emitVariant(bb, v, va, view, rsidQuery, varidQuery, nSamples, nBitsPerProb)
          bb.result()
        }
      }.saveFromByteArrays(path + ".bgen", vsm.hc.tmpDir,
        header = if (parallel && vsm.nPartitions != 0) None else Some(header), parallelWrite = parallel)

      vsm.writeSampleFile(path)
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
    assert(dataSize <= 8 && dataSize >= 0)
    if (dataSize > 0)
      bb += data.toByte
  }
}
