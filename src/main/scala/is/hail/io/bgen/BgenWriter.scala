package is.hail.io.bgen

import is.hail.annotations._
import is.hail.expr.{TString, TStruct, Type}
import is.hail.utils._
import is.hail.variant.{ArrayGenotypeView, Locus, Variant, VariantSampleMatrix}

class BGENVariantAnnotationsView(rowType: TStruct) {
  private val tva = rowType.fieldType(2) match {
    case t: TStruct => t.asInstanceOf[TStruct]
    case _ => null
  }

  private def lookupField(name: String, expected: Type): (Boolean, Int) = {
    if (tva != null) {
      tva.selfField(name) match {
        case Some(f) =>
          if (f.typ == expected)
            (true, f.index)
          else
            (false, 0)
        case None => (false, 0)
      }
    } else
      (false, 0)
  }

  private val (varidExists, varidIndex) = lookupField("varid", TString())
  private val (rsidExists, rsidIndex) = lookupField("rsid", TString())

  private var m: MemoryBuffer = _
  private var vaOffset: Long = _

  def setRegion(mb: MemoryBuffer, offset: Long) {
    this.m = mb
    vaOffset = rowType.loadField(m, offset, 2)
  }

  def setRegion(rv: RegionValue): Unit = setRegion(rv.region, rv.offset)

  def hasRsid: Boolean = rsidExists && tva.isFieldDefined(m, vaOffset, rsidIndex)

  def hasVarid: Boolean = varidExists && tva.isFieldDefined(m, vaOffset, varidIndex)

  def getRsid: String = {
    val rsidOffset = tva.loadField(m, vaOffset, rsidIndex)
    TString.loadString(m, rsidOffset)
  }

  def getVarid: String = {
    val varidOffset = tva.loadField(m, vaOffset, varidIndex)
    TString.loadString(m, varidOffset)
  }
}

class BgenPartitionWriter(rowType: TStruct, nSamples: Int, nBitsPerProb: Int) {
  import BgenWriter._
  assert(nBitsPerProb > 0 && nBitsPerProb <= 32)
  val probTotal = (1L << nBitsPerProb) - 1

  val bb: ArrayBuilder[Byte] = new ArrayBuilder[Byte]
  val uncompressedData: ArrayBuilder[Byte] = new ArrayBuilder[Byte]
  val gView = new ArrayGenotypeView(rowType)
  val vaView = new BGENVariantAnnotationsView(rowType)

  def emitVariant(rv: RegionValue): Array[Byte] = {
    bb.clear()

    val ur = new UnsafeRow(rowType, rv)
    val v = ur.getAs[Variant](1)
    vaView.setRegion(rv)
    gView.setRegion(rv)

    val nAlleles = v.nAlleles
    require(nAlleles <= 0xffff, s"Maximum number of alleles per variant is ${ 0xffff }. Found ${ v.nAlleles }.")

    val varid = if (vaView.hasVarid) vaView.getVarid else v.toString
    val rsid = if (vaView.hasRsid) vaView.getRsid else "."

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

    emitGPData(nAlleles)

    val uncompressedLength = uncompressedData.length
    val compressedLength = compress(bb, uncompressedData.result())

    updateIntToBytesLE(bb, compressedLength + 4, gtDataBlockStart)
    updateIntToBytesLE(bb, uncompressedLength, gtDataBlockStart + 4)

    bb.result()
  }

  private def emitGPData(nAlleles: Int) {
    uncompressedData.clear()
    val nGenotypes = triangle(nAlleles)

    intToBytesLE(uncompressedData, nSamples)
    shortToBytesLE(uncompressedData, nAlleles)
    uncompressedData += ploidy
    uncompressedData += ploidy

    val gpResized = new Array[Double](nGenotypes)
    val index = new Array[Int](nGenotypes)
    val indexInverse = new Array[Int](nGenotypes)
    val fractional = new Array[Double](nGenotypes)

    val samplePloidyStart = uncompressedData.length
    var i = 0
    while (i < nSamples) {
      uncompressedData += 0x82.toByte // placeholder for sample ploidy - default is missing
      i += 1
    }

    uncompressedData += phased
    uncompressedData += nBitsPerProb.toByte

    val bitPacker = new BitPacker(uncompressedData, nBitsPerProb)

    def emitNullGP() {
      var gIdx = 0
      while (gIdx < nGenotypes - 1) {
        bitPacker += 0
        gIdx += 1
      }
    }

    i = 0
    while (i < nSamples) {
      gView.setGenotype(i)

      if (gView.hasGP) {
        var idx = 0
        var gpSum = 0d
        while (idx < nGenotypes) {
          val x = gView.getGP(idx)
          gpSum += x
          gpResized(idx) = x * probTotal // Assuming sum(GP) == 1
          idx += 1
        }

        if (gpSum >= 0.999 && gpSum <= 1.001) {
          uncompressedData(samplePloidyStart + i) = ploidy
          roundWithConstantSum(gpResized, fractional, index, indexInverse, bitPacker, probTotal)
        } else {
          warn(s"GP sum was not in the range [0.999, 1.001]. Found $gpSum. Emitting missing probabilities.")
          emitNullGP()
        }
      } else {
        emitNullGP()
      }

      i += 1
    }

    bitPacker.flush()
  }
}

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

  def roundWithConstantSum(input: Array[Double], fractional: Array[Double], index: Array[Int],
    indexInverse: Array[Int], output: IntConsumer, expectedSize: Long) {
    val n = input.length
    assert(fractional.length == n && index.length == n && indexInverse.length == n)

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

    sortedIndex(fractional, index)

    i = 0
    while (i < n) {
      indexInverse(index(i)) = i
      i += 1
    }

    i = 0
    var newSize = 0d
    while (i < n) {
      val r = if (indexInverse(i) < F) input(i).ceil else input(i).floor
      if (i != n - 1)
        output += r.toUInt.intRep
      newSize += r
      i += 1
    }
    assert(newSize == expectedSize)
  }

  private def headerBlock(sampleIds: Array[String], nVariants: Long): Array[Byte] = {
    val bb = new ArrayBuilder[Byte]
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
    bb.result()
  }

  def apply[T >: Null](vsm: VariantSampleMatrix[Locus, Variant, T], path: String, nBitsPerProb: Int = 8, parallel: Boolean = false) {
      val sampleIds = vsm.stringSampleIds.toArray
      val partitionSizes = vsm.hc.sc.runJob(vsm.rdd, getIteratorSize _)
      val nVariants = partitionSizes.sum
      val nSamples = vsm.nSamples
      val localRowType = vsm.rowType

      val header = BgenWriter.headerBlock(sampleIds, nVariants)

      vsm.rdd2.mapPartitionsWithIndex { case (i: Int, it: Iterator[RegionValue]) =>
        val bpw = new BgenPartitionWriter(localRowType, nSamples, nBitsPerProb)

        val partHeader = if (parallel) {
          BgenWriter.headerBlock(sampleIds, partitionSizes(i))
        } else
          Array.empty[Byte]

        Iterator(partHeader) ++ it.map { rv => bpw.emitVariant(rv) }
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
