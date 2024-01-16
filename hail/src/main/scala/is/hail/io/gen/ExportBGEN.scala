package is.hail.io.gen
import is.hail.expr.ir.ByteArrayBuilder
import is.hail.io.fs.FS

object BgenWriter {
  val ploidy: Byte = 2
  val phased: Byte = 0
  val totalProb: Int = 255

  def shortToBytesLE(bb: ByteArrayBuilder, i: Int): Unit = {
    bb += (i & 0xff).toByte
    bb += ((i >>> 8) & 0xff).toByte
  }

  def intToBytesLE(bb: ByteArrayBuilder, i: Int): Unit = {
    bb += (i & 0xff).toByte
    bb += ((i >>> 8) & 0xff).toByte
    bb += ((i >>> 16) & 0xff).toByte
    bb += ((i >>> 24) & 0xff).toByte
  }

  def stringToBytesWithShortLength(bb: ByteArrayBuilder, s: String): Int = {
    val bytes = s.getBytes
    val l = bytes.length
    shortToBytesLE(bb, l)
    bb ++= bytes
    2 + l
  }

  def stringToBytesWithIntLength(bb: ByteArrayBuilder, s: String): Int = {
    val bytes = s.getBytes
    val l = bytes.length
    intToBytesLE(bb, l)
    bb ++= bytes
    4 + l
  }

  def updateIntToBytesLE(bb: ByteArrayBuilder, i: Int, pos: Int): Unit = {
    bb(pos) = (i & 0xff).toByte
    bb(pos + 1) = ((i >>> 8) & 0xff).toByte
    bb(pos + 2) = ((i >>> 16) & 0xff).toByte
    bb(pos + 3) = ((i >>> 24) & 0xff).toByte
  }

  def headerBlock(sampleIds: Array[String], nVariants: Long, compression: Int): Array[Byte] = {
    val bb = new ByteArrayBuilder()
    val nSamples = sampleIds.length
    assert(nVariants < (1L << 32))

    val magicNumbers = Array("b", "g", "e", "n").flatMap(_.getBytes)
    val flags = compression | (0x02 << 2) | (0x01 << 31)
    val headerLength = 20

    intToBytesLE(bb, 0) // placeholder for offset
    intToBytesLE(bb, headerLength)
    intToBytesLE(bb, ((nVariants << 32) >> 32).toInt)
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

  def writeSampleFile(fs: FS, path: String, sampleIds: Array[String]): Unit = {
    fs.writeTable(
      path + ".sample",
      "ID_1 ID_2 missing" :: "0 0 0" :: sampleIds.map(s => s"$s $s 0").toList,
    )
  }

  def roundWithConstantSum(
    input: Array[Double],
    fractional: Array[Double],
    index: Array[Int],
    indexInverse: Array[Int],
    output: ByteArrayBuilder,
    expectedSize: Long,
  ): Unit = {
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

    resetIndex(index)
    quickSortWithIndex(fractional, index, 0, fractional.length)

    i = 0
    while (i < n) {
      indexInverse(index(i)) = i
      i += 1
    }

    i = 0
    var newSize = 0d
    while (i < n) {
      val r = if (indexInverse(i) < F) input(i).ceil.toInt else input(i).floor.toInt
      assert(r >= 0 && r < 256)
      if (i != n - 1)
        output += r.toByte
      newSize += r
      i += 1
    }
    assert(newSize == expectedSize)
  }

  private def resetIndex(index: Array[Int]): Unit = {
    var i = 0
    while (i < index.length) {
      index(i) = i
      i += 1
    }
  }

  private def quickSortWithIndex(a: Array[Double], idx: Array[Int], start: Int, n: Int): Unit = {
    def swap(i: Int, j: Int): Unit = {
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
}
