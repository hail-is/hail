package is.hail.io.gen

import is.hail.HailContext
import is.hail.annotations.{RegionValue, UnsafeRow}
import is.hail.expr.ir.MatrixValue
import is.hail.expr.types.physical.PStruct
import is.hail.utils.ArrayBuilder
import is.hail.variant.{ArrayGenotypeView, RegionValueVariant, View}
import is.hail.utils._
import org.apache.hadoop.io.IOUtils
import org.apache.spark.TaskContext
import org.apache.spark.sql.Row

object BgenWriter {
  val ploidy: Byte = 2
  val phased: Byte = 0
  val totalProb: Int = 255

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

  def updateIntToBytesLE(bb: ArrayBuilder[Byte], i: Int, pos: Int) {
    bb(pos) = (i & 0xff).toByte
    bb(pos + 1) = ((i >>> 8) & 0xff).toByte
    bb(pos + 2) = ((i >>> 16) & 0xff).toByte
    bb(pos + 3) = ((i >>> 24) & 0xff).toByte
  }

  def headerBlock(sampleIds: IndexedSeq[String], nVariants: Long): Array[Byte] = {
    val bb = new ArrayBuilder[Byte]
    val nSamples = sampleIds.length
    assert(nVariants < (1L << 32))

    val magicNumbers = Array("b", "g", "e", "n").flatMap(_.getBytes)
    val flags = 0x01 | (0x02 << 2) | (0x01 << 31)
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

  def writeSampleFile(path: String, sampleIds: IndexedSeq[String]) {
    HailContext.sFS.writeTable(path + ".sample",
      "ID_1 ID_2 missing" :: "0 0 0" :: sampleIds.map(s => s"$s $s 0").toList)
  }
}

class BgenPartitionWriter(rowPType: PStruct, nSamples: Int) {
  import BgenWriter._

  val bb: ArrayBuilder[Byte] = new ArrayBuilder[Byte]
  val uncompressedData: ArrayBuilder[Byte] = new ArrayBuilder[Byte]
  val gs = new ArrayGenotypeView(rowPType)
  val v = new RegionValueVariant(rowPType)
  val va = new GenAnnotationView(rowPType)

  def emitVariant(rv: RegionValue): (Array[Byte], Long) = {
    bb.clear()

    gs.setRegion(rv)
    v.setRegion(rv)
    va.setRegion(rv)

    val alleles = v.alleles()
    val nAlleles = alleles.length
    require(nAlleles <= 0xffff, s"Maximum number of alleles per variant is ${ 0xffff }. Found ${ nAlleles }.")

    val chr = v.contig()
    val pos = v.position()

    stringToBytesWithShortLength(bb, va.varid())
    stringToBytesWithShortLength(bb, va.rsid())
    stringToBytesWithShortLength(bb, chr)
    intToBytesLE(bb, pos)
    shortToBytesLE(bb, nAlleles)

    var i = 0
    while (i < nAlleles) {
      stringToBytesWithIntLength(bb, alleles(i))
      i += 1
    }

    val gtDataBlockStart = bb.length
    intToBytesLE(bb, 0) // placeholder for length of compressed data
    intToBytesLE(bb, 0) // placeholder for length of uncompressed data

    val dropped = emitGPData(chr, pos, alleles)

    val uncompressedLength = uncompressedData.length
    val compressedLength = compress(bb, uncompressedData.result())

    updateIntToBytesLE(bb, compressedLength + 4, gtDataBlockStart)
    updateIntToBytesLE(bb, uncompressedLength, gtDataBlockStart + 4)

    (bb.result(), dropped)
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
    indexInverse: Array[Int], output: ArrayBuilder[Byte], expectedSize: Long) {
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
      val r = if (indexInverse(i) < F) input(i).ceil.toInt else input(i).floor.toInt
      assert(r >= 0 && r < 256)
      if (i != n - 1)
        output += r.toByte
      newSize += r
      i += 1
    }
    assert(newSize == expectedSize)
  }


  private def emitGPData(chr: String, pos: Int, alleles: Array[String]): Long = {
    val nAlleles = alleles.length
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
    uncompressedData += 8.toByte

    def emitNullGP() {
      var gIdx = 0
      while (gIdx < nGenotypes - 1) {
        uncompressedData += 0
        gIdx += 1
      }
    }

    var dropped = 0L
    i = 0
    while (i < nSamples) {
      gs.setGenotype(i)

      if (gs.hasGP) {
        var idx = 0
        var gpSum = 0d
        while (idx < nGenotypes) {
          val x = gs.getGP(idx)
          if (x < 0)
            fatal(s"found GP value less than 0: $x, at sample $i of variant " +
              s"$chr:$pos:${ alleles.head }:${ alleles.tail.mkString(",") }")
          gpSum += x
          gpResized(idx) = x * totalProb // Assuming sum(GP) == 1
          idx += 1
        }

        if (gpSum >= 0.999 && gpSum <= 1.001) {
          uncompressedData(samplePloidyStart + i) = ploidy
          roundWithConstantSum(gpResized, fractional, index, indexInverse, uncompressedData, totalProb)
        } else {
          dropped += 1
          emitNullGP()
        }
      } else {
        emitNullGP()
      }

      i += 1
    }

    dropped
  }
}

object ExportBGEN {
  def apply(mv: MatrixValue, path: String, exportType: Int): Unit = {
    val colValues = mv.colValues.javaValue

    val sampleIds = colValues.map(_.asInstanceOf[Row].getString(0))
    val partitionSizes = mv.rvd.countPerPartition()
    val nPartitions = partitionSizes.length
    val nVariants = partitionSizes.sum
    val nSamples = colValues.length

    val localRVRowPType = mv.rvRowPType

    val header = BgenWriter.headerBlock(sampleIds, nVariants)

    val hc = HailContext.get
    val fs = HailContext.sFS
    val bcFS = HailContext.bcFS

    fs.delete(path, recursive = true)

    val parallelOutputPath =
      if (exportType == ExportType.CONCATENATED)
        fs.getTemporaryFile(hc.tmpDir)
      else
        path + ".bgen"
    fs.mkDir(parallelOutputPath)

    val d = digitsNeeded(mv.rvd.getNumPartitions)

    val (files, droppedPerPart) = mv.rvd.crdd.boundary.mapPartitionsWithIndex { case (i: Int, it: Iterator[RegionValue]) =>
      val context = TaskContext.get
      val pf =
        parallelOutputPath + "/" +
          (if (exportType == ExportType.CONCATENATED)
            // includes stage info and UUID
            partFile(d, i, context)
          else
            // Spark style
            partFile(d, i))

      var dropped = 0L
      using(bcFS.value.unsafeWriter(pf)) { out =>
        val bpw = new BgenPartitionWriter(localRVRowPType, nSamples)

        if (exportType == ExportType.PARALLEL_HEADER_IN_SHARD) {
          out.write(
            BgenWriter.headerBlock(sampleIds, partitionSizes(i)))
        }

        it.foreach { rv =>
          val (b, d) = bpw.emitVariant(rv)
          out.write(b)
          dropped += d
        }
      }

      Iterator.single((pf, dropped))
    }
      .collect()
      .unzip

    val dropped = droppedPerPart.sum
    if (dropped != 0)
      warn(s"Set $dropped genotypes to missing: total GP probability did not lie in [0.999, 1.001].")

    if (exportType == ExportType.PARALLEL_SEPARATE_HEADER) {
      using(fs.unsafeWriter(parallelOutputPath + "/header")) { out =>
        out.write(
          BgenWriter.headerBlock(sampleIds, nVariants))
      }
    }

    if (exportType == ExportType.CONCATENATED) {
      val (_, dt) = time {
        using(fs.unsafeWriter(path + ".bgen")) { out =>
          out.write(
            BgenWriter.headerBlock(sampleIds, nVariants))

          files.foreach { f =>
            using(fs.unsafeReader(f)) { in =>
              IOUtils.copyBytes(in, out, 4096)
            }
          }
        }
      }

      info(
        s"""while writing:
           |    ${ path }.bgen
           |  merge time: ${ formatTime(dt) }""".stripMargin)

      fs.delete(parallelOutputPath, recursive = true)
    }

    BgenWriter.writeSampleFile(path, sampleIds)
  }
}
