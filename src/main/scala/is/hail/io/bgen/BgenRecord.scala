package is.hail.io.bgen

import com.google.common.io.LittleEndianDataInputStream
import is.hail.annotations._
import is.hail.expr.types.TStruct
import is.hail.io.{ ByteArrayReader, HadoopFSDataBinaryReader, KeySerializedValueRecord }
import is.hail.utils._
import is.hail.variant.{Call2, Genotype}
import java.util.zip.{ Inflater, InflaterInputStream }

abstract class BgenRecord extends KeySerializedValueRecord[(String, Int, Array[String])] {
  var ann: Annotation = _

  def setAnnotation(ann: Annotation) {
    this.ann = ann
  }

  def getAnnotation: Annotation = ann

  override def getValue(rvb: RegionValueBuilder): Unit
}

class BGen12ProbabilityArray(a: Array[Byte], nSamples: Int, nGenotypes: Int, nBitsPerProb: Int) {

  def apply(s: Int, gi: Int): UInt = {
    assert(s >= 0 && s < nSamples)
    assert(gi >= 0 && gi < nGenotypes - 1)

    val firstBit = (s * (nGenotypes - 1) + gi) * nBitsPerProb

    var byteIndex = nSamples + 10 + (firstBit >> 3)
    var r = (a(byteIndex) & 0xff) >>> (firstBit & 7)
    var rBits = 8 - (firstBit & 7)
    byteIndex += 1

    while (rBits < nBitsPerProb) {
      r |= ((a(byteIndex) & 0xff) << rBits)
      byteIndex += 1
      rBits += 8
    }

    // clear upper bits that might have garbage from last byte or
    UInt.uintFromRep(r & ((1L << nBitsPerProb) - 1).toInt)
  }
}

class BgenRecordV12(
  isCompressed: Boolean,
  nSamples: Int,
  includeGT: Boolean,
  includeGP: Boolean,
  includeDosage: Boolean,
  bfis: HadoopFSDataBinaryReader
) extends BgenRecord {
  private[this] var expectedDataSize: Int = _
  private[this] var expectedNumAlleles: Int = _
  var dataSize: Int = _
  private[this] val inf = new Inflater()
  private[this] var compressed = new Array[Byte](0)
  private[this] var decompressed = new Array[Byte](0)

  def setExpectedDataSize(size: Int) {
    this.expectedDataSize = size
  }

  def setExpectedNumAlleles(n: Int) {
    this.expectedNumAlleles = n
  }

  override def getValue(rvb: RegionValueBuilder) {
    val start = bfis.fis.getPos
    val expectedEnd = start + dataSize

    if (rvb == null) {
      bfis.fis.skipBytes(dataSize)
    } else if (!includeGT && !includeGP && !includeDosage) {
      // in this case, we don't even need to decompress anything
      rvb.startArray(nSamples) // gs
      assert(rvb.currentType().asInstanceOf[TStruct].byteSize == 0)
      rvb.unsafeAdvance(nSamples)
      rvb.endArray()
      bfis.fis.skipBytes(dataSize)
    } else {
      val bytes = if (isCompressed) {
        if (dataSize > compressed.length)
          compressed = new Array[Byte](dataSize)
        if (expectedDataSize > decompressed.length)
          decompressed = new Array[Byte](expectedDataSize)
        bfis.fis.readFully(compressed, 0, dataSize)
        inf.reset()
        inf.setInput(compressed)
        val decsize = inf.inflate(decompressed)
        if (decsize != expectedDataSize)
          fatal(s"$decsize, $expectedDataSize, ${inf.needsInput()} ${inf.needsDictionary()}")
        decompressed
      } else {
        // no compression, so compressed buffer is of correct size
        bfis.fis.readFully(compressed)
        compressed
      }

      val reader = new ByteArrayReader(bytes)

      val nRow = reader.readInt()
      if (nRow != nSamples)
        fatal(s"$nRow $nSamples")

      val nAlleles = reader.readShort()
      if (nAlleles != expectedNumAlleles)
        fatal(s"Value for `nAlleles' in genotype probability data storage is not equal to value in variant identifying data. Expected $expectedNumAlleles but found $nAlleles.")
      if (nAlleles != 2)
        fatal(s"Only biallelic variants supported, found variant with $nAlleles")

      val minPloidy = reader.read()
      val maxPloidy = reader.read()

      if (minPloidy != 2 || maxPloidy != 2)
        fatal(s"Hail only supports diploid genotypes. Found min ploidy equals `$minPloidy' and max ploidy equals `$maxPloidy'.")

      reader.skipBytes(nSamples)

      val phase = reader.read()
      if (!(phase == 0 || phase == 1))
        fatal(s"Value for phase must be 0 or 1. Found $phase.")
      val isPhased = phase == 1

      if (isPhased)
        fatal("Hail does not support phased genotypes.")

      val nBitsPerProb = reader.read()
      if (!(nBitsPerProb >= 1 && nBitsPerProb <= 32))
        fatal(s"Value for nBits must be between 1 and 32 inclusive. Found $nBitsPerProb.")
      if (nBitsPerProb != 8)
        fatal(s"Only 8-bit probabilities supported, found $nBitsPerProb")

      rvb.startArray(nSamples) // gs
      if (nBitsPerProb == 8 && nAlleles == 2) {
        val c0 = Call2.fromUnphasedDiploidGtIndex(0)
        val c1 = Call2.fromUnphasedDiploidGtIndex(1)
        val c2 = Call2.fromUnphasedDiploidGtIndex(2)

        var i = 0
        while (i < nSamples) {
          val off = nSamples + 10 + 2 * i
          val d0 = bytes(off)
          val d1 = bytes(off + 1)

          if ((bytes(8 + i) & 0x3f) != 2)
            fatal(s"Ploidy value must equal to 2. Found ${(bytes(8 + i) & 0x3f)}")

          if ((bytes(8 + i) & 0x80) != 0)
            rvb.setMissing()
          else {
            val d2 = 255 - d0 - d1
            rvb.startStruct() // g

            if (includeGT) {
              if (d0 > d1) {
                if (d0 > d2)
                  rvb.addInt(c0)
                else if (d2 > d0)
                  rvb.addInt(c2)
                else {
                  // d0 == d2
                  rvb.setMissing()
                }
              } else {
                // d0 <= d1
                if (d2 > d1)
                  rvb.addInt(c2)
                else {
                  // d2 <= d1
                  if (d1 == d0 || d1 == d2)
                    rvb.setMissing()
                  else
                    rvb.addInt(c1)
                }
              }
            }

            if (includeGP) {
              rvb.startArray(3) // GP
              rvb.addDouble(d0 / 255.0)
              rvb.addDouble(d1 / 255.0)
              rvb.addDouble(d2 / 255.0)
              rvb.endArray()
            }

            if (includeDosage) {
              val dosage = (d1 + (d2 << 1)) / 255.0
              rvb.addDouble(dosage)
            }

            rvb.endStruct() // g
          }
          i += 1
        }
      }
      rvb.endArray()
    }
    if (expectedEnd != bfis.fis.getPos)
      fatal(s"expected $expectedEnd, but found ${bfis.fis.getPos}, record size: $dataSize")
  }
}
