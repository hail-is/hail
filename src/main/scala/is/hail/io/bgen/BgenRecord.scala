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
  compressed: Boolean,
  nSamples: Int,
  includeGT: Boolean,
  includeGP: Boolean,
  includeDosage: Boolean,
  bfis: HadoopFSDataBinaryReader
) extends BgenRecord {
  var expectedDataSize: Int = _
  var expectedNumAlleles: Int = _
  var dataSize: Int = _

  def setExpectedDataSize(size: Int) {
    this.expectedDataSize = size
  }

  def setExpectedNumAlleles(n: Int) {
    this.expectedNumAlleles = n
  }

  override def getValue(rvb: RegionValueBuilder) {
    // require(input != null// , "called getValue before serialized value was set"
    // )

    // val a = if (compressed) decompress(input, expectedDataSize) else input
    // val reader = new ByteArrayReader(a)

    val start = bfis.fis.getPos
    val expectedEnd = start + dataSize

    if (rvb == null) {
      bfis.fis.skipBytes(dataSize)
    } else {
      val uncompressedInput = if (compressed)
        new InflaterInputStream(bfis.fis, new Inflater(), dataSize)
      else
        bfis.fis

      val reader =
        new LittleEndianDataInputStream(uncompressedInput)

      // val nRow = reader.readInt()
      // assert(nRow == nSamples// , s"$nRow $nSamples"
      // )
      reader.skipBytes(4)

      val nAlleles = reader.readShort()
      // assert(nAlleles == expectedNumAlleles// , s"Value for `nAlleles' in genotype probability data storage is not equal to value in variant identifying data. Expected $expectedNumAlleles but found $nAlleles."
      // )
      if (nAlleles != 2)
        fatal(s"Only biallelic variants supported, found variant with $nAlleles")

      val minPloidy = reader.read()
      val maxPloidy = reader.read()

      if (minPloidy != 2 || maxPloidy != 2)
        fatal(s"Hail only supports diploid genotypes. Found min ploidy equals `$minPloidy' and max ploidy equals `$maxPloidy'.")

      val sampleMissing = new Array[Byte](nSamples)
      reader.readFully(sampleMissing)

      // var i = 0
      // while (i < nSamples) {
      //   val ploidyAndMissingness = reader.read()
      //   // assert((ploidyAndMissingness & 0x3f) == 2// , s"Ploidy value must equal to 2. Found $ploidy."
      //   // )
      //   sampleMissing(i) = (ploidyAndMissingness & 0x80) == 1
      //   i += 1
      // }
      // assert(i == nSamples// , s"Number of ploidy values `$i' does not equal the number of samples `$nSamples'."
      // )

      val phase = reader.read()
      assert(phase == 0 || phase == 1// , s"Value for phase must be 0 or 1. Found $phase."
      )
      val isPhased = phase == 1

      if (isPhased)
        fatal("Hail does not support phased genotypes.")

      val nBitsPerProb = reader.read()
      assert(nBitsPerProb >= 1 && nBitsPerProb <= 32// , s"Value for nBits must be between 1 and 32 inclusive. Found $nBitsPerProb."
      )
      if (nBitsPerProb != 8)
        fatal(s"Only 8-bit probabilities supported, found $nBitsPerProb")

      // val nGenotypes = triangle(nAlleles)

      // val nExpectedBytesProbs = (nSamples * (nGenotypes - 1) * nBitsPerProb + 7) / 8
      // assert(reader.length == nExpectedBytesProbs + nSamples + 10// , s"Number of uncompressed bytes `${ reader.length }' does not match the expected size `$nExpectedBytesProbs'."
      // )

      rvb.startArray(nSamples) // gs
      if (!includeGT && !includeGP && !includeDosage) {
        assert(rvb.currentType().asInstanceOf[TStruct].byteSize == 0)
        rvb.unsafeAdvance(nSamples)
        // FIXME: we should use the datasize to seek past the data rather than
        // use the reader which will decompress all the bytes
        // reader.skipBytes(nSamples * 2)
        bfis.fis.seek(expectedEnd)
      } else if (nBitsPerProb == 8 && nAlleles == 2) {
        val c0 = Call2.fromUnphasedDiploidGtIndex(0)
        val c1 = Call2.fromUnphasedDiploidGtIndex(1)
        val c2 = Call2.fromUnphasedDiploidGtIndex(2)

        var i = 0
        while (i < nSamples) {
          // val off = nSamples + 10 + 2 * i
          val d0 = reader.read()
          val d1 = reader.read()

          if ((sampleMissing(i) & 0x80) == 1)
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
        // FIXME: does it still overshoot now that I tell it how much data to buffer?
        // the compressed reader will overshoot
        bfis.fis.seek(expectedEnd)
      }
      // assert(expectedEnd == bfis.fis.getPos, s"expected $expectedEnd, but found ${bfis.fis.getPos}, record size: $dataSize")
      rvb.endArray()
    }
  }
}
