package is.hail.io.bgen

import is.hail.annotations._
import is.hail.io.{ByteArrayReader, KeySerializedValueRecord}
import is.hail.utils._
import is.hail.variant.{Call2, Genotype}

abstract class BgenRecord extends KeySerializedValueRecord[(String, Int, Array[String])] {
  var ann: Annotation = _

  def setAnnotation(ann: Annotation) {
    this.ann = ann
  }

  def getAnnotation: Annotation = ann

  override def getValue(rvb: RegionValueBuilder): Unit
}

class BgenRecordV12(compressed: Boolean, nSamples: Int,
  includeGT: Boolean, includeGP: Boolean, includeDosage: Boolean) extends BgenRecord {
  var expectedDataSize: Int = _
  var expectedNumAlleles: Int = _

  def setExpectedDataSize(size: Int) {
    this.expectedDataSize = size
  }

  def setExpectedNumAlleles(n: Int) {
    this.expectedNumAlleles = n
  }

  override def getValue(rvb: RegionValueBuilder) {
    require(input != null, "called getValue before serialized value was set")

    val a = if (compressed) decompress(input, expectedDataSize) else input
    val reader = new ByteArrayReader(a)

    val nRow = reader.readInt()
    assert(nRow == nSamples, "row nSamples is not equal to header nSamples")

    val nAlleles = reader.readShort()
    assert(nAlleles == expectedNumAlleles, s"Value for `nAlleles' in genotype probability data storage is not equal to value in variant identifying data. Expected $expectedNumAlleles but found $nAlleles.")
    if (nAlleles != 2)
      fatal(s"Only biallelic variants supported, found variant with $nAlleles")

    val minPloidy = reader.read()
    val maxPloidy = reader.read()

    if (minPloidy != 2 || maxPloidy != 2)
      fatal(s"Hail only supports diploid genotypes. Found min ploidy equals `$minPloidy' and max ploidy equals `$maxPloidy'.")

    var i = 0
    while (i < nSamples) {
      val ploidy = reader.read()
      assert((ploidy & 0x3f) == 2, s"Ploidy value must equal to 2. Found $ploidy.")
      i += 1
    }
    assert(i == nSamples, s"Number of ploidy values `$i' does not equal the number of samples `$nSamples'.")

    val phase = reader.read()
    assert(phase == 0 || phase == 1, s"Value for phase must be 0 or 1. Found $phase.")
    val isPhased = phase == 1

    if (isPhased)
      fatal("Hail does not support phased genotypes.")

    val nBitsPerProb = reader.read()
    assert(nBitsPerProb >= 1 && nBitsPerProb <= 32, s"Value for nBits must be between 1 and 32 inclusive. Found $nBitsPerProb.")
    if (nBitsPerProb != 8)
      fatal(s"Only 8-bit probabilities supported, found $nBitsPerProb")

    val nGenotypes = triangle(nAlleles)

    val nExpectedBytesProbs = (nSamples * (nGenotypes - 1) * nBitsPerProb + 7) / 8
    assert(reader.length == nExpectedBytesProbs + nSamples + 10, s"Number of uncompressed bytes `${ reader.length }' does not match the expected size `$nExpectedBytesProbs'.")

    rvb.startArray(nSamples) // gs
    val c0 = Call2.fromUnphasedDiploidGtIndex(0)
    val c1 = Call2.fromUnphasedDiploidGtIndex(1)
    val c2 = Call2.fromUnphasedDiploidGtIndex(2)

    i = 0
    while (i < nSamples) {
      val sampleMissing = (a(8 + i) & 0x80) != 0
      if (sampleMissing)
        rvb.setMissing()
      else {
        rvb.startStruct() // g

        val off = nSamples + 10 + 2 * i
        val d0 = a(off) & 0xff
        val d1 = a(off + 1) & 0xff
        val d2 = 255 - d0 - d1

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
    rvb.endArray()
  }
}
