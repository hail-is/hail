package is.hail.io.bgen

import is.hail.annotations._
import is.hail.io._
import is.hail.utils._
import is.hail.variant.{Call2, Genotype}

class BgenRecordV12 (
  compressed: Boolean,
  nSamples: Int,
  includeGT: Boolean,
  includeGP: Boolean,
  includeDosage: Boolean,
  includeLid: Boolean,
  includeRsid: Boolean,
  bfis: HadoopFSDataBinaryReader,
  end: Long,
  partitionsFirstFileRowIndex: Long,
  split: BgenV12InputSplit
) {
  private[this] var rsid: String = _
  private[this] var lid: String = _
  private[this] var fileRowIdx: Long = partitionsFirstFileRowIndex - 1
  private[this] var contig: String = _
  private[this] var position: Int = _
  private[this] var alleles: Array[String] = _
  private[this] var data: Array[Byte] = _
  private[this] var dataSize: Int = 0
  private[this] var filterIndex = 0

  def getContig: String = contig

  def getPosition: Int = position

  def getAlleles: Array[String] = alleles

  def getRsid: String = rsid

  def getLid: String = lid

  def getFileRowIdx: Long = fileRowIdx

  private[this] def includeAnyEntryFields =
    includeGT || includeGP || includeDosage

  def advance(): Boolean = {
    if (split.hasFilter) {
      if (filterIndex < split.keptIndices.length) {
        fileRowIdx = split.keptIndices(filterIndex)
        filterIndex += 1
      } else
        return false
    } else {
      if (bfis.getPosition < end)
        fileRowIdx += 1
      else
        return false
    }

    if (includeLid)
      lid = bfis.readLengthAndString(2)
    else
      bfis.readLengthAndSkipString(2)
    if (includeRsid)
      rsid = bfis.readLengthAndString(2)
    else
      bfis.readLengthAndSkipString(2)
    contig = bfis.readLengthAndString(2)
    position = bfis.readInt()

    val nAlleles = bfis.readShort()
    if (nAlleles != 2)
      fatal(s"Only biallelic variants supported, found variant with $nAlleles")

    alleles = new Array[String](nAlleles)

    val ref = bfis.readLengthAndString(4)
    alleles(0) = ref

    var aIdx = 1
    while (aIdx < nAlleles) {
      alleles(aIdx) = bfis.readLengthAndString(4)
      aIdx += 1
    }

    val recodedContig = contig match {
      case "23" => "X"
      case "24" => "Y"
      case "25" => "X"
      case "26" => "MT"
      case x => x
    }

    dataSize = bfis.readInt()

    if (includeAnyEntryFields) {
      data = if (compressed) {
        val expectedDataSize = bfis.readInt()
        val input = bfis.readBytes(dataSize - 4)
        decompress(input, expectedDataSize)
      } else {
        bfis.readBytes(dataSize)
      }
      val reader = new ByteArrayReader(data)

      val nRow = reader.readInt()
      if (nRow != nSamples)
        fatal("row nSamples is not equal to header nSamples $nRow, $nSamples")

      val nAlleles2 = reader.readShort()
      if (nAlleles != nAlleles2)
        fatal(s"""Value for `nAlleles' in genotype probability data storage is
                 |not equal to value in variant identifying data. Expected
                 |$nAlleles but found $nAlleles2 at $lid.""".stripMargin)

      val minPloidy = reader.read()
      val maxPloidy = reader.read()

      if (minPloidy != 2 || maxPloidy != 2)
        fatal(s"Hail only supports diploid genotypes. Found min ploidy equals `$minPloidy' and max ploidy equals `$maxPloidy'.")

      var i = 0
      while (i < nSamples) {
        val ploidy = reader.read()
        if ((ploidy & 0x3f) != 2)
          fatal(s"Ploidy value must equal to 2. Found $ploidy.")
        i += 1
      }

      val phase = reader.read()
      if (phase != 0 && phase != 1)
        fatal(s"Value for phase must be 0 or 1. Found $phase.")
      val isPhased = phase == 1

      if (isPhased)
        fatal("Hail does not support phased genotypes.")

      val nBitsPerProb = reader.read()
      if (nBitsPerProb < 1 || nBitsPerProb > 32)
        fatal(s"Value for nBits must be between 1 and 32 inclusive. Found $nBitsPerProb.")
      if (nBitsPerProb != 8)
        fatal(s"Only 8-bit probabilities supported, found $nBitsPerProb")

      val nGenotypes = triangle(nAlleles)

      val nExpectedBytesProbs = (nSamples * (nGenotypes - 1) * nBitsPerProb + 7) / 8
      if (reader.length != nExpectedBytesProbs + nSamples + 10)
        fatal(s"""Number of uncompressed bytes `${ reader.length }' does not
                 |match the expected size `$nExpectedBytesProbs'.""".stripMargin)
    } else {
      bfis.skipBytes(dataSize)
    }

    if (split.hasFilter) {
      if (filterIndex < split.keptIndices.length) {
        // skip to next variant
        bfis.seek(split.keptPositions(filterIndex))
      }
    }

    return true
  }

  def getValue(rvb: RegionValueBuilder) {
    rvb.startArray(nSamples) // gs
    if (includeAnyEntryFields) {
      val c0 = Call2.fromUnphasedDiploidGtIndex(0)
      val c1 = Call2.fromUnphasedDiploidGtIndex(1)
      val c2 = Call2.fromUnphasedDiploidGtIndex(2)

      var i = 0
      while (i < nSamples) {
        val sampleMissing = (data(8 + i) & 0x80) != 0
        if (sampleMissing)
          rvb.setMissing()
        else {
          rvb.startStruct() // g

          val off = nSamples + 10 + 2 * i
          val d0 = data(off) & 0xff
          val d1 = data(off + 1) & 0xff
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
    } else {
      assert(rvb.currentType().byteSize == 0)
      rvb.unsafeAdvance(nSamples)
    }
    rvb.endArray()
  }

  def getAnnotation(): Annotation =
    Annotation(rsid, lid)
}
