package is.hail.io.bgen

import java.io.ByteArrayOutputStream
import java.util.zip.Deflater

import is.hail.annotations._
import is.hail.expr.TString
import is.hail.utils._
import is.hail.variant.{Genotype, Variant, VariantDataset}

import scala.collection.mutable.ArrayBuffer

object BgenWriter {
  val ploidy: Byte = 2
  val phased: Byte = 0

  def shortToBytesLE(i: Int): Array[Byte] = Array((i & 0xff).toByte, ((i >>> 8) & 0xff).toByte)

  def intToBytesLE(i: Int): Array[Byte] = {
    Array((i & 0xff).toByte,
      ((i >>> 8) & 0xff).toByte,
      ((i >>> 16) & 0xff).toByte,
      ((i >>> 24) & 0xff).toByte)
  }

  def stringToBytesWShortLength(s: String): Array[Byte] = {
    val bytes = s.getBytes
    shortToBytesLE(bytes.length) ++ bytes
  }

  def stringToBytesWIntLength(s: String): Array[Byte] = {
    val bytes = s.getBytes
    intToBytesLE(bytes.length) ++ bytes
  }

  def bitPack(input: Array[UInt], nBitsPerProb: Int): Array[Byte] = {
    require(nBitsPerProb >= 1 && nBitsPerProb <= 32, "Number of bits per probability must be between 1 and 32 inclusive.")

    val expectedNBytes = (input.length * nBitsPerProb + 7) / 8
    val output = Array.ofDim[Byte](expectedNBytes)

    val bitMask = (1L << nBitsPerProb) - 1
    var byteIndex = 0
    var data = 0L
    var dataSize = 0

    input.foreach { i =>
      data |= ((i.toLong & bitMask) << dataSize)
      dataSize += nBitsPerProb

      while (dataSize >= 8) {
        output(byteIndex) = (data & 0xffL).toByte
        data = data >>> 8
        dataSize -= 8
        byteIndex += 1
      }
    }

    if (dataSize > 0)
      output(byteIndex) = (data & 0xffL).toByte

    output
  }

  def resizeProbInts(input: Array[Int], conversionFactor: Double): Array[UInt] = {
    val resized = input.map(i => i * conversionFactor).zipWithIndex.map { case (d, i) => (i, d, math.floor(d)) }
    val totalFractional = resized.map(d => d._2 - d._3).sum

    resized.sortBy(d => -(d._2 - d._3)).zipWithIndex.map { case (r, i) =>
      if (i < totalFractional)
        (r._1, math.ceil(r._2))
      else
        (r._1, math.floor(r._2))
    }.sortBy(_._1).map(_._2.toUInt)
  }

  def emitHeaderBlock(vds: VariantDataset): Array[Byte] = {
    val nSamples = vds.nSamples
    val nVariants = vds.countVariants()
    assert(nVariants <= math.pow(2, 31) - 1, s"Maximum number of variants can export is (2^31 - 1). Found $nVariants.")

    val magicNumbers = Array("b", "g", "e", "n").flatMap(_.getBytes)
    val flags = 0x01 | (0x02 << 2) | (0x01 << 31)
    val sampleIdentifierBlock = emitSampleIdentifierBlock(vds.sampleIds.toArray)
    val headerLength = 20
    val sampleIdentifierLength = sampleIdentifierBlock.length
    val offset = headerLength + sampleIdentifierLength

    val byteBuffer = new ArrayBuilder[Byte]

    byteBuffer ++= intToBytesLE(offset)
    byteBuffer ++= intToBytesLE(headerLength)
    byteBuffer ++= intToBytesLE(nVariants.toInt)
    byteBuffer ++= intToBytesLE(nSamples)
    byteBuffer ++= magicNumbers
    byteBuffer ++= intToBytesLE(flags)

    byteBuffer.result() ++ sampleIdentifierBlock
  }

  def emitSampleIdentifierBlock(sampleIds: Array[String]): Array[Byte] = {
    val byteBuffer = new ArrayBuilder[Byte]
    sampleIds.foreach(s => byteBuffer ++= stringToBytesWShortLength(s))

    val sampleIdBytes = byteBuffer.result()
    val sampleBlockLength = sampleIdBytes.length + 8
    val nSamples = sampleIds.length

    intToBytesLE(sampleBlockLength) ++ intToBytesLE(nSamples) ++ sampleIdBytes
  }

  def emitGenotypeData(gs: Iterable[Genotype], samplePloidies: Array[Byte],
    nAlleles: Int, nBitsPerProb: Int, conversionFactor: Double, bitMask: Long): (Array[Byte], Int, Int) = {
    val byteBuffer = new ArrayBuilder[Byte]
    val compresser = new Deflater()

    val sampleProbs = new ArrayBuilder[UInt]
    val nGenotypes = triangle(nAlleles)
    val noCall = Array.fill[UInt](nGenotypes - 1)(UInt(0))

    gs.zipWithIndex.foreach { case (gt, i) =>
      val px = gt.unboxedPX
      if (px == null) {
        samplePloidies(i) = 0x80.toByte
        sampleProbs ++= noCall
      }
      else {
        samplePloidies(i) = ploidy
        sampleProbs ++= resizeProbInts(px, conversionFactor).dropRight(1)
      }
    }

    byteBuffer ++= intToBytesLE(samplePloidies.length)
    byteBuffer ++= shortToBytesLE(nAlleles)
    byteBuffer += ploidy
    byteBuffer += ploidy
    byteBuffer ++= samplePloidies
    byteBuffer += phased
    byteBuffer += nBitsPerProb.toByte
    byteBuffer ++= bitPack(sampleProbs.result(), nBitsPerProb)

    val uncompressedData = byteBuffer.result()
    compresser.setInput(uncompressedData)
    compresser.finish()

    val outputStream = new ByteArrayOutputStream(uncompressedData.length)

    val buffer = new Array[Byte](1024)
    while (!compresser.finished()) {
      val nBytesCompressed = compresser.deflate(buffer)
      outputStream.write(buffer, 0, nBytesCompressed)
    }
    outputStream.close()

    val compressedData = outputStream.toByteArray
    (compressedData, compressedData.length, uncompressedData.length)
  }

  def emitVariantBlock(v: Variant, va: Annotation, gs: Iterable[Genotype],
    rsidQuery: Option[Querier], varidQuery: Option[Querier], samplePloidies: Array[Byte], nBitsPerProb: Int,
    conversionFactor: Double, bitMask: Long): Array[Byte] = {

    val byteBuffer = new ArrayBuilder[Byte]
    val nAlleles = v.nAlleles

    require(nAlleles <= 0xffff, s"Maximum number of alleles per variant is ${ 0xffff }. Found ${ v.nAlleles }.")

    val varid = varidQuery.flatMap(q => Option(q(va))).map(_.asInstanceOf[String]).getOrElse(v.toString)
    val rsid = rsidQuery.flatMap(q => Option(q(va))).map(_.asInstanceOf[String]).getOrElse(".")

    byteBuffer ++= stringToBytesWShortLength(varid)
    byteBuffer ++= stringToBytesWShortLength(rsid)
    byteBuffer ++= stringToBytesWShortLength(v.contig)
    byteBuffer ++= intToBytesLE(v.start)
    byteBuffer ++= shortToBytesLE(nAlleles)
    byteBuffer ++= stringToBytesWIntLength(v.ref)
    v.altAlleles.foreach(a => byteBuffer ++= stringToBytesWIntLength(a.alt))

    val (gtData, compressedLength, decompressedLength) = emitGenotypeData(gs, samplePloidies, nAlleles, nBitsPerProb, conversionFactor, bitMask)
    byteBuffer ++= intToBytesLE(compressedLength + 4)
    byteBuffer ++= intToBytesLE(decompressedLength)
    byteBuffer ++= gtData

    byteBuffer.result()
  }
}
