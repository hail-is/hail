package is.hail.io.bgen

import java.io.ByteArrayOutputStream
import java.util.zip.Deflater

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

  def stringToBytesWShortLength(bb: ArrayBuilder[Byte], s: String) {
    val bytes = s.getBytes
    shortToBytesLE(bb, bytes.length)
    bytes.foreach(bb += _)
  }

  def stringToBytesWIntLength(bb: ArrayBuilder[Byte], s: String) {
    val bytes = s.getBytes
    intToBytesLE(bb, bytes.length)
    bytes.foreach(bb += _)
  }

  def bitPack(input: Array[Int], nBitsPerProb: Int): Array[Byte] = {
    val bb = new ArrayBuilder[Byte]
    bitPack(bb, input, nBitsPerProb)
    bb.result()
  }

  def bitPack(bb: ArrayBuilder[Byte], input: Array[Int], nBitsPerProb: Int) {
    require(nBitsPerProb >= 1 && nBitsPerProb <= 32, "Number of bits per probability must be between 1 and 32 inclusive.")

    val bitMask = (1L << nBitsPerProb) - 1
    var byteIndex = 0
    var data = 0L
    var dataSize = 0

    input.foreach { i =>
      data |= ((i.toUIntNoCheck.toLong & bitMask) << dataSize)
      dataSize += nBitsPerProb

      while (dataSize >= 8) {
        bb += (data & 0xffL).toByte
        data = data >>> 8
        dataSize -= 8
        byteIndex += 1
      }
    }

    if (dataSize > 0)
      bb += (data & 0xffL).toByte
  }

  def resizeProbInts(input: Array[Int], conversionFactor: Double): Array[Int] = {
    val resized = input.map(i => i.toUIntNoCheck * conversionFactor).zipWithIndex.map { case (d, i) => (i, d, math.floor(d)) }
    val totalFractional = resized.map(d => d._2 - d._3).sum

    resized.sortBy(d => -(d._2 - d._3)).zipWithIndex.map { case (r, i) =>
      if (i < totalFractional)
        (r._1, math.ceil(r._2))
      else
        (r._1, math.floor(r._2))
    }.sortBy(_._1).map(_._2.toUInt.intRep)
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

    intToBytesLE(byteBuffer, offset)
    intToBytesLE(byteBuffer, headerLength)
    intToBytesLE(byteBuffer, nVariants.toInt)
    intToBytesLE(byteBuffer, nSamples)
    magicNumbers.foreach(byteBuffer += _)
    intToBytesLE(byteBuffer, flags)

    byteBuffer.result() ++ sampleIdentifierBlock
  }

  def emitSampleIdentifierBlock(sampleIds: Array[String]): Array[Byte] = {
    val sampleIdByteBuffer = new ArrayBuilder[Byte]
    val byteBuffer = new ArrayBuilder[Byte]

    sampleIds.foreach(s => stringToBytesWShortLength(sampleIdByteBuffer, s))

    val sampleIdBytes = sampleIdByteBuffer.result()
    val sampleBlockLength = sampleIdBytes.length + 8
    val nSamples = sampleIds.length

    intToBytesLE(byteBuffer, sampleBlockLength)
    intToBytesLE(byteBuffer, nSamples)
    sampleIdBytes.foreach(byteBuffer += _)

    byteBuffer.result()
  }

  def emitGenotypeData(gs: Iterable[Genotype], nSamples: Int,
    nAlleles: Int, nBitsPerProb: Int, conversionFactor: Double, bitMask: Long): (Array[Byte], Int, Int) = {
    val byteBuffer = new ArrayBuilder[Byte]
    val compressor = new Deflater()

    val sampleProbs = new ArrayBuilder[Int]
    val nGenotypes = triangle(nAlleles)
    val noCall = new Array[Int](nGenotypes - 1)

    intToBytesLE(byteBuffer, nSamples)
    shortToBytesLE(byteBuffer, nAlleles)
    byteBuffer += ploidy
    byteBuffer += ploidy

    gs.foreach { gt =>
      val px = gt.unboxedPX
      if (px == null) {
        byteBuffer += 0x80.toByte
        noCall.foreach(sampleProbs += _)
      }
      else {
        byteBuffer += ploidy
        resizeProbInts(px, conversionFactor).dropRight(1).foreach(sampleProbs += _)
      }
    }

    byteBuffer += phased
    byteBuffer += nBitsPerProb.toByte
    bitPack(byteBuffer, sampleProbs.result(), nBitsPerProb)

    val uncompressedData = byteBuffer.result()
    compressor.setInput(uncompressedData)
    compressor.finish()

    val outputStream = new ByteArrayOutputStream(uncompressedData.length)

    val buffer = new Array[Byte](1024)
    while (!compressor.finished()) {
      val nBytesCompressed = compressor.deflate(buffer)
      outputStream.write(buffer, 0, nBytesCompressed)
    }
    outputStream.close()

    val compressedData = outputStream.toByteArray
    (compressedData, compressedData.length, uncompressedData.length)
  }

  def emitVariantBlock(v: Variant, va: Annotation, gs: Iterable[Genotype],
    rsidQuery: Option[Querier], varidQuery: Option[Querier], nSamples: Int, nBitsPerProb: Int,
    conversionFactor: Double, bitMask: Long): Array[Byte] = {

    val byteBuffer = new ArrayBuilder[Byte]
    val nAlleles = v.nAlleles

    require(nAlleles <= 0xffff, s"Maximum number of alleles per variant is ${ 0xffff }. Found ${ v.nAlleles }.")

    val varid = varidQuery.flatMap(q => Option(q(va))).map(_.asInstanceOf[String]).getOrElse(v.toString)
    val rsid = rsidQuery.flatMap(q => Option(q(va))).map(_.asInstanceOf[String]).getOrElse(".")

    stringToBytesWShortLength(byteBuffer, varid)
    stringToBytesWShortLength(byteBuffer, rsid)
    stringToBytesWShortLength(byteBuffer, v.contig)
    intToBytesLE(byteBuffer, v.start)
    shortToBytesLE(byteBuffer, nAlleles)
    stringToBytesWIntLength(byteBuffer, v.ref)
    v.altAlleles.foreach(a => stringToBytesWIntLength(byteBuffer, a.alt))

    val (gtData, compressedLength, decompressedLength) = emitGenotypeData(gs, nSamples, nAlleles, nBitsPerProb, conversionFactor, bitMask)
    intToBytesLE(byteBuffer, compressedLength + 4)
    intToBytesLE(byteBuffer, decompressedLength)
    gtData.foreach(byteBuffer += _)

    byteBuffer.result()
  }
}
