package org.broadinstitute.hail.variant

import net.jpountz.lz4.LZ4Factory

import scala.collection.mutable

// FIXME use zipWithIndex
class GenotypeStreamIterator(b: Iterator[Byte]) extends Iterator[Genotype] {
  override def hasNext: Boolean = b.hasNext

  override def next(): Genotype = {
    Genotype.read(b)
  }
}

object LZ4Utils {
  val factory = LZ4Factory.fastestInstance()
  val compressor = factory.highCompressor()
  val decompressor = factory.fastDecompressor()

  def compress(a: Array[Byte]): Array[Byte] = {
    val decompLen = a.size

    val maxLen = compressor.maxCompressedLength(decompLen)
    val compressed = Array.ofDim[Byte](maxLen)
    val compressedLen = compressor.compress(a, 0, a.size, compressed, 0, maxLen)

    compressed.take(compressedLen)
  }

  def decompress(decompLen: Int, a: Array[Byte]) = {
    val decomp = Array.ofDim[Byte](decompLen)
    val compLen = decompressor.decompress(a, 0, decomp, 0, decompLen)
    assert(compLen == a.length)

    decomp
  }
}

case class GenotypeStream(variant: Variant, decompLenOption: Option[Int], a: Array[Byte])
  extends Iterable[Genotype] {

  override def iterator: GenotypeStreamIterator = {
    decompLenOption match {
      case Some(decompLen) =>
        new GenotypeStreamIterator(LZ4Utils.decompress(decompLen, a).iterator)
      case None =>
        new GenotypeStreamIterator(a.iterator)
    }
  }

  override def newBuilder: mutable.Builder[Genotype, GenotypeStream] = {
    new GenotypeStreamBuilder(variant)
  }

  def decompressed: GenotypeStream = {
    decompLenOption match {
      case Some(decompLen) =>
        GenotypeStream(variant, None, LZ4Utils.decompress(decompLen, a))
      case None => this
    }
  }

  def compressed: GenotypeStream = {
    decompLenOption match {
      case Some(_) => this
      case None =>
        GenotypeStream(variant, Some(a.size), LZ4Utils.compress(a))
    }
  }
}

class GenotypeStreamBuilder(variant: Variant, compress: Boolean = true)
  extends mutable.Builder[Genotype, GenotypeStream] {
  val b = new mutable.ArrayBuilder.ofByte

  override def +=(g: Genotype): GenotypeStreamBuilder.this.type = {
    g.write(b)
    this
  }

  override def clear() {
    b.clear()
  }

  override def result(): GenotypeStream = {
    val a = b.result()
    if (compress)
      GenotypeStream(variant, Some(a.size), LZ4Utils.compress(a))
    else
      GenotypeStream(variant, None, a)
  }
}
