package org.broadinstitute.k3.variant

import net.jpountz.lz4.LZ4Factory
import org.broadinstitute.k3.utils.ByteStream

import scala.collection.mutable.ArrayBuilder
import scala.collection.mutable.Builder

class GenotypeStreamIterator(b: ByteStream) extends Iterator[(Int, Genotype)] {
  var i = 0

  override def hasNext: Boolean = {
    !b.eos
  }

  override def next(): (Int, Genotype) = {
    val prev = i
    i += 1
    (prev, Genotype.read(b))
  }
}

case class GenotypeStream(variant: Variant, decompLen: Int, a: Array[Byte])
  extends Iterable[(Int, Genotype)] {

  override def iterator: GenotypeStreamIterator = {
    var factory = LZ4Factory.fastestInstance()
    val decompressor = factory.fastDecompressor()

    val decomp = Array.ofDim[Byte](decompLen)
    val compLen = decompressor.decompress(a, 0, decomp, 0, decompLen);
    assert(compLen == a.length)

    new GenotypeStreamIterator(new ByteStream(decomp))
  }

  override def newBuilder: Builder[(Int, Genotype), GenotypeStream] = {
    new GenotypeStreamBuilder(variant)
  }
}

class GenotypeStreamBuilder(variant: Variant)
  extends Builder[(Int, Genotype), GenotypeStream] {
  val b = new ArrayBuilder.ofByte

  override def +=(g: (Int, Genotype)): GenotypeStreamBuilder.this.type = {
    g._2.write(b)
    this
  }

  override def clear() {
    b.clear()
  }

  override def result(): GenotypeStream = {
    val factory = LZ4Factory.fastestInstance()
    val compressor = factory.highCompressor()

    val a = b.result()
    val decompLen = a.length

    val maxLen = compressor.maxCompressedLength(decompLen)
    val compressed = Array.ofDim[Byte](maxLen)
    val compressedLen = compressor.compress(a, 0, a.length, compressed, 0, maxLen)

    new GenotypeStream(variant, decompLen, compressed.take(compressedLen))
  }
}
