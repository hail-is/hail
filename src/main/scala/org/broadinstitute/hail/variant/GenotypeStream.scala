package org.broadinstitute.hail.variant

import net.jpountz.lz4.LZ4Factory
import org.apache.spark.sql.types.StructType
import org.broadinstitute.hail.ByteIterator
import org.apache.spark.sql.Row
import org.apache.spark.sql.types._
import org.broadinstitute.hail.Utils._
import scala.collection.mutable

// FIXME use zipWithIndex
class GenotypeStreamIterator(nAlleles: Int, b: ByteIterator) extends Iterator[Genotype] {
  override def hasNext: Boolean = b.hasNext

  override def next(): Genotype = {
    Genotype.read(nAlleles, b)
  }
}

object LZ4Utils {
  val factory = LZ4Factory.fastestInstance()
  val compressor = factory.highCompressor()
  val decompressor = factory.fastDecompressor()

  def compress(a: Array[Byte]): Array[Byte] = {
    val decompLen = a.length

    val maxLen = compressor.maxCompressedLength(decompLen)
    val compressed = Array.ofDim[Byte](maxLen)
    val compressedLen = compressor.compress(a, 0, a.length, compressed, 0, maxLen)

    compressed.take(compressedLen)
  }

  def decompress(decompLen: Int, a: Array[Byte]) = {
    val decomp = Array.ofDim[Byte](decompLen)
    val compLen = decompressor.decompress(a, 0, decomp, 0, decompLen)
    assert(compLen == a.length)

    decomp
  }
}

case class GenotypeStream(nAlleles: Int, decompLenOption: Option[Int], a: Array[Byte])
  extends Iterable[Genotype] {

  override def iterator: GenotypeStreamIterator = {
    decompLenOption match {
      case Some(decompLen) =>
        new GenotypeStreamIterator(nAlleles, new ByteIterator(LZ4Utils.decompress(decompLen, a)))
      case None =>
        new GenotypeStreamIterator(nAlleles, new ByteIterator(a))
    }
  }

  override def newBuilder: mutable.Builder[Genotype, GenotypeStream] = {
    new GenotypeStreamBuilder(nAlleles)
  }

  def decompressed: GenotypeStream = {
    decompLenOption match {
      case Some(decompLen) =>
        GenotypeStream(nAlleles, None, LZ4Utils.decompress(decompLen, a))
      case None => this
    }
  }

  def compressed: GenotypeStream = {
    decompLenOption match {
      case Some(_) => this
      case None =>
        GenotypeStream(nAlleles, Some(a.length), LZ4Utils.compress(a))
    }
  }

  def toRow: Row = {
    Row.fromSeq(Array(
      decompLenOption.getOrElse(null),
      a
    ))
  }
}

object GenotypeStream {
  def schema: StructType = {
    StructType(Array(
      StructField("decompLen", IntegerType, nullable = true),
      StructField("bytes", ArrayType(ByteType), nullable = false)
    ))
  }

  def fromRow(nAlleles: Int, row: Row): GenotypeStream = {

    GenotypeStream(nAlleles,
      row.getAsOption[Int](0),
      row.getAs[mutable.WrappedArray[Byte]](1)
        .toArray)
  }
}

class GenotypeStreamBuilder(nAlleles: Int, compress: Boolean = true)
  extends mutable.Builder[Genotype, GenotypeStream] {

  val b = new mutable.ArrayBuilder.ofByte

  override def +=(g: Genotype): GenotypeStreamBuilder.this.type = {
    val gb = new GenotypeBuilder(nAlleles)
    gb.set(g)
    gb.write(b)
    this
  }


  def write(gb: GenotypeBuilder) {
    gb.write(b)
  }

  def ++=(i: Iterator[Genotype]): GenotypeStreamBuilder.this.type = {
    i.foreach(this += _)
    this
  }

  override def clear() {
    b.clear()
  }

  override def result(): GenotypeStream = {
    val a = b.result()
    if (compress)
      GenotypeStream(nAlleles, Some(a.length), LZ4Utils.compress(a))
    else
      GenotypeStream(nAlleles, None, a)
  }
}
