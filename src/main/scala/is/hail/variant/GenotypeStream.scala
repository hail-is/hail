package is.hail.variant

import java.nio.ByteBuffer

import is.hail.expr.{TBinary, TInt, TStruct, Type}
import is.hail.utils.{ByteIterator, _}
import net.jpountz.lz4.LZ4Factory
import org.apache.spark.sql.Row
import org.apache.spark.sql.types.{StructType, _}

import scala.collection.mutable

class GenotypeStreamIterator(nAlleles: Int, isDosage: Boolean, b: ByteIterator) extends Iterator[Genotype] {
  override def hasNext: Boolean = b.hasNext

  override def next(): Genotype = Genotype.read(nAlleles, isDosage, b)
}

class HardCallGenotypeStreamIterator(nAlleles: Int, isDosage: Boolean, b: ByteIterator) extends HailIterator[Int] {
  override def hasNext: Boolean = b.hasNext

  override def next(): Int = Genotype.readHardCallGenotype(nAlleles, isDosage, b)
}

class BiallelicDosageStreamIterator(nAlleles: Int, isDosage: Boolean, b: ByteIterator) extends HailIterator[Double] {
  require(nAlleles == 2)

  override def hasNext: Boolean = b.hasNext

  override def next(): Double = Genotype.readBiallelicDosage(isDosage, b)
}

class MutableGenotypeStreamIterator(nAlleles: Int, isDosage: Boolean, b: ByteIterator) extends Iterator[Genotype] {
  private val mutableGenotype = new MutableGenotype(nAlleles)

  override def hasNext: Boolean = b.hasNext

  override def next(): Genotype = {
    mutableGenotype.read(nAlleles, isDosage, b)
    mutableGenotype
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

case class GenotypeStream(nAlleles: Int, isDosage: Boolean, decompLenOption: Option[Int], a: Array[Byte])
  extends Iterable[Genotype] {

  override def iterator: GenotypeStreamIterator = {
    decompLenOption match {
      case Some(decompLen) =>
        new GenotypeStreamIterator(nAlleles, isDosage, new ByteIterator(LZ4Utils.decompress(decompLen, a)))
      case None =>
        new GenotypeStreamIterator(nAlleles, isDosage, new ByteIterator(a))
    }
  }

  def mutableIterator: MutableGenotypeStreamIterator = {
    val bytes = decompLenOption.map(dl => LZ4Utils.decompress(dl, a)).getOrElse(a)
    new MutableGenotypeStreamIterator(nAlleles, isDosage, new ByteIterator(bytes))
  }

  def gsHardCallGenotypeIterator: HardCallGenotypeStreamIterator = {
    val bytes = decompLenOption.map(dl => LZ4Utils.decompress(dl, a)).getOrElse(a)
    new HardCallGenotypeStreamIterator(nAlleles, isDosage, new ByteIterator(bytes))
  }

  def gsBiallelicDosageIterator: BiallelicDosageStreamIterator = {
    val bytes = decompLenOption.map(dl => LZ4Utils.decompress(dl, a)).getOrElse(a)
    new BiallelicDosageStreamIterator(nAlleles, isDosage, new ByteIterator(bytes))
  }

  override def newBuilder: mutable.Builder[Genotype, GenotypeStream] = {
    new GenotypeStreamBuilder(nAlleles, isDosage)
  }

  def decompressed: GenotypeStream = {
    decompLenOption match {
      case Some(decompLen) =>
        GenotypeStream(nAlleles, isDosage, None, LZ4Utils.decompress(decompLen, a))
      case None => this
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
      StructField("bytes", BinaryType, nullable = false)
    ))
  }

  def t: Type = TStruct("decompLen" -> TInt,
    "bytes" -> TBinary)

  def fromRow(nAlleles: Int, isDosage: Boolean, row: Row): GenotypeStream = {

    val bytes: Array[Byte] = row.get(1) match {
      case ab: Array[Byte] =>
        ab
      case sb: Seq[_] =>
        sb.asInstanceOf[Seq[Byte]].toArray
      case bb: ByteBuffer =>
        val b: Array[Byte] = Array.ofDim[Byte](bb.remaining())
        bb.get(b)
        b
    }

    GenotypeStream(nAlleles,
      isDosage,
      row.getAsOption[Int](0),
      bytes)
  }
}

class GenotypeStreamBuilder(nAlleles: Int, isDosage: Boolean = false)
  extends mutable.Builder[Genotype, GenotypeStream] {

  val b: ArrayBuilder[Byte] = new ArrayBuilder[Byte]()

  override def +=(g: Genotype): GenotypeStreamBuilder.this.type = {
    val gb = new GenotypeBuilder(nAlleles, isDosage)
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
    GenotypeStream(nAlleles, isDosage, Some(a.length), LZ4Utils.compress(a))
  }
}
