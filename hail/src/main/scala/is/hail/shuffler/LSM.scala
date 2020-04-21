package is.hail.shuffler

import java.io._

import com.indeed.lsmtree.core._
import com.indeed.util.serialization._
import is.hail.annotations._
import is.hail.utils._

class DataOutputIsOutputStream(
  var out: DataOutput
) extends OutputStream {
  override def close() { }
  override def flush() { }
  override def write(b: Array[Byte]) {
    out.write(b)
  }
  override def write(b: Array[Byte], off: Int, len: Int) {
    out.write(b, off, len)
  }
  override def write(b: Int) {
    out.write(b)
  }
}

class DataInputIsInputStream(
  var in: DataInput
) extends InputStream {
  override def available() = 0
  override def close() { }
  override def mark(readlimit: Int) { }
  override def markSupported() = false
  override def read(): Int = {
    in.readByte()
  }
  override def read(b: Array[Byte]): Int = {
    try {
      in.readFully(b)
      b.length
    } catch {
      case e: EOFException =>
        fatal("""this is probably a bug in the implementation of this class.
                |a consumer asked to fill an array of bytes but we didn't
                |have enough bytes to fill it. that should be OK, but DataInput
                |makes this difficult to do efficiently.""".stripMargin, e)
    }
  }
  override def read(b: Array[Byte], off: Int, len: Int): Int = {
    try {
      in.readFully(b, off, len)
      len
    } catch {
      case e: EOFException =>
        fatal("""this is probably a bug in the implementation of this class.
                |a consumer asked to fill an array of bytes but we didn't
                |have enough bytes to fill it. that should be OK, but DataInput
                |makes this difficult to do efficiently.""".stripMargin, e)
    }
  }
  override def reset() = throw new IOException("unsupported")
  override def skip(n: Long) = {
    assert(n >= 0)
    assert(n <= (1 << 31) - 1)
    in.skipBytes(n.asInstanceOf[Int])
  }
}

class RegionValueSerializer(
  makeDec: InputStream => () => Long,
  makeEnc: (OutputStream) => Long => Unit
) extends Serializer[Long] {
  def write(l: Long, out: DataOutput): Unit = {
    if (out.isInstanceOf[OutputStream]) {
      makeEnc(out.asInstanceOf[OutputStream])(l)
    } else {
      makeEnc(new DataOutputIsOutputStream(out))(l)
    }
  }

  def read(in: DataInput): Long = {
    if (in.isInstanceOf[InputStream]) {
      makeDec(in.asInstanceOf[InputStream])()
    } else {
      makeDec(new DataInputIsInputStream(in))()
    }
  }
}

class LSM (
  path: String,
  codecs: KeyedCodecSpec,
  private[this] val region: Region
) {
  val ord: UnsafeOrdering = codecs.decodedKeyPType.unsafeOrdering

  def keyDec(in: InputStream) = {
    val dec = codecs.makeKeyDec(in)

    () => dec.readRegionValue(region)
  }
  def keyEnc(out: OutputStream) = {
    val enc = codecs.makeKeyEnc(out)

    { (x: Long) =>
      // FIXME: leaky regions here, maybe?
      enc.writeRegionValue(x)
      enc.flush()
    }
  }
  def dec(in: InputStream) = {
    val dec = codecs.makeDec(in)

    () => dec.readRegionValue(region)
  }
  def enc(out: OutputStream) = {
    val enc = codecs.makeEnc(out)

    { (x: Long) =>
      // FIXME: leaky regions here, maybe?
      enc.writeRegionValue(x)
      enc.flush()
    }
  }

  val store = new StoreBuilder[Long, Long](new File(path),
    new RegionValueSerializer(keyDec _, keyEnc _),
    new RegionValueSerializer(dec _, enc _)
  ).setComparator(ord).build()
}
