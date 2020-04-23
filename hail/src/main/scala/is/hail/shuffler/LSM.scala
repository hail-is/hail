package is.hail.shuffler

import java.io._
import java.util._
import java.util.function._

import com.indeed.util.serialization._
import com.indeed.lsmtree.core._
import is.hail.annotations._
import is.hail.utils._
import org.apache.log4j.Logger

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

object LSM {
  val nKeySamples = 10000
}

class LSM (
  path: String,
  codecs: ShuffleCodecSpec
) extends AutoCloseable {
  private[this] val rootRegion: Region = Region()
  private[this] val log = Logger.getLogger(getClass.getName)
  val keyOrd: UnsafeOrdering = codecs.keyDecodedPType.unsafeOrdering
  private[this] val region = ThreadLocal.withInitial(new Supplier[Region]() {
    def get(): Region = {
      val region = Region()
      rootRegion.synchronized {
        rootRegion.addReferenceTo(region)
      }
      region
    }
  })

  def keyDec(in: InputStream) = {
    val dec = codecs.makeKeyDecoder(
      shuffleBufferSpec.buildInputBuffer(in))

    () => dec.readRegionValue(region.get)
  }
  def keyEnc(out: OutputStream) = {
    val enc = codecs.makeKeyEncoder(
      shuffleBufferSpec.buildOutputBuffer(out))

    { (x: Long) =>
      // FIXME: leaky regions here, maybe?
      enc.writeRegionValue(x)
      enc.flush()
    }
  }
  def dec(in: InputStream) = {
    val dec = codecs.makeRowDecoder(
      shuffleBufferSpec.buildInputBuffer(in))

    () => dec.readRegionValue(region.get)
  }
  def enc(out: OutputStream) = {
    val enc = codecs.makeRowEncoder(
      shuffleBufferSpec.buildOutputBuffer(out))

    { (x: Long) =>
      // FIXME: leaky regions here, maybe?
      enc.writeRegionValue(x)
      enc.flush()
    }
  }

  private[this] val store = new StoreBuilder[Long, Long](new File(path),
    new RegionValueSerializer(keyDec _, keyEnc _),
    new RegionValueSerializer(dec _, enc _)
  ).setComparator(keyOrd).build()
  private[this] val rnd = new Random()
  private[this] var processed = 0L
  private[this] var least = -1L
  private[this] var greatest = -1L
  private[this] var samples = new Array[Long](LSM.nKeySamples - 2)
  private[this] var sorted = false
  private[this] var samplesEnd = 0

  private[this] def maybeSample(k: Long): Unit = {
    if      (processed == 0)
      assert(least == -1 && greatest == -1)
    else if (processed == 1)
      assert(least != -1 && greatest == -1)
    else if (processed >= 2)
      assert(keyOrd.compare(least, greatest) < 0)
    assert(samplesEnd <= samples.length)

    if (processed == 0) {
      least = k
    } else if (processed == 1) {
      if (keyOrd.compare(k, least) < 0) {
        greatest = least
        least = k
      } else {
        greatest = k
      }
    } else if (samplesEnd < samples.length) {
      if (keyOrd.compare(k, least) < 0) {
        sorted = false
        samples(samplesEnd) = least
        least = k
        samplesEnd += 1
      } else if (keyOrd.compare(greatest, k) < 0) {
        sorted = false
        samples(samplesEnd) = greatest
        greatest = k
        samplesEnd += 1
      } else {
        sorted = false
        samples(samplesEnd) = k
        samplesEnd += 1
      }
    } else {
      var insertMe = k
      if (keyOrd.compare(k, least) < 0) {
        insertMe = least
        least = k
      } else if (keyOrd.compare(greatest, k) < 0) {
        insertMe = greatest
        greatest = k
      }

      if (rnd.nextDouble() < (samples.length * 1.0) / (processed + 1)) {
        assert(samplesEnd == samples.length)
        sorted = false
        samples(rnd.nextInt(samplesEnd)) = insertMe
      }
    }
  }

  def put(k: Long, v: Long): Unit = {
    maybeSample(k)
    store.put(k, v)
    processed += 1
  }

  def iterator(startKey: Long, inclusive: Boolean) = {
    store.iterator(startKey, inclusive)
  }

  def size: Long = processed

  def partitionKeys(nPartitions: Int): Array[Long] = {
    if (nPartitions == 0) {
      return new Array[Long](0)
    }
    if (processed == 0) {
      throw new RuntimeException("cannot partition nothing")
    }
    val currentlyGreatest = if (processed == 1) least else greatest
    assert(currentlyGreatest != -1)
    if (!sorted) {
      val boxed = samples.map(x => x: java.lang.Long).toArray[java.lang.Long]
      Arrays.sort(boxed, 0, samplesEnd, new java.util.Comparator[java.lang.Long]() {
        def compare(left: java.lang.Long, right: java.lang.Long) = keyOrd.compare(left, right)
      })
      samples = boxed.map(x => x: Long).toArray[Long]
      sorted = true
    }
    // we have samplesEnd + 2 keys, but we need one key to represent the "end",
    // so we partition one fewer elements
    val partitionSizes = partition(samplesEnd + 2 - 1, nPartitions)

    val partitionBounds = new Array[Long](nPartitions + 1)
    partitionBounds(0) = least
    var i = 1
    var nextBoundaryIndex = -1
    while (i <= nPartitions) {
      nextBoundaryIndex += partitionSizes(i - 1)
      if (nextBoundaryIndex < samplesEnd) {
        partitionBounds(i) = samples(nextBoundaryIndex)
      } else {
        partitionBounds(i) = currentlyGreatest
      }
      i += 1
    }
    partitionBounds
  }

  def close(): Unit = {
    try {
      rootRegion.close()
    } finally {
      store.close()
    }
  }
}
