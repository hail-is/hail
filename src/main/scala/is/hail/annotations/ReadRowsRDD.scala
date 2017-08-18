package is.hail.annotations

import java.io.{InputStream, OutputStream}
import java.util

import is.hail.expr._
import is.hail.utils.{SerializableHadoopConfiguration, _}
import is.hail.variant.LZ4Utils
import org.apache.commons.lang3.StringUtils
import org.apache.spark.rdd.RDD
import org.apache.spark.{Partition, SparkContext, TaskContext}

final class Decoder() {
  var inMem: Array[Byte] = _
  var inOff: Long = _

  def set(newInMem: Array[Byte]) {
    set(newInMem, 0)
  }

  def set(newInMem: Array[Byte], newInOff: Long) {
    inMem = newInMem
    inOff = newInOff
  }

  def readByte(): Byte = {
    val b = Memory.loadByte(inMem, inOff)
    inOff += 1
    b
  }

  def readFloat(): Float = {
    val d = Memory.loadFloat(inMem, inOff)
    inOff += 4
    d
  }

  def readDouble(): Double = {
    val d = Memory.loadDouble(inMem, inOff)
    inOff += 8
    d
  }

  def readBytes(mem: Long, off: Long, n: Int) {
    Memory.memcpy(mem + off, inMem, inOff, n)
    inOff += n
  }

  def readBoolean(): Boolean = readByte() != 0

  def readInt(): Int = {
    var b: Byte = readByte()
    var x: Int = b & 0x7f
    var shift: Int = 7
    while ((b & 0x80) != 0) {
      b = readByte()
      x |= ((b & 0x7f) << shift)
      shift += 7
    }

    x
  }

  def readLong(): Long = {
    var b: Byte = readByte()
    var x: Long = b & 0x7fL
    var shift: Int = 7
    while ((b & 0x80) != 0) {
      b = readByte()
      x |= ((b & 0x7fL) << shift)
      shift += 7
    }

    x
  }

  def readBinary(region: MemoryBuffer, off: Long) {
    val length = readInt()
    region.align(4)
    val boff = region.allocate(4 + length)
    region.storeAddress(off, boff)
    region.storeInt(boff, length)
    readBytes(region.mem, boff + 4, length)
  }

  def readArray(t: TArray, region: MemoryBuffer): Long = {
    val length = readInt()

    val contentSize = t.contentsByteSize(length)
    region.align(t.contentsAlignment)
    val aoff = region.allocate(contentSize)

    val nMissingBytes = (length + 7) / 8
    region.storeInt(aoff, length)
    readBytes(region.mem, aoff + 4, nMissingBytes)

    val elemsOff = aoff + t.elementsOffset(length)
    val elemSize = t.elementByteSize

    if (t.elementType == TInt32) { // fast path
      var i = 0
      while (i < length) {
        if (!region.loadBit(aoff + 4, i)) {
          val off = elemsOff + i * elemSize
          region.storeInt(off, readInt())
        }
        i += 1
      }
    } else {
      var i = 0
      while (i < length) {
        if (!region.loadBit(aoff + 4, i)) {
          val off = elemsOff + i * elemSize
          t.elementType match {
            case t2: TStruct => readStruct(t2, region, off)
            case t2: TArray =>
              val aoff = readArray(t2, region)
              region.storeAddress(off, aoff)
            case TBoolean => region.storeByte(off, readBoolean().toByte)
            case TInt32 => region.storeInt(off, readInt())
            case TInt64 => region.storeLong(off, readLong())
            case TFloat32 => region.storeFloat(off, readFloat())
            case TFloat64 => region.storeDouble(off, readDouble())
            case TBinary => readBinary(region, off)
          }
        }
        i += 1
      }
    }

    aoff
  }

  def readStruct(t: TStruct, region: MemoryBuffer, offset: Long) {
    val nMissingBytes = (t.size + 7) / 8
    readBytes(region.mem, offset, nMissingBytes)

    var i = 0
    while (i < t.size) {
      if (!region.loadBit(offset, i)) {
        val f = t.fields(i)
        val off = offset + t.byteOffsets(i)
        f.typ match {
          case t2: TStruct => readStruct(t2, region, off)
          case t2: TArray =>
            val aoff = readArray(t2, region)
            region.storeAddress(off, aoff)
          case TBoolean => region.storeByte(off, readBoolean().toByte)
          case TInt32 => region.storeInt(off, readInt())
          case TInt64 => region.storeLong(off, readLong())
          case TFloat32 => region.storeFloat(off, readFloat())
          case TFloat64 => region.storeDouble(off, readDouble())
          case TBinary => readBinary(region, off)
        }
      }
      i += 1
    }
  }

  def readRegionValue(t: Type, region: MemoryBuffer): Long = {
    val f = t.fundamentalType
    f match {
      case t: TStruct =>
        region.align(t.alignment)
        val start = region.allocate(t.byteSize)
        readStruct(t, region, start)
        start

      case t: TArray =>
        readArray(t, region)
    }
  }
}

final class Encoder() {
  var outMem: Array[Byte] = new Array[Byte](8 * 1024)
  var outOff: Long = 0

  def clear() {
    outOff = 0
  }

  def writeByte(b: Byte) {
    if (outOff + 1 > outMem.length) {
      outMem = util.Arrays.copyOf(outMem, outMem.length * 2)
    }
    Memory.storeByte(outMem, outOff, b)
    outOff += 1
  }

  def writeFloat(f: Float) {
    if (outOff + 4 > outMem.length) {
      outMem = util.Arrays.copyOf(outMem, outMem.length * 2)
    }
    Memory.storeFloat(outMem, outOff, f)
    outOff += 4
  }

  def writeDouble(d: Double) {
    if (outOff + 8 > outMem.length) {
      outMem = util.Arrays.copyOf(outMem, outMem.length * 2)
    }
    Memory.storeDouble(outMem, outOff, d)
    outOff += 8
  }

  def writeBytes(mem: Long, off: Long, n: Int) {
    if (outOff + n > outMem.length) {
      assert(outOff + n <= Int.MaxValue)
      val smallNewOutOff = (outOff + n).toInt
      outMem = util.Arrays.copyOf(outMem, math.max(smallNewOutOff, outMem.length * 2))
    }
    Memory.memcpy(outMem, outOff, mem + off, n)
    outOff += n
  }

  def writeBoolean(b: Boolean) {
    writeByte(b.toByte)
  }

  def writeInt(i: Int) {
    var j = i
    do {
      var b = j & 0x7f
      j >>>= 7
      if (j != 0)
        b |= 0x80
      writeByte(b.toByte)
    } while (j != 0)
  }

  def writeLong(l: Long) {
    var j = l
    do {
      var b = j & 0x7f
      j >>>= 7
      if (j != 0)
        b |= 0x80
      writeByte(b.toByte)
    } while (j != 0)
  }

  def writeBinary(region: MemoryBuffer, offset: Long) {
    val boff = region.loadAddress(offset)
    val length = region.loadInt(boff)
    writeInt(length)
    writeBytes(region.mem, boff + 4, length)
  }

  def writeArray(t: TArray, region: MemoryBuffer, aoff: Long) {
    val length = region.loadInt(aoff)

    val nMissingBytes = (length + 7) / 8
    writeInt(length)
    writeBytes(region.mem, aoff + 4, nMissingBytes)

    val elemsOff = aoff + t.elementsOffset(length)
    val elemSize = t.elementByteSize
    if (t.elementType == TInt32) { // fast case
      var i = 0
      while (i < length) {
        if (t.isElementDefined(region, aoff, i)) {
          val off = elemsOff + i * elemSize
          writeInt(region.loadInt(off))
        }
        i += 1
      }
    } else {
      var i = 0
      while (i < length) {
        if (t.isElementDefined(region, aoff, i)) {
          val off = elemsOff + i * elemSize
          t.elementType match {
            case t2: TStruct => writeStruct(t2, region, off)
            case t2: TArray => writeArray(t2, region, region.loadAddress(off))
            case TBoolean => writeBoolean(region.loadByte(off) != 0)
            case TInt32 => writeInt(region.loadInt(off))
            case TInt64 => writeLong(region.loadLong(off))
            case TFloat32 => writeFloat(region.loadFloat(off))
            case TFloat64 => writeDouble(region.loadDouble(off))
            case TBinary => writeBinary(region, off)
          }
        }

        i += 1
      }
    }
  }

  def writeStruct(t: TStruct, region: MemoryBuffer, offset: Long) {
    val nMissingBytes = (t.size + 7) / 8
    writeBytes(region.mem, offset, nMissingBytes)

    var i = 0
    while (i < t.size) {
      if (!region.loadBit(offset, i)) {
        val off = offset + t.byteOffsets(i)
        t.fields(i).typ match {
          case t2: TStruct => writeStruct(t2, region, off)
          case t2: TArray => writeArray(t2, region, region.loadAddress(off))
          case TBoolean => writeBoolean(region.loadByte(off) != 0)
          case TInt32 => writeInt(region.loadInt(off))
          case TInt64 => writeLong(region.loadLong(off))
          case TFloat32 => writeFloat(region.loadFloat(off))
          case TFloat64 => writeDouble(region.loadDouble(off))
          case TBinary => writeBinary(region, off)
        }
      }

      i += 1
    }
  }

  def writeRegionValue(t: Type, region: MemoryBuffer, offset: Long) {
    val f = t.fundamentalType
    (f: @unchecked) match {
      case t: TStruct =>
        writeStruct(t, region, offset)
      case t: TArray =>
        writeArray(t, region, offset)
    }
  }
}

class RichRDDRegionValue(val rdd: RDD[RegionValue]) extends AnyVal {
  def writeInt(out: OutputStream, i: Int) {
    out.write(i & 0xFF)
    out.write((i >> 8) & 0xFF)
    out.write((i >> 16) & 0xFF)
    out.write((i >> 24) & 0xFF)
  }

  def writeRows(path: String, t: TStruct) {
    val sc = rdd.sparkContext
    val hadoopConf = sc.hadoopConfiguration

    hadoopConf.mkDir(path + "/rowstore")

    val sHadoopConfBc = sc.broadcast(new SerializableHadoopConfiguration(hadoopConf))

    val nPartitions = rdd.partitions.length
    val d = digitsNeeded(nPartitions)

    val rowCount = rdd.mapPartitionsWithIndex { case (i, it) =>
      var rowCount = 0L

      val is = i.toString
      assert(is.length <= d)
      val pis = StringUtils.leftPad(is, d, "0")

      sHadoopConfBc.value.value.writeFile(path + "/rowstore/part-" + pis) { out =>
        var comp: Array[Byte] = null
        val en = new Encoder()

        it.foreach { rv =>
          // println("write rv", rv.pretty(t))

          en.clear()
          en.writeRegionValue(t.fundamentalType, rv.region, rv.offset)
          assert(en.outOff <= Int.MaxValue)
          val smallOutOff = en.outOff.toInt

          val decomp = en.outMem
          val decompLen = smallOutOff

          val maxCompLen = LZ4Utils.maxCompressedLength(decompLen)
          if (comp == null || maxCompLen > comp.length)
            comp = new Array[Byte](maxCompLen)

          val compLen = LZ4Utils.compress(comp, decomp, decompLen)

          // println("write", decompLen, compLen)

          writeInt(out, decompLen)
          writeInt(out, compLen)
          out.write(comp, 0, compLen)

          rowCount += 1
        }

        // -1
        writeInt(out, -1)
      }

      Iterator(rowCount)
    }
      .fold(0L)(_ + _)

    info(s"wrote $rowCount records")
  }
}

case class ReadRowsRDDPartition(index: Int) extends Partition

object ReadRowsRDD {
  def readInt(in: InputStream): Int = {
    val b0 = in.read()
    val b1 = in.read()
    val b2 = in.read()
    val b3 = in.read()

    (b0 & 0xFF) |
      ((b1 & 0xFF) << 8) |
      ((b2 & 0xFF) << 16) |
      ((b3 & 0xFF) << 24)
  }
}

class ReadRowsRDD(sc: SparkContext,
  path: String, t: TStruct, nPartitions: Int) extends RDD[RegionValue](sc, Nil) {

  override def getPartitions: Array[Partition] =
    Array.tabulate(nPartitions)(i => ReadRowsRDDPartition(i))

  private val sHadoopConfBc = sc.broadcast(new SerializableHadoopConfiguration(sc.hadoopConfiguration))

  override def compute(split: Partition, context: TaskContext): Iterator[RegionValue] = {
    val d = digitsNeeded(nPartitions)
    val localPath = path
    val localT = t

    new Iterator[RegionValue] {
      private val in = {
        val is = split.index.toString
        assert(is.length <= d)
        val pis = StringUtils.leftPad(is, d, "0")
        sHadoopConfBc.value.value.unsafeReader(localPath + "/rowstore/part-" + pis)
      }

      private var decompLen = ReadRowsRDD.readInt(in)

      private val region = MemoryBuffer()
      private val rv = RegionValue(region, 0)

      private val dec = new Decoder()

      private var comp: Array[Byte] = _
      private var decomp: Array[Byte] = _

      def hasNext: Boolean = decompLen != -1

      def next(): RegionValue = {
        if (!hasNext)
          throw new NoSuchElementException("next on empty iterator")

        val compLen = ReadRowsRDD.readInt(in)

        // println("read", decompLen, compLen)

        if (comp == null || comp.length < compLen)
          comp = new Array[Byte](compLen)
        in.read(comp, 0, compLen)

        if (decomp == null || decomp.length < decompLen)
          decomp = new Array[Byte](decompLen)
        LZ4Utils.decompress(decomp, decompLen, comp, compLen)

        dec.set(decomp, 0)
        region.clear()
        rv.offset = dec.readRegionValue(localT.fundamentalType, region)

        // println("read rv", rv.pretty(t))

        decompLen = ReadRowsRDD.readInt(in)
        if (decompLen == -1) {
          in.close()
        }

        rv
      }
    }
  }
}
