package is.hail.shuffler

import is.hail.annotations._
import is.hail.expr.ir._
import is.hail.expr.types.virtual._
import is.hail.expr.types.physical._
import is.hail.rvd.RVDPartitioner
import is.hail.asm4s._
import is.hail.io._
import is.hail.utils._
import java.io.{ ByteArrayInputStream, ByteArrayOutputStream, InputStream, OutputStream }
import java.nio.ByteBuffer
import java.util.function.{ BiConsumer, Consumer, Supplier }
import java.util.{ Arrays, Comparator, SortedMap, TreeMap }
import java.util.concurrent.ConcurrentHashMap
import scala.collection.mutable
import scala.reflect.classTag

class LongArrayByte(val _1: Long, val _2: Array[Byte])

trait RegionCompareFunction { def apply(region: Region, l: Long, r: Long): Int }

object Shuffle {
  private def compileCompare(
    keyType: PBaseStruct,
    sortOrders: Array[SortOrder]
  ): RegionCompareFunction = {
    require(keyType.required)
    val fb = new EmitFunctionBuilder[RegionCompareFunction](
      Array(NotGenericTypeInfo[Region], NotGenericTypeInfo[Long], NotGenericTypeInfo[Long]),
      NotGenericTypeInfo[Int],
      namePrefix = "compare")
    val apply = fb.apply_method
    val left = apply.getArg[Long](2)
    val right = apply.getArg[Long](3)
    val ordering = keyType.codeOrdering(apply, sortOrders)
    fb.emit(ordering.compareNonnull(
      coerce[ordering.T](Region.getIRIntermediate(keyType)(left)),
      coerce[ordering.T](Region.getIRIntermediate(keyType)(right))))
    fb.result()()
  }
}

class Shuffle (
  private[this] val id: Long,
  wireKeyType: PBaseStruct,
  sortOrder: Array[SortOrder],
  bufferSpec: BufferSpec,
  private[this] val inPartitions: Int,
  private[this] val outPartitionsOrBoundsBytes: Either[Int, Array[Byte]]
) {
  require(wireKeyType.required)
  require(wireKeyType.size == sortOrder.size)
  import Shuffle._
  // FIXME: do threads see out of date region information? What are the
  // semantics of one thread writing to off heap memory and another thread
  // reading from off heap memory?
  private[this] val regions = new ConcurrentHashMap[Thread, Region]()
  private[this] val region = ThreadLocal.withInitial { () =>
    val r = Region()
    regions.put(Thread.currentThread(), r)
    r
  }
  private[this] val keyCodec = new Codec(TypedCodecSpec(wireKeyType, bufferSpec))
  private[this] val comparer = compileCompare(wireKeyType, sortOrder)
  trait IntervalContainsFunction {
    def apply(r: Region, i: Long, mi: Boolean, k: Long, mk: Boolean): Boolean
  }
  private[this] val (nOutPartitions, maybeBounds) = outPartitionsOrBoundsBytes match {
    case Left(n) => (n, None)
    case Right(boundsBytes) =>
      val intervalType = PInterval(wireKeyType, true)
      val boundsType = PArray(intervalType, true)
      val codec = new Codec(TypedCodecSpec(boundsType, bufferSpec))
      val dec = new ByteArrayDecoder(codec.buildDecoder)
      val r = region.get
      val off = dec.regionValueFromBytes(r, boundsBytes)
      val (_, makeIntervalContains) = ExecuteContext.scoped(ctx => Compile[IntervalContainsFunction, Boolean](
        ctx,
        None,
        Seq(("interval", intervalType, classTag[Long]),
          ("key", wireKeyType, classTag[Long])),
        Array[MaybeGenericTypeInfo[_]](
          NotGenericTypeInfo[Region],
          NotGenericTypeInfo[Long],
          NotGenericTypeInfo[Boolean],
          NotGenericTypeInfo[Long],
          NotGenericTypeInfo[Boolean]),
        NotGenericTypeInfo[Boolean],
        ApplySpecial(
          "contains",
          Seq(
            Ref("interval", intervalType.virtualType),
            Ref("key", wireKeyType.virtualType)),
          TBoolean(true)),
        nSpecialArgs = 1,
        // if optimize = true we need a hail context
        optimize = false))
      val intervalContains = makeIntervalContains(0, region.get)
      val len = boundsType.loadLength(off)
      val partitionContainsKey = (r: Region, i: Int, kOff: Long) =>
        intervalContains(r, boundsType.loadElement(off, len, i), false, kOff, false)
      (len, Some(partitionContainsKey))
  }
  private[this] def str(region: Region, offset: Long): String =
    UnsafeRow.read(keyCodec.memType, region, offset).toString
  private[this] var finished: Array[(ArrayBuilder[Long], mutable.ArrayBuffer[Array[Byte]])] =
    new Array(nOutPartitions)
  // mutable.ArrayBuffer is not thread-safe, but only one partitionId-attemptId
  // pair is talking to us at a time
  private[this] val pending
      : Array[ConcurrentHashMap[Int, (ArrayBuilder[Long], mutable.ArrayBuffer[Array[Byte]])]] =
    Array.fill(inPartitions)(new ConcurrentHashMap())
  private[this] var partitionOffsets: Array[Int] = null
  private[this] var output: Array[LongArrayByte] = null
  // private[this] var deletedPartitions: java.util.Set[Int] = null

  private[this] val decoder = ThreadLocal.withInitial(
    () => new ByteArrayDecoder(keyCodec.buildDecoder))
  private[this] val encoder = ThreadLocal.withInitial(
    () => new ByteArrayEncoder(keyCodec.buildEncoder))
  def addMany(partitionId: Int, attemptId: Int, pairs: Int, bb: ByteBuffer): Unit = {
    val part = pending(partitionId)
    part.putIfAbsent(attemptId, (new ArrayBuilder[Long](), new mutable.ArrayBuffer()))
    val (attemptKeys, attemptValues) = part.get(attemptId)
    val localDecoder = decoder.get
    val localRegion = region.get
    var i = 0
    while (i < pairs) {
      val key = ByteUtils.readByteArray(bb)
      val value = ByteUtils.readByteArray(bb)
      attemptKeys += localDecoder.regionValueFromBytes(localRegion, key)
      attemptValues.append(value)
      i += 1
    }
  }

  def finishPartition(partitionId: Int, attemptId: Int): Unit = {
    val part = pending(partitionId)
    if (part == null) {
      log.info(s"""received a second finish for a finished partition
                  |${partitionId} ${attemptId}""".stripMargin)
    } else {
      if (!part.containsKey(attemptId)) {
        throw new RuntimeException(
          s"""bogus attempt id ${attemptId} for ${partitionId} not in
           |${part.keySet} in shuffle ${id}""".stripMargin)
      }
      pending(partitionId) = null
      assert(!finished.contains(partitionId))
      finished(partitionId) = part.get(attemptId)
    }
  }

  def closed(): Boolean = partitionOffsets != null

  def close(os: OutputStream): Unit = {
    require(pending.forall(_ == null))
    require(finished.size == inPartitions)
    var nElements = 0
    var i = 0
    while (i < finished.length) {
      nElements += finished(i)._1.length
      i += 1
    }
    output = new Array[LongArrayByte](nElements)
    val localRegion = region.get
    i = 0
    var k = 0
    while (i < finished.length) {
      val (keys, values) = finished(i)
      var j = 0
      while (j < keys.length) {
        output(k) = new LongArrayByte(keys(j), values(j))
        j += 1
        k += 1
      }
      finished(i) = null
      i += 1
    }
    finished = null
    // Arrays.sort(output, new Comparator[LongArrayByte]() {
    Arrays.parallelSort(output, new Comparator[LongArrayByte]() {
      override def compare(l: LongArrayByte, r: LongArrayByte): Int =
        comparer(localRegion, l._1, r._1) })
    maybeBounds match {
      case None =>
        val n = output.length
        val x = n / nOutPartitions
        val partitionSize = Array.fill(nOutPartitions)(x)
        var i = 0
        while (i < n - x * nOutPartitions) {
          partitionSize(i) += 1
          i += 1
        }
        partitionOffsets = partitionSize.scanLeft(0)(_ + _).toArray
      case Some(partitionContainsKey) =>
        val ordering = keyCodec.memType.virtualType.ordering
        partitionOffsets = new Array[Int](nOutPartitions + 1)
        var i = 0
        var partitionIndex = 0
        while (partitionIndex < nOutPartitions) {
          while (i < output.length &&
            // FIXME: put intervals into region and compile this
            !partitionContainsKey(localRegion, partitionIndex, output(i)._1)) {
            i += 1
          }
          partitionOffsets(partitionIndex) = i
          partitionIndex += 1
        }
        if (nOutPartitions > 0) {
          while (i < output.length &&
            partitionContainsKey(localRegion, nOutPartitions - 1, output(i)._1)) {
            i += 1
          }
          partitionOffsets(nOutPartitions) = i
        }
    }
    // deletedPartitions = ConcurrentHashMap.newKeySet[Int]()
    // var i = 0
    // while (i < nOutPartitions) {
    //   deletedPartitions.add(i)
    //   i += 1
    // }
    assert(partitionOffsets.length == nOutPartitions + 1)
    val localEncoder = encoder.get
    i = 0
    while (i < nOutPartitions) {
      writeByteArray(os,
        localEncoder.regionValueToBytes(localRegion, output(partitionOffsets(i))._1))
      i += 1
    }
    writeByteArray(os,
      localEncoder.regionValueToBytes(localRegion, output(output.length - 1)._1))
  }

  private[this] def writeInt(out: OutputStream, i: Int): Unit = {
    out.write(i)
    out.write(i >> 8)
    out.write(i >> 16)
    out.write(i >> 24)
  }

  private[this] def writeByteArray(out: OutputStream, bytes: Array[Byte]): Unit = {
    writeInt(out, bytes.length)
    out.write(bytes)
  }

  def get(partitionId: Int, os: OutputStream): Unit = {
    val localRegion = region.get
    val localEncoder = encoder.get
    require(partitionOffsets != null)
    if (partitionId > nOutPartitions) {
      throw new RuntimeException(
        s"no such partition id ${partitionId} in shuffle ${id}")
    }
    var start = partitionOffsets(partitionId)
    val end = partitionOffsets(partitionId + 1)
    writeInt(os, end - start)
    while (start < end) {
      val kv = output(start)
      writeByteArray(os, localEncoder.regionValueToBytes(localRegion, kv._1))
      writeByteArray(os, kv._2)
      start += 1
    }
  }

  def close() {
    regions.values().forEach((x: Region) => x.close())
  }

  // def deletePartition(partitionId: Int): Unit = {
  //   require(deletedPartitions != null)
  //   require(0 <= partitionId && partitionId < outPartitions, partitionId.toString)
  //   deletedPartitions.remove(partitionId)
  // }

  // def allPartitionsDeleted(): Boolean = deletedPartitions.isEmpty
}
