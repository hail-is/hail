package is.hail.shuffler
import is.hail.HailContext
import is.hail.annotations.{ Region, RegionValue, RegionValueBuilder, SafeRow, UnsafeRow }
import is.hail.cxx
import is.hail.expr.ir.ExecuteContext
import is.hail.expr.types.physical._
import is.hail.io._
import is.hail.rvd.{ RVD, RVDContext, RVDPartitioner, RVDType }
import is.hail.sparkextras.ContextRDD
import is.hail.expr.ir.{ Ascending, SortOrder }
import is.hail.utils._
import is.hail.utils.HTTPClient
import java.io.{ BufferedReader, ByteArrayInputStream, ByteArrayOutputStream, InputStreamReader }
import java.net.{ URL, HttpURLConnection }
import java.io.{ InputStream, OutputStream }
import java.nio.ByteBuffer
import java.nio.channels.Channels
import java.nio.charset.{ StandardCharsets }
import java.util.HashMap
import java.util.concurrent.{ Callable, Executor, Executors, ThreadPoolExecutor, TimeUnit }
import org.apache.hadoop.fs.FSDataInputStream
import org.apache.spark.{ Dependency, Partition, TaskContext }
import org.apache.spark.rdd.RDD
import org.json4s.jackson.JsonMethods
import scala.collection.mutable.{ ArrayBuffer }
import scala.concurrent.ExecutionContext
import scala.reflect.ClassTag
import org.json4s._
import org.json4s.jackson.{JsonMethods, Serialization}


object ShuffleClient {
  private implicit val f = new DefaultFormats() {}

  private[this] def typeToParsableUTF8Bytes(t: PType): Array[Byte] =
    ByteUtils.stringToBytes(t.toString())

  private[this] def sortOrdersToParseableBytes(so: Array[SortOrder]): Array[Byte] =
    so.map(_.serialize)

  def activeShuffleBufferSpec = BufferSpec.parseOrDefault(
    HailContext.get.flags.get("shuffle_buffer_spec"),
    BufferSpec.defaultUncompressed)

  def shuffle(
    serverUrl: String,
    rowPType: PBaseStruct,
    sortFieldsAndOrders: Array[(Int, SortOrder)],
    rvd: RVD,
    outPartitioning: Either[Int, RVDPartitioner],
    executeContext: ExecuteContext,
    bufferSpec: BufferSpec = activeShuffleBufferSpec,
    bufferByteSize: Int = 1 * 1024 * 1024,
    httpChunkSize: Int = 64 * 1024
  ): RVD = {
    val keyFieldIndices = sortFieldsAndOrders.map(_._1)
    val newType = rvd.typ.copy(key =
      keyFieldIndices.map(rvd.typ.rowType.fieldNames))
    val keyPType = newType.kType.setRequired(true).asInstanceOf[PStruct]
    val parts = rvd.getNumPartitions
    val bufferSpec = ShuffleClient.activeShuffleBufferSpec
    val shuffler = ShuffleClient(
      serverUrl,
      rowPType,
      keyPType,
      sortFieldsAndOrders,
      parts,
      outPartitioning,
      executeContext,
      bufferSpec)

    val keyCodec = new Codec(TypedCodecSpec(keyPType, bufferSpec))
    val kEnc = keyCodec.buildEncoder
    val kDec = keyCodec.buildDecoder
    val decodedKeyPType = keyCodec.memType.asInstanceOf[PStruct]

    val valueCodec = new Codec(TypedCodecSpec(rowPType, bufferSpec))
    val vEnc = valueCodec.buildEncoder
    val vDec = valueCodec.buildDecoder
    val decodedRowPType = valueCodec.memType.asInstanceOf[PStruct]

    val hc = HailContext.get
    val sc = hc.sc
    val fs = hc.sFS

    val buffer = BufferClient.create(executeContext)

    val bufferId = buffer.id
    val _bufferWorkers = buffer.getWorkers
    val leader = _bufferWorkers(0)
    val bufferWorkers = _bufferWorkers.map(x =>
      buffer.rootUrl.replace(leader, x)
    )
    val BUFSIZE = 10 * 1024 * 1024

    val addRdd = rvd.crdd.cmapPartitions { (ctx, it) =>
      val context = TaskContext.get
      val partitionId = context.partitionId()
      val taskAttemptId = context.taskAttemptId()
      val rvb = new RegionValueBuilder(ctx.region)
      val shufflerPartition = shuffler.startPartition(partitionId, taskAttemptId)
      val kBae = new ByteArrayEncoder(kEnc)
      val buffer = new BufferClient(
        bufferWorkers(context.partitionId() % bufferWorkers.length),
        bufferId)

      val os = new ByteArrayOutputStream(BUFSIZE)
      val blocks = new ArrayBuilder[(Int, Int)]()
      val keys = new ArrayBuilder[Array[Byte]]()

      using(vEnc(os)) { enc =>
        it.foreach { rv =>
          val pos = os.size()
          enc.writeRegionValue(rv.region, rv.offset)
          enc.flush()
          blocks += ((pos, os.size() - pos))
          rvb.start(keyPType)
          rvb.startStruct()
          rvb.addFields(rowPType, rv.region, rv.offset, keyFieldIndices)
          rvb.endStruct()
          val sortKeyOff = rvb.end()
          keys += kBae.regionValueToBytes(rv.region, sortKeyOff)
          ctx.region.clear()
          if (os.size() > BUFSIZE) {
            assert(os.size() < 50 * 1024 * 1024)
            val ks = keys.result()
            val bs = blocks.result()
            val ba = os.toByteArray()
            val (s, fileId, pos, n) = buffer.write(httpos => httpos.write(ba))
            log.info(s"wrote ${ba.length} bytes to buffer (${pos} ${n})")
            var i = 0
            while (i < ks.length) {
              val key = ks(i)
              val (off, n) = bs(i)
              val baos = new ByteArrayOutputStream()
              Serialization.write(Array(s, fileId, pos + off, n), baos)
              shufflerPartition.add(key, baos.toByteArray())
              i += 1
            }
            blocks.clear()
            keys.clear()
            os.reset()
            log.info(s"wrote ${i} keys")
          }
        }
        if (os.size() > 0) {
          assert(os.size() < 50 * 1024 * 1024)
          val ks = keys.result()
          val bs = blocks.result()
          val ba = os.toByteArray()
          val (s, fileId, pos, n) = buffer.write(httpos => httpos.write(ba))
          log.info(s"wrote ${ba.length} bytes to buffer (${pos} ${n})")
          var i = 0
          while (i < ks.length) {
            val key = ks(i)
            val (off, n) = bs(i)
            val baos = new ByteArrayOutputStream()
            Serialization.write(Array(s, fileId, pos + off, n), baos)
            shufflerPartition.add(key, baos.toByteArray())
            i += 1
            blocks.clear()
            keys.clear()
            os.reset()
          }
          log.info(s"wrote ${i} keys")
        }
      }
      shufflerPartition.finishPartition()
      Iterator.empty
    }.clearingRun
    sc.runJob(addRdd, (it: Iterator[Unit]) => it.foreach(_ => ()), (_, _: Unit) => ())
    val keyBytes = shuffler.end_input()
    val shuffledCRDD = ContextRDD.weaken[RVDContext](sc.parallelize((0 until parts), parts)).cflatMap { (ctx, _) =>
      val bufferKeys = shuffler.get(TaskContext.get.partitionId).map { bytes =>
        val bais = new ByteArrayInputStream(bytes)
        val JArray(List(JString(s), JInt(fileId), JInt(pos), JInt(n))) = JsonMethods.parse(bais)
        (s, fileId.toInt, pos.toInt, n.toInt)
      }

      val keyGroups = new ArrayBuilder[Array[BufferClient.Key]]()
      var i = 0
      while (i < bufferKeys.length) {
        var s = 0
        val group = new ArrayBuilder[BufferClient.Key]()
        assert(bufferKeys(i)._4 < BUFSIZE)
        while (i < bufferKeys.length && s < BUFSIZE) {
          s += bufferKeys(i)._4
          group += bufferKeys(i)
          i += 1
        }
        keyGroups += group.result()
      }

      val rv = RegionValue(ctx.region)
      val r = Region()
      r.addReferenceTo(ctx.region)
      keyGroups.result().iterator.flatMap { keys =>
        buffer.readMany(keys, { in =>
          vDec(in).readRegionValue(r)
        }).iterator.map { (off: Long) =>
          rv.setOffset(off)
          rv
        }
      }
    }
    val decodedKeys = Region.scoped { r =>
      RegionValue.pointerFromBytes(kDec, r, keyBytes.iterator)
        .map(SafeRow.read(decodedKeyPType, r, _))
        .toArray
    }
    val partitionIntervals = new Array[Interval](parts)
    var i = 0
    while (i < partitionIntervals.length) {
      partitionIntervals(i) = Interval(
        decodedKeys(i),
        decodedKeys(i + 1),
        true,
        i + 1 != partitionIntervals.length)
      i += 1
    }
    RVD(
      RVDType(decodedRowPType, newType.key),
      new RVDPartitioner(
        decodedKeyPType.virtualType,
        partitionIntervals),
      shuffledCRDD)
  }

  def apply(
    serverUrl: String,
    rowPType: PBaseStruct,
    keyPType: PBaseStruct,
    sortFieldsAndOrders: Array[(Int, SortOrder)],
    inPartitions: Int,
    outPartitioning: Either[Int, RVDPartitioner],
    executeContext: ExecuteContext,
    bufferSpec: BufferSpec = activeShuffleBufferSpec,
    bufferByteSize: Int = 1 * 1024 * 1024,
    httpChunkSize: Int = 64 * 1024
  ): ShuffleClient = {
    val apiUrl = serverUrl + "/api/v1alpha"
    log.info(s"shuffling bufferSpec: ${bufferSpec.toString}")

    val s = bufferSpec.toString
    val bufferSpecBytes = ByteUtils.stringToBytes(s)
    val typeBytes = typeToParsableUTF8Bytes(keyPType)
    val sortOrderBytes = sortOrdersToParseableBytes(sortFieldsAndOrders.map(_._2))
    val (partitioningDescriminatorByte, partitioningBytes, nOutPartitions) = outPartitioning match {
      case Left(nOutPartitions) =>
        val bytes = new Array[Byte](4)
        ByteUtils.writeInt(bytes, 0, nOutPartitions)
        (0.toByte, bytes, nOutPartitions)
      case Right(partitioner) =>
        assert(keyPType.virtualType == partitioner.kType.setRequired(true))
        val typ = PArray(PInterval(keyPType, true), true)
        val codec = new Codec(TypedCodecSpec(typ, bufferSpec))
        val enc = new ByteArrayEncoder(codec.buildEncoder)
        Region.scoped { r =>
          val rvb = new RegionValueBuilder(r)
          val bounds = partitioner.rangeBounds
          log.info(s"bounds length: ${bounds.length}")
          rvb.start(typ)
          rvb.addAnnotation(typ.virtualType, bounds.toFastIndexedSeq)
          (1.toByte, enc.regionValueToBytes(r, rvb.end()), bounds.length)
        }
    }

    val id = HTTPClient.post(
      apiUrl,
      4 + 4 + bufferSpecBytes.length +
        + 4 + typeBytes.length
        + sortOrderBytes.length +
        + 1 + 4 + partitioningBytes.length,
      { out =>
        ByteUtils.writeInt(out, inPartitions)
        ByteUtils.writeByteArray(out, bufferSpecBytes)
        ByteUtils.writeByteArray(out, typeBytes)
        out.write(sortOrderBytes)
        out.write(partitioningDescriminatorByte)
        ByteUtils.writeByteArray(out, partitioningBytes)
        out.flush()
      }, { in =>
        using(new BufferedReader(new InputStreamReader(in))) { br =>
          val id = java.lang.Long.parseLong(br.readLine())
          assert(in.read() == -1)
          id
        }
      })
    // FIXME: do we need two shuffles? only reason to pass outPartitions through
    // is so we can get the list of keys from the shuffler after we're done, but
    // if we tell the shuffler the partitioner, there's no need for it to tell
    // us the key bounds, we already know.
    val shuffleSpecificApiUrl = apiUrl + "/" + id
    val client = new ShuffleClient(shuffleSpecificApiUrl, bufferByteSize, httpChunkSize, nOutPartitions, bufferSpec)
    Runtime.getRuntime.addShutdownHook(new Thread(new Runnable() {
      override def run(): Unit = {
        log.info(s"shutdown hook shuffle $id")
        HTTPClient.delete(shuffleSpecificApiUrl)
      }
    }))
    executeContext.addOnExit { () =>
      HTTPClient.delete(shuffleSpecificApiUrl)
    }
    client
  }
}

class ShuffleClient private (
  shuffleSpecificApiUrl: String,
  bufferByteSize: Int,
  httpChunkSize: Int,
  private[this] val outPartitions: Int,
  private[this] val bufferSpec: BufferSpec
) extends Serializable {
  import ShuffleClient._
  def startPartition(
    partitionId: Int,
    attemptId: Long
  ): ShuffleClientPartition = new ShuffleClientPartition(partitionId, attemptId)

  class ShuffleClientPartition(
    private[this] val partitionId: Int,
    private[this] val attemptId: Long
  ) {
    private[this] var count: Int = 0
    private[this] var addBuffer: Array[Byte] = new Array[Byte](bufferByteSize)
    private[this] var bufferIndex: Int = 0

    private[this] def flushBuffer(): Unit = {
      HTTPClient.post(
        shuffleSpecificApiUrl,
        bufferIndex,
        { out =>
          ByteUtils.writeInt(out, partitionId)
          ByteUtils.writeLong(out, attemptId)
          ByteUtils.writeInt(out, count)
          out.write(addBuffer, 0, bufferIndex)
          out.flush()
        },
        _ => (),
        httpChunkSize)
      bufferIndex = 0
      count = 0
    }

    def add(
      key: Array[Byte],
      value: Array[Byte]
    ): Unit = {
      if (bufferIndex + 4 + key.size + 4 + value.size >= bufferByteSize) {
        flushBuffer()
      }
      bufferIndex = ByteUtils.writeByteArray(addBuffer, bufferIndex, key)
      bufferIndex = ByteUtils.writeByteArray(addBuffer, bufferIndex, value)
      count += 1
    }

    def finishPartition(): Unit = {
      if (bufferIndex != 0) {
        flushBuffer()
      }
      HTTPClient.post(shuffleSpecificApiUrl + "/finish_partition",
        4 + 8,
        { out =>
          ByteUtils.writeInt(out, partitionId)
          ByteUtils.writeLong(out, attemptId)
        })
    }
  }

  def end_input(): Array[Array[Byte]] = HTTPClient.post(shuffleSpecificApiUrl + "/close", 0, _ => (), { in =>
    val keys = new Array[Array[Byte]](outPartitions + 1)
    var i = 0
    while (i < keys.length) {
      keys(i) = ByteUtils.readByteArray(in)
      i += 1
    }
    keys
  })

  def get(
    partition: Int
  ): Array[Array[Byte]] = HTTPClient.get(shuffleSpecificApiUrl + "/" + partition,
    { in =>
      val elementsLength = ByteUtils.readInt(in)
      val values = new Array[Array[Byte]](elementsLength)
      var elementIndex = 0
      while (elementIndex < elementsLength) {
        // FIXME: should I stop sending the key?
        ByteUtils.skipByteArray(in)
        values(elementIndex) = ByteUtils.readByteArray(in)
        elementIndex += 1
      }
      log.info(s"got $elementsLength elements")
      values
    })
}

