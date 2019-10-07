package is.hail.shuffler

import com.google.api.gax.retrying.RetrySettings
import com.google.auth.oauth2.{ AccessToken, GoogleCredentials }
import is.hail.HailContext
import is.hail.annotations.{ Region, RegionValue, RegionValueBuilder, SafeRow, UnsafeRow }
import is.hail.cxx
import is.hail.expr.ir.ExecuteContext
import is.hail.expr.types.physical._
import is.hail.io._
import is.hail.rvd.{ RVD, RVDContext, RVDPartitioner, RVDType }
import is.hail.sparkextras.ContextRDD
import is.hail.table.{ Ascending, SortOrder }
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
import org.threeten.bp.Duration
import scala.collection.mutable.{ ArrayBuffer }
import scala.concurrent.ExecutionContext
import scala.reflect.ClassTag
import com.google.cloud.storage._


object ShuffleClient {
  private[this] def typeToParsableUTF8Bytes(t: PType): Array[Byte] =
    ByteUtils.stringToBytes(t.parsableString())

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
    val bucketName = "danking"
    val shufflePath = "shuffles/123"
    val storageUrl = s"gs://$bucketName/$shufflePath"
    val keyFieldIndices = sortFieldsAndOrders.map(_._1)
    val newType = rvd.typ.copy(key =
      keyFieldIndices.map(rvd.typ.rowType.fieldNames))
    val keyPType = newType.kType.setRequired(true).asInstanceOf[PStruct]
    val parts = rvd.getNumPartitions
    val bufferSpec = ShuffleClient.activeShuffleBufferSpec
    val shuffler = ShuffleClient(
      serverUrl,
      storageUrl,
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

    val addRdd = rvd.crdd.cmapPartitions { (ctx, it) =>
      val context = TaskContext.get
      val partitionId = context.partitionId()
      val taskAttemptId = context.taskAttemptId()
      val rvb = new RegionValueBuilder(ctx.region)
      val shufflerPartition = shuffler.startPartition(partitionId, taskAttemptId)
      val kBae = new ByteArrayEncoder(kEnc)
      val vBae = new ByteArrayEncoder(vEnc)
      val path = s"${storageUrl}/${partitionId}-${taskAttemptId}"
      fs.writeFileNoCompression(path) { out =>
        using(vEnc(out)) { enc =>
          it.foreach { rv =>
            rvb.start(keyPType)
            // unsafe version of selectRegionValue, for speed reasons
            rvb.startStruct()
            rvb.addFields(rowPType, rv.region, rv.offset, keyFieldIndices)
            rvb.endStruct()
            val sortKeyOff = rvb.end()
            val sortKeyBytes = kBae.regionValueToBytes(rv.region, sortKeyOff)
            val rowBytes = new Array[Byte](28)
            var off = 0
            off = ByteUtils.writeInt(rowBytes, off, partitionId)
            off = ByteUtils.writeLong(rowBytes, off, taskAttemptId)
            val start = out.getPos
            off = ByteUtils.writeLong(rowBytes, off, start)
            enc.writeRegionValue(rv.region, rv.offset)
            enc.flush()
            off = ByteUtils.writeLong(rowBytes, off, out.getPos - start)
            ctx.region.clear()
            shufflerPartition.add(sortKeyBytes, rowBytes)
          }
        }
      }
      shufflerPartition.finishPartition()
      Iterator.empty
    }.clearingRun
    sc.runJob(addRdd, (it: Iterator[Unit]) => it.foreach(_ => ()), (_, _: Unit) => ())
    val keyBytes = shuffler.end_input()
    // val readParallelism = hc.flags.get("shuffle_read_parallelism").toInt
    val shuffledCRDD = ContextRDD.weaken[RVDContext](sc.parallelize((0 until parts), parts)).cflatMap { (ctx, _) =>
      val accessToken = GoogleCredentials.getApplicationDefault
        // https://cloud.google.com/storage/docs/authentication
        .createScoped("https://www.googleapis.com/auth/devstorage.read_write")
        .refreshAccessToken
        .getTokenValue
      val authorizationHeader = s"Bearer ${accessToken}"
      // val asyncPool = new AsyncPool(readParallelism)
      val futureBytes = shuffler.get(TaskContext.get.partitionId).map { bytes =>
        val bais = new ByteArrayInputStream(bytes)
        val sourcePartitionId = ByteUtils.readInt(bais)
        val sourceAttemptId = ByteUtils.readLong(bais)
        val position = ByteUtils.readLong(bais)
        val length = ByteUtils.readLong(bais)
        val path = s"${shufflePath}/${sourcePartitionId}-${sourceAttemptId}"
        ShuffleAsyncPool.pool.get.future { () =>
          HTTPClient.get(
            s"https://${bucketName}.storage.googleapis.com/${path}",
            Map(
              "Authorization" -> authorizationHeader,
              "Range" -> s"bytes=$position-${position+length}"
            ), { is =>
              val a = new Array[Byte](length)
              is.readRepeatedly(a)
              a
            })
        }
      }.toArray

      val bais = new RestartableByteArrayInputStream()
      val dec = vDec(bais)
      val rv = RegionValue()
      futureBytes.iterator.map { fut =>
        rv.setRegion(ctx.region)
        bais.restart(fut.get)
        rv.setOffset(dec.readRegionValue(ctx.region))
        rv
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
    storageUrl: String,
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
        assert(partitioner.kType.physicalType.setRequired(true) == keyPType)
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
        // HailContext.get.sFS.delete(storageUrl, recursive = true)
      }
    }))
    executeContext.addOnExit { () =>
      HTTPClient.delete(shuffleSpecificApiUrl)
      // HailContext.get.sFS.delete(storageUrl, recursive = true)
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

  // def cleanUpResourcesWithGCOf[T: ClassTag](rdd: RDD[T]): RDD[T] = {
  //   val localShuffleSpecificApiUrl = shuffleSpecificApiUrl
  //   // Runtime.getRuntime.addShutdownHook(new Thread(new Runnable() {
  //   //   override def run(): Unit = {
  //   //     log.info(s"shutdown hook $rdd")
  //   //     HTTPClient.delete(localShuffleSpecificApiUrl)
  //   //   }
  //   // }))
  //   new RDD[T](rdd) {
  //     override def getPartitions: Array[Partition] =
  //       firstParent[T].partitions
  //     override def compute(split: Partition, context: TaskContext): Iterator[T] =
  //       firstParent[T].iterator(split, context)
  //     override def getPreferredLocations(partition: Partition): Seq[String] =
  //       firstParent[T].preferredLocations(partition)
  //     override def finalize(): Unit = {
  //       log.info(s"finalizing $rdd")
  //       log.info(Thread.currentThread().getStackTrace.mkString(", "))
  //       log.info(Thread.currentThread().getId().toString)
  //       log.info(Thread.currentThread().getName())
  //       HTTPClient.delete(localShuffleSpecificApiUrl)
  //     }
  //   }
  // }

  // FIXME: when can I delete a shuffle?
  // def close(partition: Int): Unit = HTTPClient.delete(shuffleSpecificApiUrl + "/" + partition)
}

