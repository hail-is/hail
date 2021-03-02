package is.hail.backend.service

import java.io._
import java.net._
import java.nio.charset.StandardCharsets
import java.util.concurrent._

import is.hail.HailContext
import is.hail.annotations._
import is.hail.asm4s._
import is.hail.backend.{Backend, BackendContext, BroadcastValue, HailTaskContext}
import is.hail.expr.JSONAnnotationImpex
import is.hail.expr.ir.lowering.{DArrayLowering, LowerDistributedSort, LoweringPipeline, TableStage, TableStageDependency}
import is.hail.expr.ir.{Compile, ExecuteContext, IR, IRParser, Literal, MakeArray, MakeTuple, OwningTempFileManager, ShuffleRead, ShuffleWrite, SortField, ToStream}
import is.hail.io.fs.GoogleStorageFS
import is.hail.linalg.BlockMatrix
import is.hail.rvd.RVDPartitioner
import is.hail.services._
import is.hail.services.batch_client.BatchClient
import is.hail.services.shuffler.ShuffleClient
import is.hail.types._
import is.hail.types.encoded._
import is.hail.types.physical._
import is.hail.types.virtual._
import is.hail.utils.{log => donotuseme, _}
import is.hail.variant.ReferenceGenome
import org.apache.commons.io.IOUtils
import org.apache.log4j.Logger
import org.apache.spark.sql.Row
import org.json4s.JsonAST._
import org.json4s.jackson.JsonMethods
import org.json4s.{DefaultFormats, Formats}
import org.newsclub.net.unix.{AFUNIXSocket, AFUNIXSocketAddress, AFUNIXServerSocket}


import scala.collection.mutable
import scala.reflect.ClassTag
import scala.annotation.switch

class ServiceTaskContext(val partitionId: Int) extends HailTaskContext {
  override type BackendType = ServiceBackend

  override def stageId(): Int = 0

  override def attemptNumber(): Int = 0
}

object WorkerTimer {
  private val log = Logger.getLogger(getClass.getName())
}

class WorkerTimer() {
  import WorkerTimer._

  var startTimes: mutable.Map[String, Long] = mutable.Map()
  def start(label: String): Unit = {
    startTimes.put(label, System.nanoTime())
  }

  def end(label: String): Unit = {
    val endTime = System.nanoTime()
    val startTime = startTimes.get(label)
    startTime.foreach { s =>
      val durationMS = "%.6f".format((endTime - s).toDouble / 1000000.0)
      log.info(s"$label took $durationMS ms.")
    }
  }
}

object Worker {
  private val log = Logger.getLogger(getClass.getName())

  def main(args: Array[String]): Unit = {
    if (args.length != 2) {
      throw new IllegalArgumentException(s"expected two arguments, not: ${ args.length }")
    }
    val root = args(0)
    val i = args(1).toInt
    val timer = new WorkerTimer()

    var scratchDir = System.getenv("HAIL_WORKER_SCRATCH_DIR")
    if (scratchDir == null)
      scratchDir = ""

    log.info(s"running job $i at root $root wih scratch directory '$scratchDir'")

    timer.start(s"Job $i")
    timer.start("readInputs")

    val fs = retryTransientErrors {
      using(new FileInputStream(s"$scratchDir/gsa-key/key.json")) { is =>
        new GoogleStorageFS(IOUtils.toString(is))
      }
    }

    val f = retryTransientErrors {
      using(new ObjectInputStream(fs.openNoCompression(s"$root/f"))) { is =>
        is.readObject().asInstanceOf[(Array[Byte], HailTaskContext) => Array[Byte]]
      }
    }

    var offset = 0L
    var length = 0

    retryTransientErrors {
      using(fs.openNoCompression(s"$root/context.offsets")) { is =>
        is.seek(i * 12)
        offset = is.readLong()
        length = is.readInt()
      }
    }

    val context = retryTransientErrors {
      using(fs.openNoCompression(s"$root/contexts")) { is =>
        is.seek(offset)
        val context = new Array[Byte](length)
        is.readFully(context)
        context
      }
    }
    timer.end("readInputs")
    timer.start("executeFunction")

    val hailContext = HailContext(
      ServiceBackend(), skipLoggingConfiguration = true, quiet = true)
    val htc = new ServiceTaskContext(i)
    HailTaskContext.setTaskContext(htc)
    val result = f(context, htc)
    HailTaskContext.finish()

    timer.end("executeFunction")
    timer.start("writeOutputs")

    using(fs.createNoCompression(s"$root/result.$i")) { os =>
      os.write(result)
    }
    timer.end("writeOutputs")
    timer.end(s"Job $i")
    log.info(s"finished job $i at root $root")
  }
}

class ServiceBackendContext(
  val username: String,
  @transient val sessionID: String,
  val billingProject: String,
  val bucket: String
) extends BackendContext {
  def tokens(): Tokens =
    new Tokens(Map((DeployConfig.get.defaultNamespace, sessionID)))
}

object ServiceBackend {
  private val log = Logger.getLogger(getClass.getName())

  def apply(): ServiceBackend = {
    new ServiceBackend()
  }
}

class User(
  val username: String,
  val tmpdir: String,
  val fs: GoogleStorageFS)

class ServiceBackend() extends Backend {
  import ServiceBackend.log

  private[this] val users = new ConcurrentHashMap[String, User]()

  def addUser(username: String, key: String): Unit = synchronized {
    val previous = users.put(username, new User(username, "/tmp", new GoogleStorageFS(key)))
    assert(previous == null)
  }

  def userContext[T](username: String, timer: ExecutionTimer)(f: (ExecuteContext) => T): T = {
    val user = users.get(username)
    assert(user != null, username)
    ExecuteContext.scoped(user.tmpdir, "file:///tmp", this, user.fs, timer, null)(f)
  }

  def defaultParallelism: Int = 10

  def broadcast[T: ClassTag](_value: T): BroadcastValue[T] = new BroadcastValue[T] with Serializable {
    def value: T = _value
  }

  def parallelizeAndComputeWithIndex(_backendContext: BackendContext, collection: Array[Array[Byte]], dependency: Option[TableStageDependency] = None)(f: (Array[Byte], HailTaskContext) => Array[Byte]): Array[Array[Byte]] = {
    val backendContext = _backendContext.asInstanceOf[ServiceBackendContext]

    val user = users.get(backendContext.username)
    assert(user != null, backendContext.username)
    val fs = user.fs

    val n = collection.length

    val token = tokenUrlSafe(32)

    log.info(s"parallelizeAndComputeWithIndex: nPartitions $n token $token")

    val root = s"gs://${ backendContext.bucket }/tmp/hail/query/$token"

    log.info(s"parallelizeAndComputeWithIndex: token $token: writing f")

    using(new ObjectOutputStream(fs.create(s"$root/f"))) { os =>
      os.writeObject(f)
    }

    log.info(s"parallelizeAndComputeWithIndex: token $token: writing context offsets")

    using(fs.createNoCompression(s"$root/context.offsets")) { os =>
      var o = 0L
      var i = 0
      while (i < n) {
        val len = collection(i).length
        os.writeLong(o)
        os.writeInt(len)
        i += 1
        o += len
      }
    }

    log.info(s"parallelizeAndComputeWithIndex: token $token: writing contexts")

    using(fs.createNoCompression(s"$root/contexts")) { os =>
      collection.foreach { context =>
        os.write(context)
      }
    }

    val jobs = new Array[JObject](n)
    var i = 0
    while (i < n) {
      jobs(i) = JObject(
          "always_run" -> JBool(false),
          "job_id" -> JInt(i),
          "parent_ids" -> JArray(List()),
          "process" -> JObject(
            "command" -> JArray(List(
              JString("is.hail.backend.service.Worker"),
              JString(root),
              JString(s"$i"))),
            "type" -> JString("jvm")),
          "mount_tokens" -> JBool(true))
      i += 1
    }

    log.info(s"parallelizeAndComputeWithIndex: token $token: running job")

    val batchClient = BatchClient.fromSessionID(backendContext.sessionID)
    val batch = batchClient.run(
      JObject(
        "billing_project" -> JString(backendContext.billingProject),
        "n_jobs" -> JInt(n),
        "token" -> JString(token)),
      jobs)
    implicit val formats: Formats = DefaultFormats
    val batchID = (batch \ "id").extract[Int]
    val batchState = (batch \ "state").extract[String]
    if (batchState != "success")
      throw new RuntimeException(s"batch $batchID failed: $batchState")

    log.info(s"parallelizeAndComputeWithIndex: token $token: reading results")

    val r = new Array[Array[Byte]](n)
    i = 0  // reusing
    while (i < n) {
      r(i) = using(fs.openNoCompression(s"$root/result.$i")) { is =>
        IOUtils.toByteArray(is)
      }
      i += 1
    }
    r
  }

  def stop(): Unit = ()

  def valueType(username: String, s: String): String = {
    ExecutionTimer.logTime("ServiceBackend.valueType") { timer =>
      userContext(username, timer) { ctx =>
        val x = IRParser.parse_value_ir(ctx, s)
        x.typ.toString
      }
    }
  }

  def tableType(username: String, s: String): String = {
    ExecutionTimer.logTime("ServiceBackend.tableType") { timer =>
      userContext(username, timer) { ctx =>
        val x = IRParser.parse_table_ir(ctx, s)
        val t = x.typ
        val jv = JObject("global" -> JString(t.globalType.toString),
          "row" -> JString(t.rowType.toString),
          "row_key" -> JArray(t.key.map(f => JString(f)).toList))
        JsonMethods.compact(jv)
      }
    }
  }

  def matrixTableType(username: String, s: String): String = {
    ExecutionTimer.logTime("ServiceBackend.matrixTableType") { timer =>
      userContext(username, timer) { ctx =>
        val x = IRParser.parse_matrix_ir(ctx, s)
        val t = x.typ
        val jv = JObject("global" -> JString(t.globalType.toString),
          "col" -> JString(t.colType.toString),
          "col_key" -> JArray(t.colKey.map(f => JString(f)).toList),
          "row" -> JString(t.rowType.toString),
          "row_key" -> JArray(t.rowKey.map(f => JString(f)).toList),
          "entry" -> JString(t.entryType.toString))
        JsonMethods.compact(jv)
      }
    }
  }

  def blockMatrixType(username: String, s: String): String = {
    ExecutionTimer.logTime("ServiceBackend.blockMatrixType") { timer =>
      userContext(username, timer) { ctx =>
        val x = IRParser.parse_blockmatrix_ir(ctx, s)
        val t = x.typ
        val jv = JObject("element_type" -> JString(t.elementType.toString),
          "shape" -> JArray(t.shape.map(s => JInt(s)).toList),
          "is_row_vector" -> JBool(t.isRowVector),
          "block_size" -> JInt(t.blockSize))
        JsonMethods.compact(jv)
      }
    }
  }

  def referenceGenome(username: String, name: String): String = {
    ReferenceGenome.getReference(name).toJSONString
  }

  private[this] def execute(ctx: ExecuteContext, _x: IR): Option[(Annotation, PType)] = {
    val x = LoweringPipeline.darrayLowerer(true)(DArrayLowering.All).apply(ctx, _x)
      .asInstanceOf[IR]
    if (x.typ == TVoid) {
      val (_, f) = Compile[AsmFunction1RegionUnit](ctx,
        FastIndexedSeq[(String, PType)](),
        FastIndexedSeq[TypeInfo[_]](classInfo[Region]), UnitInfo,
        x,
        optimize = true)

      f(0, ctx.r)(ctx.r)
      None
    } else {
      val (pt, f) = Compile[AsmFunction1RegionLong](ctx,
        FastIndexedSeq[(String, PType)](),
        FastIndexedSeq[TypeInfo[_]](classInfo[Region]), LongInfo,
        MakeTuple.ordered(FastIndexedSeq(x)),
        optimize = true)

      val a = f(0, ctx.r)(ctx.r)
      val retPType = pt.asInstanceOf[PBaseStruct]
      Some((new UnsafeRow(retPType, ctx.r, a).get(0), retPType.types(0)))
    }
  }

  def execute(username: String, sessionID: String, billingProject: String, bucket: String, code: String, token: String): String = {
    ExecutionTimer.logTime("ServiceBackend.execute") { timer =>
      userContext(username, timer) { ctx =>
        log.info(s"executing: ${token}")
        ctx.backendContext = new ServiceBackendContext(username, sessionID, billingProject, bucket)

        execute(ctx, IRParser.parse_value_ir(ctx, code)) match {
          case Some((v, t)) =>
            JsonMethods.compact(
              JObject(List("value" -> JSONAnnotationImpex.exportAnnotation(v, t.virtualType),
                "type" -> JString(t.virtualType.toString))))
          case None =>
            JsonMethods.compact(
              JObject(List("value" -> null, "type" -> JString(TVoid.toString))))
        }
      }
    }
  }

  def flags(): String = {
    JsonMethods.compact(JObject(HailContext.get.flags.available.toArray().map { case f: String =>
      val v = HailContext.getFlag(f)
      f -> (if (v == null) JNull else JString(v))
    }: _*))
  }

  def getFlag(name: String): String = {
    val v = HailContext.getFlag(name)
    JsonMethods.compact(if (v == null) JNull else JString(v))
  }

  def setFlag(name: String, value: String): String = {
    val v = HailContext.getFlag(name)
    HailContext.setFlag(name, value)
    JsonMethods.compact(if (v == null) JNull else JString(v))
  }

  def unsetFlag(name: String): String = {
    val v = HailContext.getFlag(name)
    HailContext.setFlag(name, null)
    JsonMethods.compact(if (v == null) JNull else JString(v))
  }

  def lowerDistributedSort(
    ctx: ExecuteContext,
    stage: TableStage,
    sortFields: IndexedSeq[SortField],
    relationalLetsAbove: Map[String, IR],
    rowTypeRequiredness: RStruct
  ): TableStage = {
    val region = ctx.r
    val rowType = stage.rowType
    val keyFields = sortFields.map(_.field).toArray
    val keyType = rowType.typeAfterSelectNames(keyFields)
    val rowEType = EType.fromTypeAndAnalysis(rowType, rowTypeRequiredness).asInstanceOf[EBaseStruct]
    val keyEType = EType.fromTypeAndAnalysis(keyType, rowTypeRequiredness.select(keyFields)).asInstanceOf[EBaseStruct]
    val shuffleType = TShuffle(sortFields, rowType, rowEType, keyEType)
    val shuffleClient = new ShuffleClient(shuffleType, ctx)
    assert(keyType == shuffleClient.codecs.keyType)
    val keyDecodedPType = shuffleClient.codecs.keyDecodedPType
    shuffleClient.start()
    val uuid = shuffleClient.uuid

    ctx.ownCleanup({ () =>
      using(new ShuffleClient(shuffleType, uuid, ctx)) { shuffleClient =>
        shuffleClient.stop()
      }
    })

    try {
      val Some((successfulPartitionIdsAndGlobals, pType)) = execute(
        ctx,
        stage.mapCollectWithGlobals
          (relationalLetsAbove)
          { partition => ShuffleWrite(Literal(shuffleType, uuid), partition) }
          { (rows, globals) => MakeTuple.ordered(Seq(rows, globals)) })
      val globals = successfulPartitionIdsAndGlobals.asInstanceOf[UnsafeRow].get(1)

      val partitionBoundsPointers = shuffleClient.partitionBounds(region, stage.numPartitions)
      val partitionIntervals = partitionBoundsPointers.zip(partitionBoundsPointers.drop(1)).map { case (l, r) =>
        Interval(SafeRow(keyDecodedPType, l), SafeRow(keyDecodedPType, r), includesStart = true, includesEnd = false)
      }
      val last = partitionIntervals.last
      partitionIntervals(partitionIntervals.length - 1) = Interval(
        last.left.point, last.right.point, includesStart = true, includesEnd = true)

      val partitioner = new RVDPartitioner(keyType, partitionIntervals.toFastIndexedSeq)

      TableStage(
        globals = Literal(stage.globalType, globals),
        partitioner = partitioner,
        TableStageDependency.none,
        contexts = ToStream(MakeArray(
          partitionIntervals.map(interval => Literal(TInterval(keyType), interval)),
          TArray(TInterval(keyType)))),
        interval => ShuffleRead(Literal(shuffleType, uuid), interval))
    } finally {
      shuffleClient.close()
    }
  }

  def persist(backendContext: BackendContext, id: String, value: BlockMatrix, storageLevel: String): Unit = ???

  def unpersist(backendContext: BackendContext, id: String): Unit = ???

  def getPersistedBlockMatrix(backendContext: BackendContext, id: String): BlockMatrix = ???

  def getPersistedBlockMatrixType(backendContext: BackendContext, id: String): BlockMatrixType = ???

  def loadReferencesFromDataset(
    username: String,
    sessionID: String,
    billingProject: String,
    bucket: String,
    path: String
  ): String = {
    ExecutionTimer.logTime("ServiceBackend.loadReferencesFromDataset") { timer =>
      userContext(username, timer) { ctx =>
        ReferenceGenome.fromHailDataset(ctx.fs, path)
      }
    }
  }
}

class EndOfInputException extends RuntimeException

object ServiceBackendSocketAPI {
  private val log = Logger.getLogger(getClass.getName())
}

class ServiceBackendSocketAPI(backend: ServiceBackend, socket: Socket) extends Thread {
  import ServiceBackendSocketAPI._

  private[this] val LOAD_REFERENCES_FROM_DATASET = 1
  private[this] val VALUE_TYPE = 2
  private[this] val TABLE_TYPE = 3
  private[this] val MATRIX_TABLE_TYPE = 4
  private[this] val BLOCK_MATRIX_TYPE = 5
  private[this] val REFERENCE_GENOME = 6
  private[this] val EXECUTE = 7
  private[this] val FLAGS = 8
  private[this] val GET_FLAG = 9
  private[this] val UNSET_FLAG = 10
  private[this] val SET_FLAG = 11
  private[this] val ADD_USER = 12
  private[this] val GOODBYE = 254

  private[this] val in = socket.getInputStream
  private[this] val out = socket.getOutputStream

  private[this] val dummy = new Array[Byte](8)

  def read(bytes: Array[Byte], off: Int, n: Int): Unit = {
    assert(off + n <= bytes.length)
    var read = 0
    while (read < n) {
      val r = in.read(bytes, off + read, n - read)
      if (r < 0) {
        throw new EndOfInputException
      } else {
        read += r
      }
    }
  }

  def readInt(): Int = {
    read(dummy, 0, 4)
    Memory.loadInt(dummy, 0)
  }

  def readLong(): Long = {
    read(dummy, 0, 8)
    Memory.loadLong(dummy, 0)
  }

  def readBytes(): Array[Byte] = {
    val n = readInt()
    val bytes = new Array[Byte](n)
    read(bytes, 0, n)
    bytes
  }

  def readString(): String = new String(readBytes(), StandardCharsets.UTF_8)

  def writeBool(b: Boolean): Unit = {
    out.write(if (b) 1 else 0)
  }

  def writeInt(v: Int): Unit = {
    Memory.storeInt(dummy, 0, v)
    out.write(dummy, 0, 4)
  }

  def writeLong(v: Long): Unit = {
    Memory.storeLong(dummy, 0, v)
    out.write(dummy)
  }

  def writeBytes(bytes: Array[Byte]): Unit = {
    writeInt(bytes.length)
    out.write(bytes)
  }

  def writeString(s: String): Unit = writeBytes(s.getBytes(StandardCharsets.UTF_8))

  def eventLoop(): Unit = {
    var continue = true
    while (continue) {
      val cmd = readInt()

      (cmd: @switch) match {
        case LOAD_REFERENCES_FROM_DATASET =>
          val username = readString()
          val sessionId = readString()
          val billingProject = readString()
          val bucket = readString()
          val path = readString()
          try {
            val result = backend.loadReferencesFromDataset(username, sessionId, billingProject, bucket, path)
            writeBool(true)
            writeString(result)
          } catch {
            case t: Throwable =>
              writeBool(false)
              writeString(formatException(t))
          }

        case VALUE_TYPE =>
          val username = readString()
          val s = readString()
          try {
            val result = backend.valueType(username, s)
            writeBool(true)
            writeString(result)
          } catch {
            case t: Throwable =>
              writeBool(false)
              writeString(formatException(t))
          }

        case TABLE_TYPE =>
          val username = readString()
          val s = readString()
          try {
            val result = backend.tableType(username, s)
            writeBool(true)
            writeString(result)
          } catch {
            case t: Throwable =>
              writeBool(false)
              writeString(formatException(t))
          }

        case MATRIX_TABLE_TYPE =>
          val username = readString()
          val s = readString()
          try {
            val result = backend.matrixTableType(username, s)
            writeBool(true)
            writeString(result)
          } catch {
            case t: Throwable =>
              writeBool(false)
              writeString(formatException(t))
          }

        case BLOCK_MATRIX_TYPE =>
          val username = readString()
          val s = readString()
          try {
            val result = backend.blockMatrixType(username, s)
            writeBool(true)
            writeString(result)
          } catch {
            case t: Throwable =>
              writeBool(false)
              writeString(formatException(t))
          }

        case REFERENCE_GENOME =>
          val username = readString()
          val name = readString()
          try {
            val result = backend.referenceGenome(username, name)
            writeBool(true)
            writeString(result)
          } catch {
            case t: Throwable =>
              writeBool(false)
              writeString(formatException(t))
          }

        case EXECUTE =>
          val username = readString()
          val sessionId = readString()
          val billingProject = readString()
          val bucket = readString()
          val code = readString()
          val token = readString()
          try {
            val result = backend.execute(username, sessionId, billingProject, bucket, code, token)
            writeBool(true)
            writeString(result)
          } catch {
            case t: Throwable =>
              writeBool(false)
              writeString(formatException(t))
          }

        case FLAGS =>
          try {
            val result = backend.flags()
            writeBool(true)
            writeString(result)
          } catch {
            case t: Throwable =>
              writeBool(false)
              writeString(formatException(t))
          }

        case GET_FLAG =>
          val name = readString()
          try {
            val result = backend.getFlag(name)
            writeBool(true)
            writeString(result)
          } catch {
            case t: Throwable =>
              writeBool(false)
              writeString(formatException(t))
          }

        case SET_FLAG =>
          val name = readString()
          val value = readString()
          try {
            val result = backend.setFlag(name, value)
            writeBool(true)
            writeString(result)
          } catch {
            case t: Throwable =>
              writeBool(false)
              writeString(formatException(t))
          }

        case UNSET_FLAG =>
          val name = readString()
          try {
            val result = backend.unsetFlag(name)
            writeBool(true)
            writeString(result)
          } catch {
            case t: Throwable =>
              writeBool(false)
              writeString(formatException(t))
          }

        case ADD_USER =>
          val name = readString()
          val gsaKey = readString()
          try {
            val result = backend.addUser(name, gsaKey)
            writeBool(true)
          } catch {
            case t: Throwable =>
              writeBool(false)
              writeString(formatException(t))
          }

        case GOODBYE =>
          continue = false
          writeInt(GOODBYE)
      }
    }
  }

  override def run(): Unit = {
    try {
      eventLoop()
    } catch {
      case t: Throwable =>
        log.info("ServiceBackendSocketAPI caught exception", t)
    } finally {
      socket.close()
    }
  }
}

object ServiceBackendMain {
  private val log = Logger.getLogger(getClass.getName())

  def main(argv: Array[String]): Unit = {
    assert(argv.length == 1, argv.toFastIndexedSeq)
    val udsAddress = argv(0)
    val executor = Executors.newCachedThreadPool()
    val backend = new ServiceBackend()
    HailContext(backend, "hail.log", false, false, 50, skipLoggingConfiguration = true, 3)

    val ss = AFUNIXServerSocket.newInstance()
    ss.bind(new AFUNIXSocketAddress(new File(udsAddress)))
    try {
      log.info(s"serving on ${udsAddress}")
      while (true) {
        val sock = ss.accept()
        try {
          log.info(s"accepted")
          executor.execute(new ServiceBackendSocketAPI(backend, sock))
        } catch {
          case e: SocketException => {
            log.info(s"exception while handing socket to thread", e)
            sock.close()
          }
        }
      }
    } catch {
      case se: SocketException =>
        fatal("unexpected closed server socket", se)
    }
  }
}
