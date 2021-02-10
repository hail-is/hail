package is.hail.backend.service

import java.io._

import is.hail.HailContext
import is.hail.annotations.{Annotation, Region, SafeRow, UnsafeRow}
import is.hail.asm4s._
import is.hail.backend.{Backend, BackendContext, BroadcastValue, HailTaskContext}
import is.hail.expr.JSONAnnotationImpex
import is.hail.expr.ir.lowering.{DArrayLowering, LowerDistributedSort, LoweringPipeline, TableStage, TableStageDependency}
import is.hail.expr.ir.{Compile, ExecuteContext, IR, IRParser, Literal, MakeArray, MakeTuple, ShuffleRead, ShuffleWrite, SortField, ToStream}
import is.hail.io.fs.GoogleStorageFS
import is.hail.linalg.BlockMatrix
import is.hail.rvd.RVDPartitioner
import is.hail.services.{DeployConfig, Tokens}
import is.hail.services.batch_client.BatchClient
import is.hail.services.shuffler.ShuffleClient
import is.hail.types._
import is.hail.types.encoded.{EBaseStruct, EType}
import is.hail.types.physical.{PBaseStruct, PType}
import is.hail.types.virtual.{TArray, TInt64, TInterval, TShuffle, TStruct, Type}
import is.hail.utils._
import is.hail.variant.ReferenceGenome
import org.apache.commons.io.IOUtils
import org.apache.log4j.LogManager
import org.apache.spark.sql.Row
import org.json4s.JsonAST._
import org.json4s.jackson.JsonMethods
import org.json4s.{DefaultFormats, Formats}

import scala.collection.mutable
import scala.reflect.ClassTag

class ServiceTaskContext(val partitionId: Int) extends HailTaskContext {
  override type BackendType = ServiceBackend

  override def stageId(): Int = 0

  override def attemptNumber(): Int = 0
}

object Worker {
  def main(args: Array[String]): Unit = {
    if (args.length != 2) {
      throw new IllegalArgumentException(s"expected two arguments, not: ${ args.length }")
    }
    val root = args(0)
    val i = args(1).toInt

    var scratchDir = System.getenv("HAIL_WORKER_SCRATCH_DIR")
    if (scratchDir == null)
      scratchDir = ""

    log.info(s"running job $i at root $root wih scratch directory '$scratchDir'")

    val fs = using(new FileInputStream(s"$scratchDir/gsa-key/key.json")) { is =>
      new GoogleStorageFS(IOUtils.toString(is))
    }

    val f = using(new ObjectInputStream(fs.openNoCompression(s"$root/f"))) { is =>
      is.readObject().asInstanceOf[(Array[Byte], HailTaskContext) => Array[Byte]]
    }

    var offset = 0L
    var length = 0

    using(fs.openNoCompression(s"$root/context.offsets")) { is =>
      is.seek(i * 12)
      offset = is.readLong()
      length = is.readInt()
    }

    val context = using(fs.openNoCompression(s"$root/contexts")) { is =>
      is.seek(offset)
      val context = new Array[Byte](length)
      is.readFully(context)
      context
    }

    val htc = new ServiceTaskContext(i)
    HailTaskContext.setTaskContext(htc)
    val result = f(context, htc)
    HailTaskContext.finish()

    using(fs.createNoCompression(s"$root/result.$i")) { os =>
      os.write(result)
    }
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
  lazy val log = LogManager.getLogger("is.hail.backend.service.ServiceBackend")

  def apply(): ServiceBackend = {
    new ServiceBackend()
  }
}

class User(
  val username: String,
  val tmpdir: String,
  val fs: GoogleStorageFS)

final class Response(val status: Int, val value: String)

class ServiceBackend() extends Backend {
  import ServiceBackend.log

  private[this] val workerImage = System.getenv("HAIL_QUERY_WORKER_IMAGE")

  private[this] val users = mutable.Map[String, User]()

  def addUser(username: String, key: String): Unit = {
    assert(!users.contains(username))
    users += username -> new User(username, "/tmp", new GoogleStorageFS(key))
  }

  def removeUser(username: String): Unit = {
    assert(users.contains(username))
    users -= username
  }

  def userContext[T](username: String, timer: ExecutionTimer)(f: (ExecuteContext) => T): T = {
    val user = users(username)
    ExecuteContext.scoped(user.tmpdir, "file:///tmp", this, user.fs, timer)(f)
  }

  def defaultParallelism: Int = 10

  def broadcast[T: ClassTag](_value: T): BroadcastValue[T] = new BroadcastValue[T] with Serializable {
    def value: T = _value
  }

  def parallelizeAndComputeWithIndex(_backendContext: BackendContext, collection: Array[Array[Byte]], dependency: Option[TableStageDependency] = None)(f: (Array[Byte], HailTaskContext) => Array[Byte]): Array[Array[Byte]] = {
    val backendContext = _backendContext.asInstanceOf[ServiceBackendContext]

    val user = users(backendContext.username)
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

  def formatException(e: Exception): String = {
    using(new StringWriter()) { sw =>
      using(new PrintWriter(sw)) { pw =>
        e.printStackTrace(pw)
        sw.toString
      }
    }
  }

  def statusForException(f: => String): Response = {
    try {
      new Response(200, f)
    } catch {
      case e: HailException =>
        new Response(400, formatException(e))
      case e: Exception =>
        new Response(500, formatException(e))
    }
  }

  def valueType(username: String, s: String): Response = {
    statusForException {
      ExecutionTimer.logTime("ServiceBackend.valueType") { timer =>
        userContext(username, timer) { ctx =>
          val x = IRParser.parse_value_ir(ctx, s)
          x.typ.toString
        }
      }
    }
  }

  def tableType(username: String, s: String): Response = {
    statusForException {
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
  }

  def matrixTableType(username: String, s: String): Response = {
    statusForException {
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
  }

  def blockMatrixType(username: String, s: String): Response = {
    statusForException {
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
  }

  def referenceGenome(username: String, name: String): Response = {
    statusForException {
      ReferenceGenome.getReference(name).toJSONString
    }
  }

  private[this] def execute(ctx: ExecuteContext, _x: IR): (Annotation, PType) = {
    val x = LoweringPipeline.darrayLowerer(true)(DArrayLowering.All).apply(ctx, _x)
      .asInstanceOf[IR]
    val (pt, f) = Compile[AsmFunction1RegionLong](ctx,
      FastIndexedSeq[(String, PType)](),
      FastIndexedSeq[TypeInfo[_]](classInfo[Region]), LongInfo,
      MakeTuple.ordered(FastIndexedSeq(x)),
      optimize = true)

    val a = f(0, ctx.r)(ctx.r)
    val retPType = pt.asInstanceOf[PBaseStruct]
    (new UnsafeRow(retPType, ctx.r, a).get(0), retPType.types(0))
  }
  def execute(username: String, sessionID: String, billingProject: String, bucket: String, code: String): Response = {
    statusForException {
      ExecutionTimer.logTime("ServiceBackend.execute") { timer =>
        userContext(username, timer) { ctx =>
          ctx.backendContext = new ServiceBackendContext(username, sessionID, billingProject, bucket)

          val (v, t) = execute(ctx, IRParser.parse_value_ir(ctx, code))

          JsonMethods.compact(
            JObject(List("value" -> JSONAnnotationImpex.exportAnnotation(v, t.virtualType),
              "type" -> JString(t.virtualType.toString))))
        }
      }
    }
  }

  def flags(): Response = {
    statusForException {
      JsonMethods.compact(JObject(HailContext.get.flags.available.toArray().map { case f: String =>
        val v = HailContext.getFlag(f)
        f -> (if (v == null) JNull else JString(v))
      }: _*))
    }
  }

  def getFlag(name: String): Response = {
    statusForException {
      val v = HailContext.getFlag(name)
      JsonMethods.compact(if (v == null) JNull else JString(v))
    }
  }

  def setFlag(name: String, value: String): Response = {
    statusForException {
      val v = HailContext.getFlag(name)
      HailContext.setFlag(name, value)
      JsonMethods.compact(if (v == null) JNull else JString(v))
    }
  }

  def unsetFlag(name: String): Response = {
    statusForException {
      val v = HailContext.getFlag(name)
      HailContext.setFlag(name, null)
      JsonMethods.compact(if (v == null) JNull else JString(v))
    }
  }

  def lowerDistributedSort(
    ctx: ExecuteContext,
    stage: TableStage,
    sortFields: IndexedSeq[SortField],
    relationalLetsAbove: Map[String, IR],
    tableTypeRequiredness: RTable
  ): TableStage = {
    val region = ctx.r
    val rowType = stage.rowType
    val keyFields = sortFields.map(_.field).toArray
    val keyType = rowType.typeAfterSelectNames(keyFields)
    val rowEType = EType.fromTypeAndAnalysis(rowType, tableTypeRequiredness.rowType).asInstanceOf[EBaseStruct]
    val keyEType = EType.fromTypeAndAnalysis(keyType, tableTypeRequiredness.rowType.select(keyFields)).asInstanceOf[EBaseStruct]
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
      val successfulPartitionIds = execute(
        ctx,
        stage.mapCollect(relationalLetsAbove)(
          ShuffleWrite(Literal(shuffleType, uuid), _)))

      val partitionBoundsPointers = shuffleClient.partitionBounds(region, stage.numPartitions)
      val partitionIntervals = partitionBoundsPointers.zip(partitionBoundsPointers.drop(1)).map { case (l, r) =>
        Interval(SafeRow(keyDecodedPType, l), SafeRow(keyDecodedPType, r), includesStart = true, includesEnd = false)
      }
      val last = partitionIntervals.last
      partitionIntervals(partitionIntervals.length - 1) = Interval(
        last.left.point, last.right.point, includesStart = true, includesEnd = true)

      val partitioner = new RVDPartitioner(keyType, partitionIntervals.toFastIndexedSeq)

      TableStage(
        globals = Literal(TStruct(), Row()),
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
  ): Response = {
    statusForException {
      ExecutionTimer.logTime("ServiceBackend.loadReferencesFromDataset") { timer =>
        userContext(username, timer) { ctx =>
          ReferenceGenome.fromHailDataset(ctx.fs, path)
        }
      }
    }
  }
}
