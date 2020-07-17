package is.hail.backend.service

import java.io.{DataOutputStream, FileInputStream, FileOutputStream, ObjectInputStream, ObjectOutputStream, PrintWriter, RandomAccessFile, StringWriter}

import is.hail.annotations.{Region, UnsafeRow}
import is.hail.asm4s._
import is.hail.backend.{Backend, BackendContext, BroadcastValue}
import is.hail.expr.JSONAnnotationImpex
import is.hail.expr.ir.lowering.{DArrayLowering, LowerDistributedSort, LowererUnsupportedOperation, LoweringPipeline, TableStage}
import is.hail.expr.ir.{Compile, ExecuteContext, IR, IRParser, MakeTuple, SortField}
import is.hail.types.physical.{PBaseStruct, PType}
import is.hail.io.fs.{FS, GoogleStorageFS}
import is.hail.services.batch_client.BatchClient
import is.hail.types.virtual.Type
import is.hail.utils._
import org.apache.commons.io.IOUtils
import org.apache.log4j.LogManager
import org.json4s.{DefaultFormats, Formats}
import org.json4s.JsonAST.{JArray, JBool, JInt, JObject, JString}
import org.json4s.jackson.JsonMethods

import scala.collection.mutable
import scala.reflect.ClassTag

object Worker {
  def main(args: Array[String]): Unit = {
    if (args.length != 2)
      throw new IllegalArgumentException(s"expected one argument, not: ${ args.length }")

    val root = args(0)
    val i = args(1).toInt

    val fs = using(new FileInputStream("/gsa-key/key.json")) { is =>
      new GoogleStorageFS(IOUtils.toString(is))
    }

    val f = using(new ObjectInputStream(fs.openNoCompression(s"$root/f"))) { is =>
      is.readObject().asInstanceOf[(Array[Byte], Int) => Array[Byte]]
    }

    var offset = 0L
    var length = 0

    using(fs.openNoCompression(s"$root/context.offsets")) { is =>
      is.seek(i * 12)
      offset = is.readLong()
      length = is.readInt()
    }

    println(s"offset $offset length $length")

    val context = using(fs.openNoCompression(s"$root/contexts")) { is =>
      is.seek(offset)
      val context = new Array[Byte](length)
      is.readFully(context)
      context
    }

    val result = f(context, i)

    using(fs.createNoCompression(s"$root/result.$i")) { os =>
      os.write(result)
    }
  }
}

class ServiceBackendContext(
  val username: String,
  @transient val sessionID: String,
  val billingProject: String,
  val bucket: String
) extends BackendContext

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

  def userContext[T](username: String)(f: (ExecuteContext) => T): T = {
    val user = users(username)
    ExecuteContext.scoped(user.tmpdir, "file:///tmp", this, user.fs)(f)
  }

  def defaultParallelism: Int = 10

  def broadcast[T: ClassTag](_value: T): BroadcastValue[T] = new BroadcastValue[T] with Serializable {
    def value: T = _value
  }

  def parallelizeAndComputeWithIndex(_backendContext: BackendContext, collection: Array[Array[Byte]])(f: (Array[Byte], Int) => Array[Byte]): Array[Array[Byte]] = {
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
          "image" -> JString(workerImage),
          "mount_docker_socket" -> JBool(false),
          "command" -> JArray(List(
            JString("/bin/bash"),
            JString("-c"),
            JString(s"java -cp $$SPARK_HOME/jars/*:/hail.jar is.hail.backend.service.Worker $root $i"))),
          "job_id" -> JInt(i),
          "parent_ids" -> JArray(List()))
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
      userContext(username) { ctx =>
        val x = IRParser.parse_value_ir(ctx, s)
        x.typ.toString
      }
    }
  }

  def tableType(username: String, s: String): Response = {
    statusForException {
      userContext(username) { ctx =>
        val x = IRParser.parse_table_ir(ctx, s)
        val t = x.typ
        val jv = JObject("global" -> JString(t.globalType.toString),
          "row" -> JString(t.rowType.toString),
          "row_key" -> JArray(t.key.map(f => JString(f)).toList))
        JsonMethods.compact(jv)
      }
    }
  }

  def matrixTableType(username: String, s: String): Response = {
    statusForException {
      userContext(username) { ctx =>
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

  def blockMatrixType(username: String, s: String): Response = {
    statusForException {
      userContext(username) { ctx =>
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

  def execute(username: String, sessionID: String, billingProject: String, bucket: String, code: String): Response = {
    statusForException {
      userContext(username) { ctx =>
        ctx.backendContext = new ServiceBackendContext(username, sessionID, billingProject, bucket)

        var x = IRParser.parse_value_ir(ctx, code)
        x = LoweringPipeline.darrayLowerer(true)(DArrayLowering.All).apply(ctx, x)
          .asInstanceOf[IR]
        val (pt, f) = Compile[AsmFunction1RegionLong](ctx,
          FastIndexedSeq[(String, PType)](),
          FastIndexedSeq[TypeInfo[_]](classInfo[Region]), LongInfo,
          MakeTuple.ordered(FastIndexedSeq(x)),
          optimize = true)

        val a = f(0, ctx.r)(ctx.r)
        val v = new UnsafeRow(pt.asInstanceOf[PBaseStruct], ctx.r, a)

        JsonMethods.compact(
          JObject(List("value" -> JSONAnnotationImpex.exportAnnotation(v.get(0), x.typ),
            "type" -> JString(x.typ.toString))))
      }
    }
  }

  def lowerDistributedSort(ctx: ExecuteContext, stage: TableStage, sortFields: IndexedSeq[SortField], relationalLetsAbove: Seq[(String, Type)]): TableStage = {
    // Use a local sort for the moment to enable larger pipelines to run
    LowerDistributedSort.localSort(ctx, stage, sortFields, relationalLetsAbove)
  }
}
