package is.hail.backend

import com.sun.net.httpserver.{HttpContext, HttpExchange, HttpHandler, HttpServer}
import java.io._
import java.net._
import java.nio.charset.StandardCharsets

import org.json4s._
import org.json4s.jackson.{JsonMethods, Serialization}

import is.hail.asm4s._
import is.hail.backend.spark.SparkBackend
import is.hail.expr.ir.lowering.{TableStage, TableStageDependency}
import is.hail.expr.ir.{CodeCacheKey, CompiledFunction, LoweringAnalyses, SortField, TableIR, TableReader}
import is.hail.io.{BufferSpec, TypedCodecSpec}
import is.hail.io.fs._
import is.hail.io.plink.LoadPlink
import is.hail.io.vcf.LoadVCF
import is.hail.expr.ir.{IRParser, BaseIR}
import is.hail.linalg.BlockMatrix
import is.hail.types._
import is.hail.types.encoded.EType
import is.hail.types.virtual.TFloat64
import is.hail.types.physical.PTuple
import is.hail.utils._
import is.hail.variant.ReferenceGenome

import scala.collection.mutable
import scala.reflect.ClassTag
import is.hail.expr.ir.IRParserEnvironment


case class IRTypePayload(ir: String)
case class LoadReferencesFromDatasetPayload(path: String)
case class FromFASTAFilePayload(name: String, fasta_file: String, index_file: String,
    x_contigs: Array[String], y_contigs: Array[String], mt_contigs: Array[String],
    par: Array[String])
case class ParseVCFMetadataPayload(path: String)
case class ImportFamPayload(path: String, quant_pheno: Boolean, delimiter: String, missing: String)
case class ExecutePayload(ir: String, stream_codec: String)

object BackendServer {
  def apply(backend: Backend) = new BackendServer(backend)
}

class BackendServer(backend: Backend) {
  // 0 => let the OS pick an available port
  private val httpServer = HttpServer.create(new InetSocketAddress(0), 10)
  private val handler = new BackendHttpHandler(backend)

  def port = httpServer.getAddress.getPort

  def start(): Unit = {
    httpServer.createContext("/", handler)
    httpServer.setExecutor(null)
    httpServer.start()
  }

  def stop(): Unit = {
    httpServer.stop(10)
  }
}

class BackendHttpHandler(backend: Backend) extends HttpHandler {
  def handle(exchange: HttpExchange): Unit = {
    implicit val formats: Formats = DefaultFormats

    try {
      val body = using(exchange.getRequestBody)(JsonMethods.parse(_))
      if (exchange.getRequestURI.getPath == "/execute") {
          val config = body.extract[ExecutePayload]
          backend.execute(config.ir, false) { (ctx, res, timings) =>
            exchange.getResponseHeaders().add("X-Hail-Timings", timings)
            res match {
              case Left(_) => exchange.sendResponseHeaders(200, -1L)
              case Right((t, off)) =>
                exchange.sendResponseHeaders(200, 0L)  // 0 => an arbitrarily long response body
                using(exchange.getResponseBody()) { os =>
                  backend.encodeToOutputStream(ctx, t, off, config.stream_codec, os)
                }
            }
          }
          return
      }
      val response: Array[Byte] = exchange.getRequestURI.getPath match {
        case "/value/type" => backend.valueType(body.extract[IRTypePayload].ir)
        case "/table/type" => backend.tableType(body.extract[IRTypePayload].ir)
        case "/matrixtable/type" => backend.matrixTableType(body.extract[IRTypePayload].ir)
        case "/blockmatrix/type" => backend.blockMatrixType(body.extract[IRTypePayload].ir)
        case "/references/load" => backend.loadReferencesFromDataset(body.extract[LoadReferencesFromDatasetPayload].path)
        case "/references/from_fasta" =>
          val config = body.extract[FromFASTAFilePayload]
          backend.fromFASTAFile(config.name, config.fasta_file, config.index_file,
            config.x_contigs, config.y_contigs, config.mt_contigs, config.par)
        case "/vcf/metadata/parse" => backend.parseVCFMetadata(body.extract[ParseVCFMetadataPayload].path)
        case "/fam/import" =>
          val config = body.extract[ImportFamPayload]
          backend.importFam(config.path, config.quant_pheno, config.delimiter, config.missing)
      }

      exchange.sendResponseHeaders(200, response.length)
      using(exchange.getResponseBody())(_.write(response))
    } catch {
      case t: Throwable =>
        error(t.getMessage)
        val (shortMessage, expandedMessage, errorId) = handleForPython(t)
        val errorJson = JObject(
          "short" -> JString(shortMessage),
          "expanded" -> JString(expandedMessage),
          "error_id" -> JInt(errorId)
        )
        val errorBytes = JsonMethods.compact(errorJson).getBytes(StandardCharsets.UTF_8)
        exchange.sendResponseHeaders(500, errorBytes.length)
        using(exchange.getResponseBody())(_.write(errorBytes))
    }
  }
}

object Backend {

  private var id: Long = 0L
  def nextID(): String = {
    id += 1
    s"hail_query_$id"
  }
}

abstract class BroadcastValue[T] { def value: T }

trait BackendContext {
  def executionCache: ExecutionCache
}

abstract class Backend {
  val persistedIR: mutable.Map[String, BaseIR] = mutable.Map()

  def removeJavaIR(id: String): Unit = persistedIR.remove(id)

  def defaultParallelism: Int

  def canExecuteParallelTasksOnDriver: Boolean = true

  def broadcast[T: ClassTag](value: T): BroadcastValue[T]

  def persist(backendContext: BackendContext, id: String, value: BlockMatrix, storageLevel: String): Unit

  def unpersist(backendContext: BackendContext, id: String): Unit

  def getPersistedBlockMatrix(backendContext: BackendContext, id: String): BlockMatrix

  def getPersistedBlockMatrixType(backendContext: BackendContext, id: String): BlockMatrixType

  def parallelizeAndComputeWithIndex(
    backendContext: BackendContext,
    fs: FS,
    collection: IndexedSeq[(Array[Byte], Int)],
    stageIdentifier: String,
    dependency: Option[TableStageDependency] = None
  )(f: (Array[Byte], HailTaskContext, HailClassLoader, FS) => Array[Byte])
  : (Option[Throwable], IndexedSeq[(Array[Byte], Int)])

  def stop(): Unit

  def asSpark(op: String): SparkBackend =
    fatal(s"${ getClass.getSimpleName }: $op requires SparkBackend")

  def shouldCacheQueryInfo: Boolean = true

  def lookupOrCompileCachedFunction[T](k: CodeCacheKey)(f: => CompiledFunction[T]): CompiledFunction[T]

  var references: Map[String, ReferenceGenome] = Map.empty

  def addDefaultReferences(): Unit = {
    references = ReferenceGenome.builtinReferences()
  }

  def addReference(rg: ReferenceGenome) {
    references.get(rg.name) match {
      case Some(rg2) =>
        if (rg != rg2) {
          fatal(s"Cannot add reference genome '${ rg.name }', a different reference with that name already exists. Choose a reference name NOT in the following list:\n  " +
            s"@1", references.keys.truncatable("\n  "))
        }
      case None =>
        references += (rg.name -> rg)
    }
  }

  def hasReference(name: String) = references.contains(name)
  def removeReference(name: String): Unit = {
    references -= name
  }

  def lowerDistributedSort(
    ctx: ExecuteContext,
    stage: TableStage,
    sortFields: IndexedSeq[SortField],
    rt: RTable,
    nPartitions: Option[Int]
  ): TableReader

  final def lowerDistributedSort(
    ctx: ExecuteContext,
    stage: TableStage,
    sortFields: IndexedSeq[SortField],
    rt: RTable
  ): TableReader =
    lowerDistributedSort(ctx, stage, sortFields, rt, None)

  final def lowerDistributedSort(
    ctx: ExecuteContext,
    inputIR: TableIR,
    sortFields: IndexedSeq[SortField],
    rt: RTable,
    nPartitions: Option[Int] = None
  ): TableReader = {
    val analyses = LoweringAnalyses.apply(inputIR, ctx)
    val inputStage = tableToTableStage(ctx, inputIR, analyses)
    lowerDistributedSort(ctx, inputStage, sortFields, rt, nPartitions)
  }

  def tableToTableStage(ctx: ExecuteContext,
    inputIR: TableIR,
    analyses: LoweringAnalyses
  ): TableStage

  def withExecuteContext[T](methodName: String): (ExecuteContext => T) => T

  final def valueType(s: String): Array[Byte] = {
    withExecuteContext("tableType") { ctx =>
      val v = IRParser.parse_value_ir(s, IRParserEnvironment(ctx, irMap = persistedIR.toMap))
      v.typ.toString.getBytes(StandardCharsets.UTF_8)
    }
  }

  private[this] def jsonToBytes(f: => JValue): Array[Byte] = {
    JsonMethods.compact(f).getBytes(StandardCharsets.UTF_8)
  }

  final def tableType(s: String): Array[Byte] = jsonToBytes {
    withExecuteContext("tableType") { ctx =>
      val x = IRParser.parse_table_ir(s, IRParserEnvironment(ctx, irMap = persistedIR.toMap))
      x.typ.toJSON
    }
  }

  final def matrixTableType(s: String): Array[Byte] = jsonToBytes {
    withExecuteContext("matrixTableType") { ctx =>
      IRParser.parse_matrix_ir(s, IRParserEnvironment(ctx, irMap = persistedIR.toMap)).typ.pyJson
    }
  }

  final def blockMatrixType(s: String): Array[Byte] = jsonToBytes {
    withExecuteContext("blockMatrixType") { ctx =>
      val x = IRParser.parse_blockmatrix_ir(s, IRParserEnvironment(ctx, irMap = persistedIR.toMap))
      val t = x.typ
      JObject(
        "element_type" -> JString(t.elementType.toString),
        "shape" -> JArray(t.shape.map(s => JInt(s)).toList),
        "is_row_vector" -> JBool(t.isRowVector),
        "block_size" -> JInt(t.blockSize)
      )
    }
  }

  def loadReferencesFromDataset(path: String): Array[Byte] = {
    withExecuteContext("loadReferencesFromDataset") { ctx =>
      val rgs = ReferenceGenome.fromHailDataset(ctx.fs, path)
      rgs.foreach(addReference)

      implicit val formats: Formats = defaultJSONFormats
      Serialization.write(rgs.map(_.toJSON).toFastSeq).getBytes(StandardCharsets.UTF_8)
    }
  }

  def fromFASTAFile(name: String, fastaFile: String, indexFile: String,
    xContigs: Array[String], yContigs: Array[String], mtContigs: Array[String],
    parInput: Array[String]): Array[Byte] = {
    withExecuteContext("fromFASTAFile") { ctx =>
      val rg = ReferenceGenome.fromFASTAFile(ctx, name, fastaFile, indexFile,
        xContigs, yContigs, mtContigs, parInput)
      rg.toJSONString.getBytes(StandardCharsets.UTF_8)
    }
  }

  def parseVCFMetadata(path: String): Array[Byte] = jsonToBytes {
    withExecuteContext("parseVCFMetadata") { ctx =>
      val metadata = LoadVCF.parseHeaderMetadata(ctx.fs, Set.empty, TFloat64, path)
      implicit val formats = defaultJSONFormats
      Extraction.decompose(metadata)
    }
  }

  def importFam(path: String, isQuantPheno: Boolean, delimiter: String, missingValue: String): Array[Byte] = {
    withExecuteContext("importFam") { ctx =>
      LoadPlink.importFamJSON(ctx.fs, path, isQuantPheno, delimiter, missingValue).getBytes(StandardCharsets.UTF_8)
    }
  }

  def execute(ir: String, timed: Boolean)(consume: (ExecuteContext, Either[Unit, (PTuple, Long)], String) => Unit): Unit = ()

  def encodeToOutputStream(ctx: ExecuteContext, t: PTuple, off: Long, bufferSpecString: String, os: OutputStream): Unit = {
    val bs = BufferSpec.parseOrDefault(bufferSpecString)
    assert(t.size == 1)
    val elementType = t.fields(0).typ
    val codec = TypedCodecSpec(
      EType.fromTypeAllOptional(elementType.virtualType), elementType.virtualType, bs)
    assert(t.isFieldDefined(off, 0))
    codec.encode(ctx, elementType, t.loadField(off, 0), os)
  }
}

trait BackendWithCodeCache {
  private[this] val codeCache: Cache[CodeCacheKey, CompiledFunction[_]] = new Cache(50)
  def lookupOrCompileCachedFunction[T](k: CodeCacheKey)(f: => CompiledFunction[T]): CompiledFunction[T] = {
    codeCache.get(k) match {
      case Some(v) => v.asInstanceOf[CompiledFunction[T]]
      case None =>
        val compiledFunction = f
        codeCache += ((k, compiledFunction))
        compiledFunction
    }
  }
}

trait BackendWithNoCodeCache {
  def lookupOrCompileCachedFunction[T](k: CodeCacheKey)(f: => CompiledFunction[T]): CompiledFunction[T] = f
}
