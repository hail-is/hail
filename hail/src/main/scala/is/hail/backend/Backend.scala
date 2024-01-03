package is.hail.backend

import java.io._
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


object Backend {

  private var id: Long = 0L
  def nextID(): String = {
    id += 1
    s"hail_query_$id"
  }

  private var irID: Int = 0
  def nextIRID(): Int = {
    irID += 1
    irID
  }
}

abstract class BroadcastValue[T] { def value: T }

trait BackendContext {
  def executionCache: ExecutionCache
}

abstract class Backend {
  val persistedIR: mutable.Map[Int, BaseIR] = mutable.Map()

  protected[this] def addJavaIR(ir: BaseIR): Int = {
    val id = Backend.nextIRID()
    persistedIR += (id -> ir)
    id
  }

  def removeJavaIR(id: Int): Unit = persistedIR.remove(id)

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
    collection: Array[Array[Byte]],
    stageIdentifier: String,
    dependency: Option[TableStageDependency] = None
  )(
    f: (Array[Byte], HailTaskContext, HailClassLoader, FS) => Array[Byte]
  ): Array[Array[Byte]]

  def parallelizeAndComputeWithIndexReturnAllErrors(
    backendContext: BackendContext,
    fs: FS,
    collection: IndexedSeq[(Array[Byte], Int)],
    stageIdentifier: String,
    dependency: Option[TableStageDependency] = None
  )(
    f: (Array[Byte], HailTaskContext, HailClassLoader, FS) => Array[Byte]
  ): (Option[Throwable], IndexedSeq[(Array[Byte], Int)])

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
    withExecuteContext("valueType") { ctx =>
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
      EType.fromPythonTypeEncoding(elementType.virtualType), elementType.virtualType, bs)
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
