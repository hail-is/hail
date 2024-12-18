package is.hail.backend

import is.hail.asm4s._
import is.hail.backend.Backend.jsonToBytes
import is.hail.backend.spark.SparkBackend
import is.hail.expr.ir.{
  BaseIR, CodeCacheKey, CompiledFunction, IR, IRParser, IRParserEnvironment, LoweringAnalyses,
  SortField, TableIR, TableReader,
}
import is.hail.expr.ir.lowering.{TableStage, TableStageDependency}
import is.hail.io.{BufferSpec, TypedCodecSpec}
import is.hail.io.fs._
import is.hail.io.plink.LoadPlink
import is.hail.io.vcf.LoadVCF
import is.hail.linalg.BlockMatrix
import is.hail.types._
import is.hail.types.encoded.EType
import is.hail.types.physical.PTuple
import is.hail.types.virtual.{BlockMatrixType, TFloat64}
import is.hail.utils._
import is.hail.variant.ReferenceGenome

import scala.collection.mutable
import scala.reflect.ClassTag

import java.io._
import java.nio.charset.StandardCharsets

import org.json4s._
import org.json4s.jackson.JsonMethods
import sourcecode.Enclosing

object Backend {

  private var id: Long = 0L

  def nextID(): String = {
    id += 1
    s"hail_query_$id"
  }

  def encodeToOutputStream(
    ctx: ExecuteContext,
    t: PTuple,
    off: Long,
    bufferSpecString: String,
    os: OutputStream,
  ): Unit = {
    val bs = BufferSpec.parseOrDefault(bufferSpecString)
    assert(t.size == 1)
    val elementType = t.fields(0).typ
    val codec = TypedCodecSpec(
      EType.fromPythonTypeEncoding(elementType.virtualType),
      elementType.virtualType,
      bs,
    )
    assert(t.isFieldDefined(off, 0))
    codec.encode(ctx, elementType, t.loadField(off, 0), os)
  }

  def jsonToBytes(f: => JValue): Array[Byte] =
    JsonMethods.compact(f).getBytes(StandardCharsets.UTF_8)
}

abstract class BroadcastValue[T] { def value: T }

trait BackendContext {
  def executionCache: ExecutionCache
}

abstract class Backend extends Closeable {
  val persistedIR: mutable.Map[Int, BaseIR] = mutable.Map()

  def defaultParallelism: Int

  def canExecuteParallelTasksOnDriver: Boolean = true

  def broadcast[T: ClassTag](value: T): BroadcastValue[T]

  def persist(backendContext: BackendContext, id: String, value: BlockMatrix, storageLevel: String)
    : Unit

  def unpersist(backendContext: BackendContext, id: String): Unit

  def getPersistedBlockMatrix(backendContext: BackendContext, id: String): BlockMatrix

  def getPersistedBlockMatrixType(backendContext: BackendContext, id: String): BlockMatrixType

  def parallelizeAndComputeWithIndex(
    backendContext: BackendContext,
    fs: FS,
    contexts: IndexedSeq[Array[Byte]],
    stageIdentifier: String,
    dependency: Option[TableStageDependency] = None,
    partitions: Option[IndexedSeq[Int]] = None,
  )(
    f: (Array[Byte], HailTaskContext, HailClassLoader, FS) => Array[Byte]
  ): (Option[Throwable], IndexedSeq[(Array[Byte], Int)])

  def asSpark(op: String): SparkBackend =
    fatal(s"${getClass.getSimpleName}: $op requires SparkBackend")

  def shouldCacheQueryInfo: Boolean = true

  def lookupOrCompileCachedFunction[T](k: CodeCacheKey)(f: => CompiledFunction[T])
    : CompiledFunction[T]

  def references: mutable.Map[String, ReferenceGenome]

  def lowerDistributedSort(
    ctx: ExecuteContext,
    stage: TableStage,
    sortFields: IndexedSeq[SortField],
    rt: RTable,
    nPartitions: Option[Int],
  ): TableReader

  final def lowerDistributedSort(
    ctx: ExecuteContext,
    stage: TableStage,
    sortFields: IndexedSeq[SortField],
    rt: RTable,
  ): TableReader =
    lowerDistributedSort(ctx, stage, sortFields, rt, None)

  final def lowerDistributedSort(
    ctx: ExecuteContext,
    inputIR: TableIR,
    sortFields: IndexedSeq[SortField],
    rt: RTable,
    nPartitions: Option[Int] = None,
  ): TableReader = {
    val analyses = LoweringAnalyses.apply(inputIR, ctx)
    val inputStage = tableToTableStage(ctx, inputIR, analyses)
    lowerDistributedSort(ctx, inputStage, sortFields, rt, nPartitions)
  }

  def tableToTableStage(ctx: ExecuteContext, inputIR: TableIR, analyses: LoweringAnalyses)
    : TableStage

  def withExecuteContext[T](f: ExecuteContext => T)(implicit E: Enclosing): T

  final def valueType(s: String): Array[Byte] =
    jsonToBytes {
      withExecuteContext { ctx =>
        IRParser.parse_value_ir(s, IRParserEnvironment(ctx, persistedIR.toMap)).typ.toJSON
      }
    }

  final def tableType(s: String): Array[Byte] =
    jsonToBytes {
      withExecuteContext { ctx =>
        IRParser.parse_table_ir(s, IRParserEnvironment(ctx, persistedIR.toMap)).typ.toJSON
      }
    }

  final def matrixTableType(s: String): Array[Byte] =
    jsonToBytes {
      withExecuteContext { ctx =>
        IRParser.parse_matrix_ir(s, IRParserEnvironment(ctx, persistedIR.toMap)).typ.toJSON
      }
    }

  final def blockMatrixType(s: String): Array[Byte] =
    jsonToBytes {
      withExecuteContext { ctx =>
        IRParser.parse_blockmatrix_ir(s, IRParserEnvironment(ctx, persistedIR.toMap)).typ.toJSON
      }
    }

  def loadReferencesFromDataset(path: String): Array[Byte]

  def fromFASTAFile(
    name: String,
    fastaFile: String,
    indexFile: String,
    xContigs: Array[String],
    yContigs: Array[String],
    mtContigs: Array[String],
    parInput: Array[String],
  ): Array[Byte] =
    withExecuteContext { ctx =>
      jsonToBytes {
        Extraction.decompose {
          ReferenceGenome.fromFASTAFile(ctx, name, fastaFile, indexFile,
            xContigs, yContigs, mtContigs, parInput).toJSON
        }(defaultJSONFormats)
      }
    }

  def parseVCFMetadata(path: String): Array[Byte] =
    withExecuteContext { ctx =>
      jsonToBytes {
        Extraction.decompose {
          LoadVCF.parseHeaderMetadata(ctx.fs, Set.empty, TFloat64, path)
        }(defaultJSONFormats)
      }
    }

  def importFam(path: String, isQuantPheno: Boolean, delimiter: String, missingValue: String)
    : Array[Byte] =
    withExecuteContext { ctx =>
      LoadPlink.importFamJSON(ctx.fs, path, isQuantPheno, delimiter, missingValue).getBytes(
        StandardCharsets.UTF_8
      )
    }

  def execute(ctx: ExecuteContext, ir: IR): Either[Unit, (PTuple, Long)]
}

trait BackendWithCodeCache {
  private[this] val codeCache: Cache[CodeCacheKey, CompiledFunction[_]] = new Cache(50)

  def lookupOrCompileCachedFunction[T](k: CodeCacheKey)(f: => CompiledFunction[T])
    : CompiledFunction[T] = {
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
  def lookupOrCompileCachedFunction[T](k: CodeCacheKey)(f: => CompiledFunction[T])
    : CompiledFunction[T] = f
}
