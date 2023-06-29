package is.hail.backend

import is.hail.asm4s._
import is.hail.backend.spark.SparkBackend
import is.hail.expr.ir.analyses.SemanticHash
import is.hail.expr.ir.lowering.{TableStage, TableStageDependency}
import is.hail.expr.ir.{CodeCacheKey, CompiledFunction, LoweringAnalyses, SortField, TableIR, TableReader}
import is.hail.io.fs._
import is.hail.linalg.BlockMatrix
import is.hail.types._
import is.hail.utils._
import is.hail.variant.ReferenceGenome

import scala.reflect.ClassTag

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
  )(f: (Array[Byte], HailTaskContext, HailClassLoader, FS) => Array[Byte])
  : (Option[Throwable], IndexedSeq[(Int, Array[Byte])])

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
    rt: RTable
  ): TableReader

  final def lowerDistributedSort(
    ctx: ExecuteContext,
    inputIR: TableIR,
    sortFields: IndexedSeq[SortField],
    rt: RTable
  ): TableReader = {
    val analyses = LoweringAnalyses.apply(inputIR, ctx)
    val inputStage = tableToTableStage(ctx, inputIR, analyses)
    lowerDistributedSort(ctx, inputStage, sortFields, rt)
  }

  def tableToTableStage(ctx: ExecuteContext,
    inputIR: TableIR,
    analyses: LoweringAnalyses
  ): TableStage
}

trait BackendWithCodeCache {
  private[this] val codeCache: Cache[CodeCacheKey, CompiledFunction[_]] = new Cache(50)
  def lookupOrCompileCachedFunction[T](k: CodeCacheKey)(f: => CompiledFunction[T]): CompiledFunction[T] = {
    codeCache.get(k) match {
      case Some(v) => v.asInstanceOf[CompiledFunction[T]]
      case None =>
        val compiledFunction = f
        codeCache += ((k, f))
        f
    }
  }
}

trait BackendWithNoCodeCache {
  def lookupOrCompileCachedFunction[T](k: CodeCacheKey)(f: => CompiledFunction[T]): CompiledFunction[T] = f
}
