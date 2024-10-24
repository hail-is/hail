package is.hail.backend.local

import is.hail.{CancellingExecutorService, HailContext, HailFeatureFlags}
import is.hail.annotations.Region
import is.hail.asm4s._
import is.hail.backend._
import is.hail.expr.Validate
import is.hail.expr.ir.{IRParser, _}
import is.hail.expr.ir.analyses.SemanticHash
import is.hail.expr.ir.lowering._
import is.hail.io.fs._
import is.hail.linalg.BlockMatrix
import is.hail.types._
import is.hail.types.physical.PTuple
import is.hail.types.physical.stypes.PTypeReferenceSingleCodeType
import is.hail.types.virtual.{BlockMatrixType, TVoid}
import is.hail.utils._
import is.hail.variant.ReferenceGenome

import scala.collection.JavaConverters._
import scala.reflect.ClassTag

import java.io.PrintWriter

import com.google.common.util.concurrent.MoreExecutors
import org.apache.hadoop
import sourcecode.Enclosing

class LocalBroadcastValue[T](val value: T) extends BroadcastValue[T] with Serializable

class LocalTaskContext(val partitionId: Int, val stageId: Int) extends HailTaskContext {
  override def attemptNumber(): Int = 0
}

object LocalBackend {
  private var theLocalBackend: LocalBackend = _

  def apply(
    tmpdir: String,
    logFile: String = "hail.log",
    quiet: Boolean = false,
    append: Boolean = false,
    skipLoggingConfiguration: Boolean = false,
  ): LocalBackend = synchronized {
    require(theLocalBackend == null)

    if (!skipLoggingConfiguration)
      HailContext.configureLogging(logFile, quiet, append)

    theLocalBackend = new LocalBackend(tmpdir)
    theLocalBackend.addDefaultReferences()
    theLocalBackend
  }

  def stop(): Unit = synchronized {
    if (theLocalBackend != null) {
      theLocalBackend = null
      // Hadoop does not honor the hadoop configuration as a component of the cache key for file
      // systems, so we blow away the cache so that a new configuration can successfully take
      // effect.
      // https://github.com/hail-is/hail/pull/12133#issuecomment-1241322443
      hadoop.fs.FileSystem.closeAll()
    }
  }
}

class LocalBackend(val tmpdir: String) extends Backend with BackendWithCodeCache {

  private[this] val flags = HailFeatureFlags.fromMap(sys.env)
  private[this] val theHailClassLoader = new HailClassLoader(getClass().getClassLoader())

  def getFlag(name: String): String = flags.get(name)

  def setFlag(name: String, value: String) = flags.set(name, value)

  // called from python
  val availableFlags: java.util.ArrayList[String] =
    flags.available

  // flags can be set after construction from python
  def fs: FS = FS.buildRoutes(None, Some(flags), sys.env)

  override def withExecuteContext[T](f: ExecuteContext => T)(implicit E: Enclosing): T =
    ExecutionTimer.logTime { timer =>
      val fs = this.fs
      ExecuteContext.scoped(
        tmpdir,
        tmpdir,
        this,
        fs,
        timer,
        null,
        theHailClassLoader,
        flags,
        new BackendContext {
          override val executionCache: ExecutionCache =
            ExecutionCache.fromFlags(flags, fs, tmpdir)
        },
        new IrMetadata(),
      )(f)
    }

  def broadcast[T: ClassTag](value: T): BroadcastValue[T] = new LocalBroadcastValue[T](value)

  private[this] var stageIdx: Int = 0

  private[this] def nextStageId(): Int = {
    val current = stageIdx
    stageIdx += 1
    current
  }

  override def parallelizeAndComputeWithIndex(
    backendContext: BackendContext,
    fs: FS,
    contexts: IndexedSeq[Array[Byte]],
    stageIdentifier: String,
    dependency: Option[TableStageDependency],
    partitions: Option[IndexedSeq[Int]],
  )(
    f: (Array[Byte], HailTaskContext, HailClassLoader, FS) => Array[Byte]
  ): (Option[Throwable], IndexedSeq[(Array[Byte], Int)]) = {

    val stageId = nextStageId()
    runAllKeepFirstError(new CancellingExecutorService(MoreExecutors.newDirectExecutorService())) {
      partitions.getOrElse(contexts.indices).map { i =>
        (
          () =>
            using(new LocalTaskContext(i, stageId)) {
              f(contexts(i), _, theHailClassLoader, fs)
            },
          i,
        )
      }
    }
  }

  def defaultParallelism: Int = 1

  def stop(): Unit = LocalBackend.stop()

  private[this] def _jvmLowerAndExecute(
    ctx: ExecuteContext,
    ir0: IR,
    print: Option[PrintWriter] = None,
  ): Either[Unit, (PTuple, Long)] =
    ctx.time {
      val ir =
        LoweringPipeline.darrayLowerer(true)(DArrayLowering.All).apply(ctx, ir0).asInstanceOf[IR]

      if (!Compilable(ir))
        throw new LowererUnsupportedOperation(s"lowered to uncompilable IR: ${Pretty(ctx, ir)}")

      ir.typ match {
        case TVoid =>
          val (_, f) = Compile[AsmFunction1RegionUnit](
            ctx,
            FastSeq(),
            FastSeq(classInfo[Region]),
            UnitInfo,
            ir,
            print = print,
          )

          Left(ctx.scopedExecution((hcl, fs, htc, r) => f(hcl, fs, htc, r)(r)))
        case _ =>
          val (Some(PTypeReferenceSingleCodeType(pt: PTuple)), f) =
            Compile[AsmFunction1RegionLong](
              ctx,
              FastSeq(),
              FastSeq(classInfo[Region]),
              LongInfo,
              MakeTuple.ordered(FastSeq(ir)),
              print = print,
            )

          Right((pt, ctx.scopedExecution((hcl, fs, htc, r) => f(hcl, fs, htc, r)(r))))
      }
    }

  override def execute(ctx: ExecuteContext, ir: IR): Either[Unit, (PTuple, Long)] =
    ctx.time {
      TypeCheck(ctx, ir)
      Validate(ir)
      val queryID = Backend.nextID()
      log.info(s"starting execution of query $queryID of initial size ${IRSize(ir)}")
      ctx.irMetadata.semhash = SemanticHash(ctx)(ir)
      val res = _jvmLowerAndExecute(ctx, ir)
      log.info(s"finished execution of query $queryID")
      res
    }

  def executeLiteral(irStr: String): Int =
    withExecuteContext { ctx =>
      val ir = IRParser.parse_value_ir(irStr, IRParserEnvironment(ctx, persistedIR.toMap))
      assert(ir.typ.isRealizable)
      execute(ctx, ir) match {
        case Left(_) => throw new HailException("Can't create literal")
        case Right((pt, addr)) =>
          val field = GetFieldByIdx(EncodedLiteral.fromPTypeAndAddress(pt, addr, ctx), 0)
          addJavaIR(field)
      }
    }

  def pyAddReference(jsonConfig: String): Unit = addReference(ReferenceGenome.fromJSON(jsonConfig))
  def pyRemoveReference(name: String): Unit = removeReference(name)

  def pyAddLiftover(name: String, chainFile: String, destRGName: String): Unit =
    withExecuteContext(ctx => references(name).addLiftover(ctx, chainFile, destRGName))

  def pyRemoveLiftover(name: String, destRGName: String) =
    references(name).removeLiftover(destRGName)

  def pyFromFASTAFile(
    name: String,
    fastaFile: String,
    indexFile: String,
    xContigs: java.util.List[String],
    yContigs: java.util.List[String],
    mtContigs: java.util.List[String],
    parInput: java.util.List[String],
  ): String =
    withExecuteContext { ctx =>
      val rg = ReferenceGenome.fromFASTAFile(
        ctx,
        name,
        fastaFile,
        indexFile,
        xContigs.asScala.toArray,
        yContigs.asScala.toArray,
        mtContigs.asScala.toArray,
        parInput.asScala.toArray,
      )
      rg.toJSONString
    }

  def pyAddSequence(name: String, fastaFile: String, indexFile: String): Unit =
    withExecuteContext(ctx => references(name).addSequence(ctx, fastaFile, indexFile))

  def pyRemoveSequence(name: String) = references(name).removeSequence()

  def parse_value_ir(s: String, refMap: java.util.Map[String, String]): IR =
    withExecuteContext { ctx =>
      IRParser.parse_value_ir(
        s,
        IRParserEnvironment(ctx, persistedIR.toMap),
        BindingEnv.eval(refMap.asScala.toMap.map { case (n, t) =>
          Name(n) -> IRParser.parseType(t)
        }.toSeq: _*),
      )
    }

  def parse_table_ir(s: String): TableIR =
    withExecuteContext { ctx =>
      IRParser.parse_table_ir(s, IRParserEnvironment(ctx, irMap = persistedIR.toMap))
    }

  def parse_matrix_ir(s: String): MatrixIR =
    withExecuteContext { ctx =>
      IRParser.parse_matrix_ir(s, IRParserEnvironment(ctx, irMap = persistedIR.toMap))
    }

  def parse_blockmatrix_ir(s: String): BlockMatrixIR =
    withExecuteContext { ctx =>
      IRParser.parse_blockmatrix_ir(s, IRParserEnvironment(ctx, irMap = persistedIR.toMap))
    }

  override def lowerDistributedSort(
    ctx: ExecuteContext,
    stage: TableStage,
    sortFields: IndexedSeq[SortField],
    rt: RTable,
    nPartitions: Option[Int],
  ): TableReader =
    LowerDistributedSort.distributedSort(ctx, stage, sortFields, rt, nPartitions)

  def persist(backendContext: BackendContext, id: String, value: BlockMatrix, storageLevel: String)
    : Unit = ???

  def unpersist(backendContext: BackendContext, id: String): Unit = ???

  def getPersistedBlockMatrix(backendContext: BackendContext, id: String): BlockMatrix = ???

  def getPersistedBlockMatrixType(backendContext: BackendContext, id: String): BlockMatrixType = ???

  def tableToTableStage(ctx: ExecuteContext, inputIR: TableIR, analyses: LoweringAnalyses)
    : TableStage =
    LowerTableIR.applyTable(inputIR, DArrayLowering.All, ctx, analyses)
}
