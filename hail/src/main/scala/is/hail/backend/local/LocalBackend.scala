package is.hail.backend.local

import is.hail.{HailContext, HailFeatureFlags}
import is.hail.annotations.{Region, SafeRow}
import is.hail.asm4s._
import is.hail.backend._
import is.hail.expr.Validate
import is.hail.expr.ir.{IRParser, _}
import is.hail.expr.ir.analyses.SemanticHash
import is.hail.expr.ir.lowering._
import is.hail.io.{BufferSpec, TypedCodecSpec}
import is.hail.io.fs._
import is.hail.linalg.BlockMatrix
import is.hail.types._
import is.hail.types.encoded.EType
import is.hail.types.physical.PTuple
import is.hail.types.physical.stypes.PTypeReferenceSingleCodeType
import is.hail.types.virtual.TVoid
import is.hail.utils._
import is.hail.variant.ReferenceGenome

import scala.collection.JavaConverters._
import scala.reflect.ClassTag

import java.io.PrintWriter

import org.apache.hadoop
import org.json4s._
import org.json4s.jackson.Serialization
import org.sparkproject.guava.util.concurrent.MoreExecutors

class LocalBroadcastValue[T](val value: T) extends BroadcastValue[T] with Serializable

class LocalTaskContext(val partitionId: Int, val stageId: Int) extends HailTaskContext {
  override def attemptNumber(): Int = 0
}

object LocalBackend {
  private var theLocalBackend: LocalBackend = _

  def apply(
    tmpdir: String,
    gcsRequesterPaysProject: String,
    gcsRequesterPaysBuckets: String,
    logFile: String = "hail.log",
    quiet: Boolean = false,
    append: Boolean = false,
    skipLoggingConfiguration: Boolean = false,
  ): LocalBackend = synchronized {
    require(theLocalBackend == null)

    if (!skipLoggingConfiguration)
      HailContext.configureLogging(logFile, quiet, append)
    theLocalBackend = new LocalBackend(
      tmpdir,
      gcsRequesterPaysProject,
      gcsRequesterPaysBuckets,
    )
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

class LocalBackend(
  val tmpdir: String,
  gcsRequesterPaysProject: String,
  gcsRequesterPaysBuckets: String,
) extends Backend with BackendWithCodeCache {
  // FIXME don't rely on hadoop
  val hadoopConf = new hadoop.conf.Configuration()

  if (gcsRequesterPaysProject != null) {
    if (gcsRequesterPaysBuckets == null) {
      hadoopConf.set("fs.gs.requester.pays.mode", "AUTO")
      hadoopConf.set("fs.gs.requester.pays.project.id", gcsRequesterPaysProject)
    } else {
      hadoopConf.set("fs.gs.requester.pays.mode", "CUSTOM")
      hadoopConf.set("fs.gs.requester.pays.project.id", gcsRequesterPaysProject)
      hadoopConf.set("fs.gs.requester.pays.buckets", gcsRequesterPaysBuckets)
    }
  }

  hadoopConf.set(
    "hadoop.io.compression.codecs",
    "org.apache.hadoop.io.compress.DefaultCodec,"
      + "is.hail.io.compress.BGzipCodec,"
      + "is.hail.io.compress.BGzipCodecTbi,"
      + "org.apache.hadoop.io.compress.GzipCodec",
  )

  private[this] val flags = HailFeatureFlags.fromEnv()
  private[this] val theHailClassLoader = new HailClassLoader(getClass().getClassLoader())

  def getFlag(name: String): String = flags.get(name)

  def setFlag(name: String, value: String) = flags.set(name, value)

  val availableFlags: java.util.ArrayList[String] = flags.available

  val fs: FS = new HadoopFS(new SerializableHadoopConfiguration(hadoopConf))

  def withExecuteContext[T](timer: ExecutionTimer): (ExecuteContext => T) => T =
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
    )

  override def withExecuteContext[T](methodName: String)(f: ExecuteContext => T): T =
    ExecutionTimer.logTime(methodName) { timer =>
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
      )(f)
    }

  def broadcast[T: ClassTag](value: T): BroadcastValue[T] = new LocalBroadcastValue[T](value)

  private[this] var stageIdx: Int = 0

  private[this] def nextStageId(): Int = {
    val current = stageIdx
    stageIdx += 1
    current
  }

  def parallelizeAndComputeWithIndex(
    backendContext: BackendContext,
    fs: FS,
    collection: Array[Array[Byte]],
    stageIdentifier: String,
    dependency: Option[TableStageDependency] = None,
  )(
    f: (Array[Byte], HailTaskContext, HailClassLoader, FS) => Array[Byte]
  ): Array[Array[Byte]] = {
    val stageId = nextStageId()
    collection.zipWithIndex.map { case (c, i) =>
      using(new LocalTaskContext(i, stageId))(htc => f(c, htc, theHailClassLoader, fs))
    }
  }

  override def parallelizeAndComputeWithIndexReturnAllErrors(
    backendContext: BackendContext,
    fs: FS,
    collection: IndexedSeq[(Array[Byte], Int)],
    stageIdentifier: String,
    dependency: Option[TableStageDependency] = None,
  )(
    f: (Array[Byte], HailTaskContext, HailClassLoader, FS) => Array[Byte]
  ): (Option[Throwable], IndexedSeq[(Array[Byte], Int)]) = {
    val stageId = nextStageId()
    runAllKeepFirstError(MoreExecutors.sameThreadExecutor) {
      collection.map { case (c, i) =>
        (
          () =>
            using(new LocalTaskContext(i, stageId)) {
              f(c, _, theHailClassLoader, fs)
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
  ): Either[Unit, (PTuple, Long)] = {
    val ir =
      LoweringPipeline.darrayLowerer(true)(DArrayLowering.All).apply(ctx, ir0).asInstanceOf[IR]

    if (!Compilable(ir))
      throw new LowererUnsupportedOperation(s"lowered to uncompilable IR: ${Pretty(ctx, ir)}")

    if (ir.typ == TVoid) {
      val (pt, f) = ctx.timer.time("Compile") {
        Compile[AsmFunction1RegionUnit](
          ctx,
          FastSeq(),
          FastSeq(classInfo[Region]),
          UnitInfo,
          ir,
          print = print,
        )
      }

      ctx.timer.time("Run") {
        Left(ctx.scopedExecution((hcl, fs, htc, r) => f(hcl, fs, htc, r).apply(r)))
      }
    } else {
      val (Some(PTypeReferenceSingleCodeType(pt: PTuple)), f) = ctx.timer.time("Compile") {
        Compile[AsmFunction1RegionLong](
          ctx,
          FastSeq(),
          FastSeq(classInfo[Region]),
          LongInfo,
          MakeTuple.ordered(FastSeq(ir)),
          print = print,
        )
      }

      ctx.timer.time("Run") {
        Right((pt, ctx.scopedExecution((hcl, fs, htc, r) => f(hcl, fs, htc, r).apply(r))))
      }
    }
  }

  private[this] def _execute(ctx: ExecuteContext, ir: IR): Either[Unit, (PTuple, Long)] = {
    TypeCheck(ctx, ir)
    Validate(ir)
    val queryID = Backend.nextID()
    log.info(s"starting execution of query $queryID of initial size ${IRSize(ir)}")
    ctx.irMetadata = ctx.irMetadata.copy(semhash = SemanticHash(ctx)(ir))
    val res = _jvmLowerAndExecute(ctx, ir)
    log.info(s"finished execution of query $queryID")
    res
  }

  def executeToJavaValue(timer: ExecutionTimer, ir: IR): (Any, ExecutionTimer) =
    withExecuteContext(timer) { ctx =>
      val result = _execute(ctx, ir) match {
        case Left(_) =>
          (null, ctx.timer)
        case Right((pt, off)) =>
          (SafeRow(pt, off).get(0), ctx.timer)
      }
      result
    }

  def executeToEncoded(timer: ExecutionTimer, ir: IR, bs: BufferSpec): Array[Byte] =
    withExecuteContext(timer) { ctx =>
      val result = _execute(ctx, ir) match {
        case Left(_) => Array[Byte]()
        case Right((pt, off)) =>
          val elementType = pt.fields(0).typ
          assert(pt.isFieldDefined(off, 0))
          val codec = TypedCodecSpec(
            EType.fromPythonTypeEncoding(elementType.virtualType),
            elementType.virtualType,
            bs,
          )
          codec.encode(ctx, elementType, pt.loadField(off, 0))
      }
      result
    }

  def executeLiteral(irStr: String): Int = {
    ExecutionTimer.logTime("SparkBackend.executeLiteral") { timer =>
      withExecuteContext(timer) { ctx =>
        val ir = IRParser.parse_value_ir(irStr, IRParserEnvironment(ctx, irMap = persistedIR.toMap))
        val t = ir.typ
        assert(t.isRealizable)
        val queryID = Backend.nextID()
        log.info(s"starting execution of query $queryID} of initial size ${IRSize(ir)}")
        val retVal = _execute(ctx, ir)
        val literalIR = retVal match {
          case Left(_) => throw new HailException("Can't create literal")
          case Right((pt, addr)) =>
            GetFieldByIdx(EncodedLiteral.fromPTypeAndAddress(pt, addr, ctx), 0)
        }
        log.info(s"finished execution of query $queryID")
        addJavaIR(literalIR)
      }
    }
  }

  override def execute(
    ir: String,
    timed: Boolean,
  )(
    consume: (ExecuteContext, Either[Unit, (PTuple, Long)], String) => Unit
  ): Unit = {
    withExecuteContext("LocalBackend.execute") { ctx =>
      val res = ctx.timer.time("execute") {
        val irData =
          IRParser.parse_value_ir(ir, IRParserEnvironment(ctx, irMap = persistedIR.toMap))
        val queryID = Backend.nextID()
        log.info(s"starting execution of query $queryID of initial size ${IRSize(irData)}")
        _execute(ctx, irData)
      }
      ctx.timer.finish()
      val timings = if (timed)
        Serialization.write(Map("timings" -> ctx.timer.toMap))(new DefaultFormats {})
      else ""
      consume(ctx, res, timings)
    }
  }

  def pyAddReference(jsonConfig: String): Unit = addReference(ReferenceGenome.fromJSON(jsonConfig))
  def pyRemoveReference(name: String): Unit = removeReference(name)

  def pyAddLiftover(name: String, chainFile: String, destRGName: String): Unit =
    ExecutionTimer.logTime("LocalBackend.pyReferenceAddLiftover") { timer =>
      withExecuteContext(timer)(ctx => references(name).addLiftover(ctx, chainFile, destRGName))
    }

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
  ): String = {
    ExecutionTimer.logTime("LocalBackend.pyFromFASTAFile") { timer =>
      withExecuteContext(timer) { ctx =>
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
    }
  }

  def pyAddSequence(name: String, fastaFile: String, indexFile: String): Unit =
    ExecutionTimer.logTime("LocalBackend.pyAddSequence") { timer =>
      withExecuteContext(timer)(ctx => references(name).addSequence(ctx, fastaFile, indexFile))
    }

  def pyRemoveSequence(name: String) = references(name).removeSequence()

  def parse_value_ir(s: String, refMap: java.util.Map[String, String]): IR =
    ExecutionTimer.logTime("LocalBackend.parse_value_ir") { timer =>
      withExecuteContext(timer) { ctx =>
        IRParser.parse_value_ir(
          s,
          IRParserEnvironment(ctx, persistedIR.toMap),
          BindingEnv.eval(refMap.asScala.toMap.mapValues(IRParser.parseType).toSeq: _*),
        )
      }
    }

  def parse_table_ir(s: String): TableIR =
    ExecutionTimer.logTime("LocalBackend.parse_table_ir") { timer =>
      withExecuteContext(timer) { ctx =>
        IRParser.parse_table_ir(s, IRParserEnvironment(ctx, irMap = persistedIR.toMap))
      }
    }

  def parse_matrix_ir(s: String): MatrixIR =
    ExecutionTimer.logTime("LocalBackend.parse_matrix_ir") { timer =>
      withExecuteContext(timer) { ctx =>
        IRParser.parse_matrix_ir(s, IRParserEnvironment(ctx, irMap = persistedIR.toMap))
      }
    }

  def parse_blockmatrix_ir(s: String): BlockMatrixIR =
    ExecutionTimer.logTime("LocalBackend.parse_blockmatrix_ir") { timer =>
      withExecuteContext(timer) { ctx =>
        IRParser.parse_blockmatrix_ir(s, IRParserEnvironment(ctx, irMap = persistedIR.toMap))
      }
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
