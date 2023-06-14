package is.hail.backend.local

import cats.syntax.all._
import is.hail.annotations.{Region, SafeRow, UnsafeRow}
import is.hail.asm4s._
import is.hail.backend._
import is.hail.backend.utils.{raisePretty, scopedExecution, timeM}
import is.hail.expr.ir.lowering._
import is.hail.expr.ir.{TypeCheck, _}
import is.hail.expr.{JSONAnnotationImpex, Validate}
import is.hail.io.fs._
import is.hail.io.plink.LoadPlink
import is.hail.io.{BufferSpec, TypedCodecSpec}
import is.hail.linalg.BlockMatrix
import is.hail.types._
import is.hail.types.encoded.EType
import is.hail.types.physical.PTuple
import is.hail.types.physical.stypes.{PTypeReferenceSingleCodeType, SingleCodeType}
import is.hail.types.virtual.TVoid
import is.hail.utils._
import is.hail.variant.ReferenceGenome
import is.hail.{HailContext, HailFeatureFlags}
import org.apache.hadoop
import org.json4s._
import org.json4s.jackson.{JsonMethods, Serialization}
import org.sparkproject.guava.util.concurrent.MoreExecutors

import java.io.PrintWriter
import scala.collection.JavaConverters._
import scala.language.higherKinds
import scala.reflect.ClassTag

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
    skipLoggingConfiguration: Boolean = false
  ): LocalBackend = synchronized {
    require(theLocalBackend == null)

    if (!skipLoggingConfiguration)
      HailContext.configureLogging(logFile, quiet, append)
    theLocalBackend = new LocalBackend(
      tmpdir,
      gcsRequesterPaysProject,
      gcsRequesterPaysBuckets
    )
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
  gcsRequesterPaysBuckets: String
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
      + "org.apache.hadoop.io.compress.GzipCodec")

  private[this] val flags = HailFeatureFlags.fromEnv()
  private[this] val theHailClassLoader = new HailClassLoader(getClass().getClassLoader())

  def getFlag(name: String): String = flags.get(name)

  def setFlag(name: String, value: String) = flags.set(name, value)

  val availableFlags: java.util.ArrayList[String] = flags.available

  val fs: FS = new HadoopFS(new SerializableHadoopConfiguration(hadoopConf))

  def withExecuteContext[T](timer: ExecutionTimer)(f: ExecuteContext => T): T =
    ExecuteContext.scoped(tmpdir, tmpdir, this, fs, timer, null, theHailClassLoader, this.references, flags) {
      ctx =>
        ctx.backendContext = new BackendContext {
          override def executionCache: ExecutionCache =
            ExecutionCache.fromFlags(flags, fs, tmpdir)
        }
      f(ctx)
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
    collection: Array[Array[Byte]],
    stageIdentifier: String,
    dependency: Option[TableStageDependency] = None
  )(f: (Array[Byte], HailTaskContext, HailClassLoader, FS) => Array[Byte])
  : (Option[Throwable], IndexedSeq[(Int, Array[Byte])]) = {
    val stageId = nextStageId()
    runAllKeepFirstError(MoreExecutors.sameThreadExecutor) {
      collection.zipWithIndex.map { case (c, i) =>
        () => using(new LocalTaskContext(i, stageId)) {
          f(c, _, theHailClassLoader, fs)
        }
      }
    }
  }

  def defaultParallelism: Int = 1

  def stop(): Unit = LocalBackend.stop()

  private[this] def _jvmLowerAndExecute[M[_]](ir0: IR, print: Option[PrintWriter] = None)
                                             (implicit M: MonadLower[M])
  : M[(Option[SingleCodeType], Long)] =
    LoweringPipeline.darrayLowerer(true)(DArrayLowering.All)(ir0).flatMap { case ir: IR =>

      if (!Compilable(ir))
        raisePretty(pretty => new LowererUnsupportedOperation(s"lowered to uncompilable IR: ${pretty(ir)}"))
      else
        ir.typ match {
        case TVoid =>
          for {
            (pt, f) <- timeM("Compile") {
              Compile[M, AsmFunction1RegionUnit](
                FastIndexedSeq(),
                FastIndexedSeq(classInfo[Region]), UnitInfo,
                ir,
                print = print
              )
            }

            _ <- timeM("Run") {
              scopedExecution.run { case (hcl, fs, htc, r) => M.pure(f(hcl, fs, htc, r)(r)) }
            }
          } yield (pt, 0)

        case _ =>
          for {
            (pt, f) <- timeM("Compile") {
              Compile[M, AsmFunction1RegionLong](
                FastIndexedSeq(),
                FastIndexedSeq(classInfo[Region]),
                LongInfo,
                MakeTuple.ordered(FastSeq(ir)),
                print = print
              )
            }

            addr <- timeM("Run") {
              scopedExecution.run { case (hcl, fs, htc, r) => M.pure(f(hcl, fs, htc, r)(r)) }
            }
          } yield (pt, addr)
      }
    }

  private[this] def _execute(ctx: ExecuteContext, ir: IR): (Option[SingleCodeType], Long) = {
    import Lower.monadLowerInstanceForLower
    (for {
      _ <- TypeCheck(ir)
      _ = Validate(ir)
      queryID = Backend.nextID()
      _ = log.info(s"starting execution of query $queryID of initial size ${IRSize(ir)}")
      result <- _jvmLowerAndExecute(ir)
      _ = log.info(s"finished execution of query $queryID")
    } yield result).runA(ctx, LoweringState())
  }

  def executeToJavaValue(timer: ExecutionTimer, ir: IR): Any =
    withExecuteContext(timer) { ctx =>
      val (pt, a) = _execute(ctx, ir)
      val result = pt match {
        case None =>
          (null, ctx.timer)
        case Some(PTypeReferenceSingleCodeType(pt: PTuple)) =>
          (SafeRow(pt, a).get(0), ctx.timer)
      }
      result
    }

  def executeToEncoded(timer: ExecutionTimer, ir: IR, bs: BufferSpec): Array[Byte] =
    withExecuteContext(timer) { ctx =>
      val (pt, a) = _execute(ctx, ir)
      val result = pt match {
        case None =>
          Array[Byte]()
        case Some(PTypeReferenceSingleCodeType(pt: PTuple)) =>
          val elementType = pt.fields(0).typ
          assert(pt.isFieldDefined(a, 0))
          val codec = TypedCodecSpec(
            EType.fromTypeAllOptional(elementType.virtualType), elementType.virtualType, bs)
          codec.encode(ctx, elementType, pt.loadField(a, 0))
      }
      result
    }


  def executeLiteral(ir: IR): IR = {
    ExecutionTimer.logTime("LocalBackend.executeLiteral") { timer =>
      val t = ir.typ
      assert(t.isRealizable)
      val (value, timings) = executeToJavaValue(timer, ir)
      Literal.coerce(t, value)
    }
  }

  def executeEncode(ir: IR, bufferSpecString: String, timed: Boolean): (Array[Byte], String) = {
    val (bytes, timer) = ExecutionTimer.time("LocalBackend.encodeToBytes") { timer =>
      val bs = BufferSpec.parseOrDefault(bufferSpecString)
      withExecuteContext(timer) { ctx =>
        executeToEncoded(timer, ir, bs)
      }
    }
    (bytes, if (timed) Serialization.write(Map("timings" -> timer.toMap))(new DefaultFormats {}) else "")
  }

  def decodeToJSON(ptypeString: String, b: Array[Byte], bufferSpecString: String): String = {
    ExecutionTimer.logTime("LocalBackend.decodeToJSON") { timer =>
      val t = IRParser.parsePType(ptypeString)
      val bs = BufferSpec.parseOrDefault(bufferSpecString)
      val codec = TypedCodecSpec(EType.defaultFromPType(t), t.virtualType, bs)
      withExecuteContext(timer) { ctx =>
        val (pt, off) = codec.decode(ctx, t.virtualType, b, ctx.r)
        assert(pt.virtualType == t.virtualType)
        JsonMethods.compact(JSONAnnotationImpex.exportAnnotation(
          UnsafeRow.read(pt, ctx.r, off), pt.virtualType))
      }
    }
  }

  def pyAddReference(jsonConfig: String): Unit = addReference(ReferenceGenome.fromJSON(jsonConfig))
  def pyRemoveReference(name: String): Unit = removeReference(name)

  def pyAddLiftover(name: String, chainFile: String, destRGName: String): Unit = {
    ExecutionTimer.logTime("LocalBackend.pyReferenceAddLiftover") { timer =>
      withExecuteContext(timer) { ctx =>
        references(name).addLiftover(ctx, chainFile, destRGName)
      }
    }
  }
  def pyRemoveLiftover(name: String, destRGName: String) = references(name).removeLiftover(destRGName)

  def pyFromFASTAFile(name: String, fastaFile: String, indexFile: String,
    xContigs: java.util.List[String], yContigs: java.util.List[String], mtContigs: java.util.List[String],
    parInput: java.util.List[String]): String = {
    ExecutionTimer.logTime("LocalBackend.pyFromFASTAFile") { timer =>
      withExecuteContext(timer) { ctx =>
        val rg = ReferenceGenome.fromFASTAFile(ctx, name, fastaFile, indexFile,
          xContigs.asScala.toArray, yContigs.asScala.toArray, mtContigs.asScala.toArray, parInput.asScala.toArray)
        rg.toJSONString
      }
    }
  }

  def pyAddSequence(name: String, fastaFile: String, indexFile: String): Unit = {
    ExecutionTimer.logTime("LocalBackend.pyAddSequence") { timer =>
      withExecuteContext(timer) { ctx =>
        references(name).addSequence(ctx, fastaFile, indexFile)
      }
    }
  }
  def pyRemoveSequence(name: String) = references(name).removeSequence()

  def parse_value_ir(s: String, refMap: java.util.Map[String, String], irMap: java.util.Map[String, BaseIR]): IR = {
    ExecutionTimer.logTime("LocalBackend.parse_value_ir") { timer =>
      withExecuteContext(timer) { ctx =>
        IRParser.parse_value_ir(s, IRParserEnvironment(ctx, BindingEnv.eval(refMap.asScala.toMap.mapValues(IRParser.parseType).toSeq: _*), irMap.asScala.toMap))
      }
    }
  }

  def parse_table_ir(s: String, irMap: java.util.Map[String, BaseIR]): TableIR = {
    ExecutionTimer.logTime("LocalBackend.parse_table_ir") { timer =>
      withExecuteContext(timer) { ctx =>
        IRParser.parse_table_ir(s, IRParserEnvironment(ctx, irMap = irMap.asScala.toMap))
      }
    }
  }

  def parse_matrix_ir(s: String, irMap: java.util.Map[String, BaseIR]): MatrixIR = {
    ExecutionTimer.logTime("LocalBackend.parse_matrix_ir") { timer =>
      withExecuteContext(timer) { ctx =>
        IRParser.parse_matrix_ir(s, IRParserEnvironment(ctx, irMap = irMap.asScala.toMap))
      }
    }
  }

  def parse_blockmatrix_ir(
    s: String, irMap: java.util.Map[String, BaseIR]
  ): BlockMatrixIR = {
    ExecutionTimer.logTime("LocalBackend.parse_blockmatrix_ir") { timer =>
      withExecuteContext(timer) { ctx =>
        IRParser.parse_blockmatrix_ir(s, IRParserEnvironment(ctx, irMap = irMap.asScala.toMap))
      }
    }
  }

  override def lowerDistributedSort[M[_]: MonadLower](
    stage: TableStage,
    sortFields: IndexedSeq[SortField],
    rt: RTable
  ): M[TableReader] =
    LowerDistributedSort.distributedSort(stage, sortFields, rt)

  def pyLoadReferencesFromDataset(path: String): String = {
    val rgs = ReferenceGenome.fromHailDataset(fs, path)
    rgs.foreach(addReference)

    implicit val formats: Formats = defaultJSONFormats
    Serialization.write(rgs.map(_.toJSON).toFastIndexedSeq)
  }

  def pyImportFam(path: String, isQuantPheno: Boolean, delimiter: String, missingValue: String): String =
    LoadPlink.importFamJSON(fs, path, isQuantPheno, delimiter, missingValue)

  def persist(backendContext: BackendContext, id: String, value: BlockMatrix, storageLevel: String): Unit = ???

  def unpersist(backendContext: BackendContext, id: String): Unit = ???

  def getPersistedBlockMatrix(backendContext: BackendContext, id: String): BlockMatrix = ???

  def getPersistedBlockMatrixType(backendContext: BackendContext, id: String): BlockMatrixType = ???

  def tableToTableStage[M[_]: MonadLower](inputIR: TableIR, analyses: LoweringAnalyses)
  : M[TableStage] =
    LowerTableIR.applyTable(inputIR, DArrayLowering.All, analyses)
}
