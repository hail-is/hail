package is.hail.backend.local

import is.hail.{HailContext, HailFeatureFlags}
import is.hail.annotations.{Region, SafeRow, UnsafeRow}
import is.hail.asm4s._
import is.hail.backend._
import is.hail.expr.ir.lowering._
import is.hail.expr.ir.{IRParser, _}
import is.hail.expr.{JSONAnnotationImpex, Validate}
import is.hail.io.bgen.IndexBgen
import is.hail.io.fs.{FS, HadoopFS}
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
import org.apache.hadoop
import org.json4s.DefaultFormats
import org.json4s.jackson.{JsonMethods, Serialization}

import java.io.PrintWriter
import scala.collection.JavaConverters._
import scala.reflect.ClassTag

class LocalBroadcastValue[T](val value: T) extends BroadcastValue[T] with Serializable

class LocalTaskContext(val partitionId: Int, val stageId: Int) extends HailTaskContext {
  override def attemptNumber(): Int = 0
}

object LocalBackend {
  private var theLocalBackend: LocalBackend = _

  def apply(tmpdir: String): LocalBackend = synchronized {
    require(theLocalBackend == null)

    theLocalBackend = new LocalBackend(tmpdir)
    theLocalBackend
  }

  def stop(): Unit = synchronized {
    if (theLocalBackend != null) {
      theLocalBackend = null
    }
  }
}

class LocalBackend(
  val tmpdir: String
) extends Backend {
  // FIXME don't rely on hadoop
  val hadoopConf = new hadoop.conf.Configuration()
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

  def withExecuteContext[T](timer: ExecutionTimer)(f: ExecuteContext => T): T = {
    ExecuteContext.scoped(tmpdir, tmpdir, this, fs, timer, null, theHailClassLoader, flags)(f)
  }

  def broadcast[T: ClassTag](value: T): BroadcastValue[T] = new LocalBroadcastValue[T](value)

  private[this] var stageIdx: Int = 0

  private[this] def nextStageId(): Int = {
    val current = stageIdx
    stageIdx += 1
    current
  }

  def parallelizeAndComputeWithIndex(backendContext: BackendContext, fs: FS, collection: Array[Array[Byte]], dependency: Option[TableStageDependency] = None)(f: (Array[Byte], HailTaskContext, HailClassLoader, FS) => Array[Byte]): Array[Array[Byte]] = {
    val stageId = nextStageId()
    collection.zipWithIndex.map { case (c, i) =>
      val htc = new LocalTaskContext(i, stageId)
      val bytes = f(c, htc, theHailClassLoader, fs)
      htc.finish()
      bytes
    }
  }

  def defaultParallelism: Int = 1

  def stop(): Unit = LocalBackend.stop()

  private[this] def _jvmLowerAndExecute(ctx: ExecuteContext, ir0: IR, print: Option[PrintWriter] = None): (Option[SingleCodeType], Long) = {
    val ir = LoweringPipeline.darrayLowerer(true)(DArrayLowering.All).apply(ctx, ir0).asInstanceOf[IR]

    if (!Compilable(ir))
      throw new LowererUnsupportedOperation(s"lowered to uncompilable IR: ${ Pretty(ctx, ir) }")

    if (ir.typ == TVoid) {
      val (pt, f) = ctx.timer.time("Compile") {
        Compile[AsmFunction1RegionUnit](ctx,
          FastIndexedSeq(),
          FastIndexedSeq(classInfo[Region]), UnitInfo,
          ir,
          print = print)
      }

      ctx.timer.time("Run") {
        f(ctx.theHailClassLoader, fs, 0, ctx.r).apply(ctx.r)
        (pt, 0)
      }
    } else {
      val (pt, f) = ctx.timer.time("Compile") {
        Compile[AsmFunction1RegionLong](ctx,
          FastIndexedSeq(),
          FastIndexedSeq(classInfo[Region]), LongInfo,
          MakeTuple.ordered(FastSeq(ir)),
          print = print)
      }

      ctx.timer.time("Run") {
        (pt, f(ctx.theHailClassLoader, fs, 0, ctx.r).apply(ctx.r))
      }
    }
  }

  private[this] def _execute(ctx: ExecuteContext, ir: IR): (Option[SingleCodeType], Long) = {
    TypeCheck(ctx, ir)
    Validate(ir)
    val queryID = Backend.nextID()
    log.info(s"starting execution of query $queryID of initial size ${ IRSize(ir) }")
    val res = _jvmLowerAndExecute(ctx, ir)
    log.info(s"finished execution of query $queryID")
    res
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

  def executeEncode(ir: IR, bufferSpecString: String): (Array[Byte], String) = {
    val (bytes, timer) = ExecutionTimer.time("LocalBackend.encodeToBytes") { timer =>
      val bs = BufferSpec.parseOrDefault(bufferSpecString)
      withExecuteContext(timer) { ctx =>
        executeToEncoded(timer, ir, bs)
      }
    }
    (bytes, Serialization.write(Map("timings" -> timer.toMap))(new DefaultFormats {}))
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

  def pyIndexBgen(
    files: java.util.List[String],
    indexFileMap: java.util.Map[String, String],
    rg: String,
    contigRecoding: java.util.Map[String, String],
    skipInvalidLoci: Boolean) {
    ExecutionTimer.logTime("LocalBackend.pyIndexBgen") { timer =>
      withExecuteContext(timer) { ctx =>
        IndexBgen(ctx, files.asScala.toArray, indexFileMap.asScala.toMap, Option(rg), contigRecoding.asScala.toMap, skipInvalidLoci)
      }
      info(s"Number of BGEN files indexed: ${ files.size() }")
    }
  }

  def pyReferenceAddLiftover(name: String, chainFile: String, destRGName: String): Unit = {
    ExecutionTimer.logTime("LocalBackend.pyReferenceAddLiftover") { timer =>
      withExecuteContext(timer) { ctx =>
        ReferenceGenome.referenceAddLiftover(ctx, name, chainFile, destRGName)
      }
    }
  }

  def pyFromFASTAFile(name: String, fastaFile: String, indexFile: String,
    xContigs: java.util.List[String], yContigs: java.util.List[String], mtContigs: java.util.List[String],
    parInput: java.util.List[String]): ReferenceGenome = {
    ExecutionTimer.logTime("LocalBackend.pyFromFASTAFile") { timer =>
      withExecuteContext(timer) { ctx =>
        ReferenceGenome.fromFASTAFile(ctx, name, fastaFile, indexFile,
          xContigs.asScala.toArray, yContigs.asScala.toArray, mtContigs.asScala.toArray, parInput.asScala.toArray)
      }
    }
  }

  def pyAddSequence(name: String, fastaFile: String, indexFile: String): Unit = {
    ExecutionTimer.logTime("LocalBackend.pyAddSequence") { timer =>
      withExecuteContext(timer) { ctx =>
        ReferenceGenome.addSequence(ctx, name, fastaFile, indexFile)
      }
    }
  }

  def parse_value_ir(s: String, refMap: java.util.Map[String, String], irMap: java.util.Map[String, BaseIR]): IR = {
    ExecutionTimer.logTime("LocalBackend.parse_value_ir") { timer =>
      withExecuteContext(timer) { ctx =>
        IRParser.parse_value_ir(s, IRParserEnvironment(ctx, refMap.asScala.toMap.mapValues(IRParser.parseType), irMap.asScala.toMap))
      }
    }
  }

  def parse_table_ir(s: String, refMap: java.util.Map[String, String], irMap: java.util.Map[String, BaseIR]): TableIR = {
    ExecutionTimer.logTime("LocalBackend.parse_table_ir") { timer =>
      withExecuteContext(timer) { ctx =>
        IRParser.parse_table_ir(s, IRParserEnvironment(ctx, refMap.asScala.toMap.mapValues(IRParser.parseType), irMap.asScala.toMap))
      }
    }
  }

  def parse_matrix_ir(s: String, refMap: java.util.Map[String, String], irMap: java.util.Map[String, BaseIR]): MatrixIR = {
    ExecutionTimer.logTime("LocalBackend.parse_matrix_ir") { timer =>
      withExecuteContext(timer) { ctx =>
        IRParser.parse_matrix_ir(s, IRParserEnvironment(ctx, refMap.asScala.toMap.mapValues(IRParser.parseType), irMap.asScala.toMap))
      }
    }
  }

  def parse_blockmatrix_ir(
    s: String, refMap: java.util.Map[String, String], irMap: java.util.Map[String, BaseIR]
  ): BlockMatrixIR = {
    ExecutionTimer.logTime("LocalBackend.parse_blockmatrix_ir") { timer =>
      withExecuteContext(timer) { ctx =>
        IRParser.parse_blockmatrix_ir(s, IRParserEnvironment(ctx, refMap.asScala.toMap.mapValues(IRParser.parseType), irMap.asScala.toMap))
      }
    }
  }

  def lowerDistributedSort(
    ctx: ExecuteContext,
    stage: TableStage,
    sortFields: IndexedSeq[SortField],
    relationalLetsAbove: Map[String, IR],
    rowTypeRequiredness: RStruct
  ): TableStage = {

    if (getFlag("use_new_shuffle") != null) {
      LowerDistributedSort.distributedSort(ctx, stage, sortFields, relationalLetsAbove, rowTypeRequiredness)
    } else {
      LowerDistributedSort.localSort(ctx, stage, sortFields, relationalLetsAbove)
    }
  }

  def pyLoadReferencesFromDataset(path: String): String =
    ReferenceGenome.fromHailDataset(fs, path)

  def pyImportFam(path: String, isQuantPheno: Boolean, delimiter: String, missingValue: String): String =
    LoadPlink.importFamJSON(fs, path, isQuantPheno, delimiter, missingValue)

  def persist(backendContext: BackendContext, id: String, value: BlockMatrix, storageLevel: String): Unit = ???

  def unpersist(backendContext: BackendContext, id: String): Unit = ???

  def getPersistedBlockMatrix(backendContext: BackendContext, id: String): BlockMatrix = ???

  def getPersistedBlockMatrixType(backendContext: BackendContext, id: String): BlockMatrixType = ???
}
