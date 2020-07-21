package is.hail.backend.local

import is.hail.annotations.UnsafeRow
import is.hail.asm4s._
import is.hail.expr.ir.IRParser
import is.hail.types.encoded.EType
import is.hail.io.{BufferSpec, TypedCodecSpec}
import is.hail.annotations.{Region, SafeRow}
import is.hail.expr.{JSONAnnotationImpex, Validate}
import is.hail.expr.ir.lowering._
import is.hail.expr.ir._
import is.hail.types.physical.{PTuple, PType, PVoid}
import is.hail.types.virtual.TVoid
import is.hail.types.physical.{PTuple, PType}
import is.hail.types.virtual.{TVoid, Type}
import is.hail.backend.{Backend, BackendContext, BroadcastValue}
import is.hail.io.fs.{FS, HadoopFS}
import is.hail.utils._
import is.hail.io.bgen.IndexBgen
import is.hail.io.plink.LoadPlink
import is.hail.variant.ReferenceGenome
import org.apache.hadoop
import org.json4s.DefaultFormats
import org.json4s.jackson.{JsonMethods, Serialization}

import scala.reflect.ClassTag
import scala.collection.JavaConverters._
import java.io.PrintWriter

class LocalBroadcastValue[T](val value: T) extends BroadcastValue[T]

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
  
  val fs: FS = new HadoopFS(new SerializableHadoopConfiguration(hadoopConf))

  def withExecuteContext[T]()(f: ExecuteContext => T): T = {
    ExecuteContext.scoped(tmpdir, tmpdir, this, fs)(f)
  }

  def broadcast[T : ClassTag](value: T): BroadcastValue[T] = new LocalBroadcastValue[T](value)

  def parallelizeAndComputeWithIndex(backendContext: BackendContext, collection: Array[Array[Byte]])(f: (Array[Byte], Int) => Array[Byte]): Array[Array[Byte]] = {
    collection.zipWithIndex.map { case (c, i) =>
      f(c, i)
    }
  }

  def defaultParallelism: Int = 1

  def stop(): Unit = LocalBackend.stop()

  private[this] def _jvmLowerAndExecute(ctx: ExecuteContext, ir0: IR, print: Option[PrintWriter] = None): (PType, Long) = {
    val ir = LoweringPipeline.darrayLowerer(true)(DArrayLowering.All).apply(ctx, ir0).asInstanceOf[IR]

    if (!Compilable(ir))
      throw new LowererUnsupportedOperation(s"lowered to uncompilable IR: ${ Pretty(ir) }")

    if (ir.typ == TVoid) {
      val (pt, f) = ctx.timer.time("Compile") {
        Compile[AsmFunction1RegionUnit](ctx,
          FastIndexedSeq[(String, PType)](),
          FastIndexedSeq(classInfo[Region]), UnitInfo,
          ir,
          print = print)
      }

      ctx.timer.time("Run") {
        f(0, ctx.r).apply(ctx.r)
        (pt, 0)
      }
    } else {
      val (pt, f) = ctx.timer.time("Compile") {
        Compile[AsmFunction1RegionLong](ctx,
          FastIndexedSeq[(String, PType)](),
          FastIndexedSeq(classInfo[Region]), LongInfo,
          MakeTuple.ordered(FastSeq(ir)),
          print = print)
      }

      ctx.timer.time("Run") {
        (pt, f(0, ctx.r).apply(ctx.r))
      }
    }
  }

  private[this] def _execute(ctx: ExecuteContext, ir: IR): (PType, Long) = {
    TypeCheck(ir)
    Validate(ir)
    _jvmLowerAndExecute(ctx, ir)
  }

  def execute(ir: IR): (Any, ExecutionTimer) =
    withExecuteContext() { ctx =>
      val (pt, a) = _execute(ctx, ir)
      pt match {
        case PVoid =>
          (null, ctx.timer)
        case pt: PTuple =>
          (SafeRow(pt, a).get(0), ctx.timer)
      }
    }

  def executeJSON(ir: IR): String = {
    val t = ir.typ
    val (value, timings) = execute(ir)
    val jsonValue = JsonMethods.compact(JSONAnnotationImpex.exportAnnotation(value, t))
    timings.finish()
    timings.logInfo()

    Serialization.write(Map("value" -> jsonValue, "timings" -> timings.asMap()))(new DefaultFormats {})
  }

  def encodeToBytes(ir: IR, bufferSpecString: String): (String, Array[Byte]) = {
    val bs = BufferSpec.parseOrDefault(bufferSpecString)
    withExecuteContext() { ctx =>
      assert(ir.typ != TVoid)
      val (pt: PTuple, a) = _execute(ctx, ir)
      assert(pt.size == 1)
      val elementType = pt.fields(0).typ
      val codec = TypedCodecSpec(
        EType.defaultFromPType(elementType), elementType.virtualType, bs)
      assert(pt.isFieldDefined(a, 0))
      (elementType.toString, codec.encode(ctx, elementType, pt.loadField(a, 0)))
    }
  }

  def decodeToJSON(ptypeString: String, b: Array[Byte], bufferSpecString: String): String = {
    val t = IRParser.parsePType(ptypeString)
    val bs = BufferSpec.parseOrDefault(bufferSpecString)
    val codec = TypedCodecSpec(EType.defaultFromPType(t), t.virtualType, bs)
    withExecuteContext() { ctx =>
      val (pt, off) = codec.decode(ctx, t.virtualType, b, ctx.r)
      assert(pt.virtualType == t.virtualType)
      JsonMethods.compact(JSONAnnotationImpex.exportAnnotation(
        UnsafeRow.read(pt, ctx.r, off), pt.virtualType))
    }
  }

  def pyIndexBgen(
    files: java.util.List[String],
    indexFileMap: java.util.Map[String, String],
    rg: String,
    contigRecoding: java.util.Map[String, String],
    skipInvalidLoci: Boolean) {
    withExecuteContext() { ctx =>
      IndexBgen(ctx, files.asScala.toArray, indexFileMap.asScala.toMap, Option(rg), contigRecoding.asScala.toMap, skipInvalidLoci)
    }
    info(s"Number of BGEN files indexed: ${ files.size() }")
  }

  def pyReferenceAddLiftover(name: String, chainFile: String, destRGName: String): Unit = {
    withExecuteContext() { ctx =>
      ReferenceGenome.referenceAddLiftover(ctx, name, chainFile, destRGName)
    }
  }

  def pyFromFASTAFile(name: String, fastaFile: String, indexFile: String,
    xContigs: java.util.List[String], yContigs: java.util.List[String], mtContigs: java.util.List[String],
    parInput: java.util.List[String]): ReferenceGenome = {
    withExecuteContext() { ctx =>
      ReferenceGenome.fromFASTAFile(ctx, name, fastaFile, indexFile,
        xContigs.asScala.toArray, yContigs.asScala.toArray, mtContigs.asScala.toArray, parInput.asScala.toArray)
    }
  }

  def pyAddSequence(name: String, fastaFile: String, indexFile: String): Unit = {
    withExecuteContext() { ctx =>
      ReferenceGenome.addSequence(ctx, name, fastaFile, indexFile)
    }
  }

  def parse_value_ir(s: String, refMap: java.util.Map[String, String], irMap: java.util.Map[String, BaseIR]): IR = {
    withExecuteContext() { ctx =>
      IRParser.parse_value_ir(s, IRParserEnvironment(ctx, refMap.asScala.toMap.mapValues(IRParser.parseType), irMap.asScala.toMap))
    }
  }

  def parse_table_ir(s: String, refMap: java.util.Map[String, String], irMap: java.util.Map[String, BaseIR]): TableIR = {
    withExecuteContext() { ctx =>
      IRParser.parse_table_ir(s, IRParserEnvironment(ctx, refMap.asScala.toMap.mapValues(IRParser.parseType), irMap.asScala.toMap))
    }
  }

  def parse_matrix_ir(s: String, refMap: java.util.Map[String, String], irMap: java.util.Map[String, BaseIR]): MatrixIR = {
    withExecuteContext() { ctx =>
      IRParser.parse_matrix_ir(s, IRParserEnvironment(ctx, refMap.asScala.toMap.mapValues(IRParser.parseType), irMap.asScala.toMap))
    }
  }

  def parse_blockmatrix_ir(
    s: String, refMap: java.util.Map[String, String], irMap: java.util.Map[String, BaseIR]
  ): BlockMatrixIR = {
    withExecuteContext() { ctx =>
      IRParser.parse_blockmatrix_ir(s, IRParserEnvironment(ctx, refMap.asScala.toMap.mapValues(IRParser.parseType), irMap.asScala.toMap))
    }
  }

  def lowerDistributedSort(ctx: ExecuteContext, stage: TableStage, sortFields: IndexedSeq[SortField], relationalLetsAbove: Seq[(String, Type)]): TableStage = {
    // Use a local sort for the moment to enable larger pipelines to run
    LowerDistributedSort.localSort(ctx, stage, sortFields, relationalLetsAbove)
  }

  def pyLoadReferencesFromDataset(path: String): String =
    ReferenceGenome.fromHailDataset(fs, path)

  def pyImportFam(path: String, isQuantPheno: Boolean, delimiter: String, missingValue: String): String =
    LoadPlink.importFamJSON(fs, path, isQuantPheno, delimiter, missingValue)
}
