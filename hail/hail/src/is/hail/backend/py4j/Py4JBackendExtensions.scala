package is.hail.backend.py4j

import is.hail.HailFeatureFlags
import is.hail.backend.{Backend, ExecuteContext, NonOwningTempFileManager, TempFileManager}
import is.hail.expr.{JSONAnnotationImpex, SparkAnnotationImpex}
import is.hail.expr.ir.{
  BaseIR, BindingEnv, BlockMatrixIR, IR, IRParser, Interpret, MatrixIR, MatrixNativeReader,
  MatrixRead, Name, NativeReaderOptions, TableIR, TableLiteral, TableValue,
}
import is.hail.expr.ir.IRParser.parseType
import is.hail.expr.ir.defs.{EncodedLiteral, GetFieldByIdx}
import is.hail.expr.ir.functions.IRFunctionRegistry
import is.hail.io.reference.{IndexedFastaSequenceFile, LiftOver}
import is.hail.linalg.RowMatrix
import is.hail.types.physical.PStruct
import is.hail.types.virtual.{TArray, TInterval}
import is.hail.utils.{defaultJSONFormats, log, toRichIterable, FastSeq, HailException, Interval}
import is.hail.variant.ReferenceGenome

import scala.collection.mutable
import scala.jdk.CollectionConverters.{
  asScalaBufferConverter, mapAsScalaMapConverter, seqAsJavaListConverter,
}

import java.nio.charset.StandardCharsets
import java.util

import org.apache.spark.sql.DataFrame
import org.json4s
import org.json4s.Formats
import org.json4s.jackson.{JsonMethods, Serialization}
import sourcecode.Enclosing

trait Py4JBackendExtensions {
  def backend: Backend
  def references: mutable.Map[String, ReferenceGenome]
  def flags: HailFeatureFlags
  def longLifeTempFileManager: TempFileManager

  def pyGetFlag(name: String): String =
    flags.get(name)

  def pySetFlag(name: String, value: String): Unit =
    flags.set(name, value)

  def pyAvailableFlags: java.util.ArrayList[String] =
    flags.available

  private[this] var irID: Int = 0

  private[this] def nextIRID(): Int = {
    irID += 1
    irID
  }

  private[this] def addJavaIR(ctx: ExecuteContext, ir: BaseIR): Int = {
    val id = nextIRID()
    ctx.PersistedIrCache += (id -> ir)
    id
  }

  def pyRemoveJavaIR(id: Int): Unit =
    backend.withExecuteContext(_.PersistedIrCache.remove(id))

  def pyAddSequence(name: String, fastaFile: String, indexFile: String): Unit =
    backend.withExecuteContext { ctx =>
      references(name).addSequence(IndexedFastaSequenceFile(ctx.fs, fastaFile, indexFile))
    }

  def pyRemoveSequence(name: String): Unit =
    references(name).removeSequence()

  def pyExportBlockMatrix(
    pathIn: String,
    pathOut: String,
    delimiter: String,
    header: String,
    addIndex: Boolean,
    exportType: String,
    partitionSize: java.lang.Integer,
    entries: String,
  ): Unit =
    backend.withExecuteContext { ctx =>
      val rm = RowMatrix.readBlockMatrix(ctx.fs, pathIn, partitionSize)
      entries match {
        case "full" =>
          rm.export(ctx, pathOut, delimiter, Option(header), addIndex, exportType)
        case "lower" =>
          rm.exportLowerTriangle(ctx, pathOut, delimiter, Option(header), addIndex, exportType)
        case "strict_lower" =>
          rm.exportStrictLowerTriangle(
            ctx,
            pathOut,
            delimiter,
            Option(header),
            addIndex,
            exportType,
          )
        case "upper" =>
          rm.exportUpperTriangle(ctx, pathOut, delimiter, Option(header), addIndex, exportType)
        case "strict_upper" =>
          rm.exportStrictUpperTriangle(
            ctx,
            pathOut,
            delimiter,
            Option(header),
            addIndex,
            exportType,
          )
      }
    }

  def pyRegisterIR(
    name: String,
    typeParamStrs: java.util.ArrayList[String],
    argNameStrs: java.util.ArrayList[String],
    argTypeStrs: java.util.ArrayList[String],
    returnType: String,
    bodyStr: String,
  ): Unit =
    backend.withExecuteContext { ctx =>
      IRFunctionRegistry.registerIR(
        ctx,
        name,
        typeParamStrs.asScala.toArray,
        argNameStrs.asScala.toArray,
        argTypeStrs.asScala.toArray,
        returnType,
        bodyStr,
      )
    }

  def pyExecuteLiteral(irStr: String): Int =
    backend.withExecuteContext { ctx =>
      val ir = IRParser.parse_value_ir(ctx, irStr)
      assert(ir.typ.isRealizable)
      backend.execute(ctx, ir) match {
        case Left(_) => throw new HailException("Can't create literal")
        case Right((pt, addr)) =>
          val field = GetFieldByIdx(EncodedLiteral.fromPTypeAndAddress(pt, addr, ctx), 0)
          addJavaIR(ctx, field)
      }
    }

  def pyFromDF(df: DataFrame, jKey: java.util.List[String]): (Int, String) = {
    val key = jKey.asScala.toArray.toFastSeq
    val signature =
      SparkAnnotationImpex.importType(df.schema).setRequired(true).asInstanceOf[PStruct]
    withExecuteContext(selfContainedExecution = false) { ctx =>
      val tir = TableLiteral(
        TableValue(
          ctx,
          signature.virtualType,
          key,
          df.rdd,
          Some(signature),
        ),
        ctx.theHailClassLoader,
      )
      val id = addJavaIR(ctx, tir)
      (id, JsonMethods.compact(tir.typ.toJSON))
    }
  }

  def pyToDF(s: String): DataFrame =
    withExecuteContext(selfContainedExecution = false) { ctx =>
      val tir = IRParser.parse_table_ir(ctx, s)
      Interpret(tir, ctx).toDF()
    }

  def pyReadMultipleMatrixTables(jsonQuery: String): util.List[MatrixIR] =
    backend.withExecuteContext { ctx =>
      log.info("pyReadMultipleMatrixTables: got query")
      val kvs = JsonMethods.parse(jsonQuery) match {
        case json4s.JObject(values) => values.toMap
      }

      val paths = kvs("paths").asInstanceOf[json4s.JArray].arr.toArray.map {
        case json4s.JString(s) => s
      }

      val intervalPointType = parseType(kvs("intervalPointType").asInstanceOf[json4s.JString].s)
      val intervalObjects =
        JSONAnnotationImpex.importAnnotation(kvs("intervals"), TArray(TInterval(intervalPointType)))
          .asInstanceOf[IndexedSeq[Interval]]

      val opts = NativeReaderOptions(intervalObjects, intervalPointType)
      val matrixReaders: IndexedSeq[MatrixIR] = paths.map { p =>
        log.info(s"creating MatrixRead node for $p")
        val mnr = MatrixNativeReader(ctx.fs, p, Some(opts))
        MatrixRead(mnr.fullMatrixTypeWithoutUIDs, false, false, mnr): MatrixIR
      }
      log.info("pyReadMultipleMatrixTables: returning N matrix tables")
      matrixReaders.asJava
    }

  def pyAddReference(jsonConfig: String): Unit =
    addReference(ReferenceGenome.fromJSON(jsonConfig))

  def pyRemoveReference(name: String): Unit =
    removeReference(name)

  def pyAddLiftover(name: String, chainFile: String, destRGName: String): Unit =
    backend.withExecuteContext { ctx =>
      references(name).addLiftover(references(destRGName), LiftOver(ctx.fs, chainFile))
    }

  def pyRemoveLiftover(name: String, destRGName: String): Unit =
    references(name).removeLiftover(destRGName)

  private[this] def addReference(rg: ReferenceGenome): Unit =
    ReferenceGenome.addFatalOnCollision(references, FastSeq(rg))

  private[this] def removeReference(name: String): Unit =
    references -= name

  def parse_value_ir(s: String, refMap: java.util.Map[String, String]): IR =
    backend.withExecuteContext { ctx =>
      IRParser.parse_value_ir(
        ctx,
        s,
        BindingEnv.eval(refMap.asScala.toMap.map { case (n, t) =>
          Name(n) -> IRParser.parseType(t)
        }.toSeq: _*),
      )
    }

  def parse_table_ir(s: String): TableIR =
    withExecuteContext(selfContainedExecution = false)(ctx => IRParser.parse_table_ir(ctx, s))

  def parse_matrix_ir(s: String): MatrixIR =
    withExecuteContext(selfContainedExecution = false)(ctx => IRParser.parse_matrix_ir(ctx, s))

  def parse_blockmatrix_ir(s: String): BlockMatrixIR =
    withExecuteContext(selfContainedExecution = false) { ctx =>
      IRParser.parse_blockmatrix_ir(ctx, s)
    }

  def loadReferencesFromDataset(path: String): Array[Byte] =
    backend.withExecuteContext { ctx =>
      val rgs = ReferenceGenome.fromHailDataset(ctx.fs, path)
      ReferenceGenome.addFatalOnCollision(references, rgs)

      implicit val formats: Formats = defaultJSONFormats
      Serialization.write(rgs.map(_.toJSON).toFastSeq).getBytes(StandardCharsets.UTF_8)
    }

  def withExecuteContext[T](
    selfContainedExecution: Boolean = true
  )(
    f: ExecuteContext => T
  )(implicit E: Enclosing
  ): T =
    backend.withExecuteContext { ctx =>
      val tempFileManager = longLifeTempFileManager
      if (selfContainedExecution && tempFileManager != null) f(ctx)
      else ctx.local(tempFileManager = NonOwningTempFileManager(tempFileManager))(f)
    }
}
