package is.hail.backend.driver

import is.hail.{linalg, HailFeatureFlags}
import is.hail.asm4s.HailClassLoader
import is.hail.backend._
import is.hail.backend.spark.SparkBackend
import is.hail.expr.{JSONAnnotationImpex, SparkAnnotationImpex}
import is.hail.expr.ir._
import is.hail.expr.ir.IRParser.parseType
import is.hail.expr.ir.LoweredTableReader.LoweredTableReaderCoercer
import is.hail.expr.ir.defs.{EncodedLiteral, GetFieldByIdx}
import is.hail.expr.ir.functions.IRFunctionRegistry
import is.hail.expr.ir.lowering.IrMetadata
import is.hail.io.fs._
import is.hail.io.reference.{IndexedFastaSequenceFile, LiftOver}
import is.hail.macros.void
import is.hail.types.physical.PStruct
import is.hail.types.virtual.{TArray, TInterval}
import is.hail.types.virtual.Kinds.{BlockMatrix, Matrix, Table, Value}
import is.hail.utils._
import is.hail.utils.ExecutionTimer.Timings
import is.hail.variant.ReferenceGenome

import scala.annotation.nowarn
import scala.collection.mutable
import scala.jdk.CollectionConverters.{asScalaBufferConverter, seqAsJavaListConverter}

import java.io.Closeable
import java.net.InetSocketAddress
import java.util

import com.google.api.client.http.HttpStatusCodes
import com.sun.net.httpserver.{HttpExchange, HttpServer}
import org.apache.hadoop.conf.Configuration
import org.apache.spark.sql.DataFrame
import org.json4s._
import org.json4s.jackson.{JsonMethods, Serialization}
import sourcecode.Enclosing

final class Py4JQueryDriver(backend: Backend) extends Closeable {

  private[this] val flags: HailFeatureFlags = HailFeatureFlags.fromEnv()
  private[this] val hcl = new HailClassLoader(getClass.getClassLoader)
  private[this] val references = mutable.Map(ReferenceGenome.builtinReferences().toSeq: _*)
  private[this] val blockMatrixCache = mutable.Map[String, linalg.BlockMatrix]()
  private[this] val compiledCodeCache = new Cache[CodeCacheKey, CompiledFunction[_]](50)
  private[this] val irCache = mutable.Map[Int, BaseIR]()
  private[this] val coercerCache = new Cache[Any, LoweredTableReaderCoercer](32)
  private[this] var irID: Int = 0

  private[this] var tmpdir: String = _
  private[this] var localTmpdir: String = _

  private[this] var tmpFileManager = new OwningTempFileManager(
    newFs(CloudStorageFSConfig.fromFlagsAndEnv(None, flags))
  )

  def pyFs: FS =
    synchronized(tmpFileManager.fs)

  def pyGetFlag(name: String): String =
    synchronized(flags.get(name))

  def pySetFlag(name: String, value: String): Unit =
    synchronized(flags.set(name, value))

  def pyAvailableFlags: java.util.ArrayList[String] =
    flags.available

  def pySetRemoteTmp(tmp: String): Unit =
    synchronized { tmpdir = tmp }

  def pySetLocalTmp(tmp: String): Unit =
    synchronized { localTmpdir = tmp }

  def pySetGcsRequesterPaysConfig(project: String, buckets: util.List[String]): Unit =
    synchronized {
      tmpFileManager.close()

      val cloudfsConf = CloudStorageFSConfig.fromFlagsAndEnv(None, flags)

      val rpConfig: Option[RequesterPaysConfig] =
        (
          Option(project).filter(_.nonEmpty),
          Option(buckets).map(_.asScala.toSet.filterNot(_.isBlank)).filter(_.nonEmpty),
        ) match {
          case (Some(project), buckets) => Some(RequesterPaysConfig(project, buckets))
          case (None, Some(_)) => fatal(
              "A non-empty, non-null requester pays google project is required to configure requester pays buckets."
            )
          case (None, None) => None
        }

      val fs = newFs(
        cloudfsConf.copy(
          google = (cloudfsConf.google, rpConfig) match {
            case (Some(gconf), _) => Some(gconf.copy(requester_pays_config = rpConfig))
            case (None, Some(_)) => Some(GoogleStorageFSConfig(None, rpConfig))
            case _ => None
          }
        )
      )

      tmpFileManager = new OwningTempFileManager(fs)
    }

  def pyRemoveJavaIR(id: Int): Unit =
    synchronized {
      void(irCache.remove(id))
    }

  def pyAddSequence(name: String, fastaFile: String, indexFile: String): Unit =
    synchronized {
      val seq = IndexedFastaSequenceFile(tmpFileManager.fs, fastaFile, indexFile)
      references(name).addSequence(seq)
    }

  def pyRemoveSequence(name: String): Unit =
    synchronized(references(name).removeSequence())

  def pyExportBlockMatrix(
    pathIn: String,
    pathOut: String,
    delimiter: String,
    header: String,
    addIndex: Boolean,
    exportType: String,
    partitionSize: java.lang.Integer,
    entries: String,
  ): Unit = {
    void {
      withExecuteContext() { ctx =>
        val rm = linalg.RowMatrix.readBlockMatrix(ctx.fs, pathIn, partitionSize)
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
    void {
      withExecuteContext() { ctx =>
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
    }

  def pyExecuteLiteral(irStr: String): Int =
    withExecuteContext(selfContainedExecution = false) { ctx =>
      val ir = IRParser.parse_value_ir(ctx, irStr)
      assert(ir.typ.isRealizable)
      backend.execute(ctx, ir) match {
        case Left(_) => throw new HailException("Can't create literal")
        case Right((pt, addr)) =>
          val field = GetFieldByIdx(EncodedLiteral.fromPTypeAndAddress(pt, addr, ctx), 0)
          addJavaIR(ctx, field)
      }
    }._1

  def pyFromDF(df: DataFrame, jKey: java.util.List[String]): (Int, String) =
    withExecuteContext(selfContainedExecution = false) { ctx =>
      val key = jKey.asScala.toArray.toFastSeq
      val signature =
        SparkAnnotationImpex.importType(df.schema).setRequired(true).asInstanceOf[PStruct]
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
    }._1

  def pyToDF(s: String): DataFrame =
    withExecuteContext(selfContainedExecution = false) { ctx =>
      val tir = IRParser.parse_table_ir(ctx, s)
      Interpret(tir, ctx).toDF()
    }._1

  def pyReadMultipleMatrixTables(jsonQuery: String): util.List[MatrixIR] =
    withExecuteContext(selfContainedExecution = false) { ctx =>
      implicit val fmts: Formats = DefaultFormats
      log.info("pyReadMultipleMatrixTables: got query")

      val kvs = JsonMethods.parse(jsonQuery).extract[Map[String, JValue]]
      val paths = kvs("paths").extract[IndexedSeq[String]]
      val intervalPointType = parseType(kvs("intervalPointType").extract[String])
      val intervalObjects =
        JSONAnnotationImpex.importAnnotation(kvs("intervals"), TArray(TInterval(intervalPointType)))
          .asInstanceOf[IndexedSeq[Interval]]

      val opts = NativeReaderOptions(intervalObjects, intervalPointType)
      val matrixReaders: util.List[MatrixIR] =
        paths.map { p =>
          log.info(s"creating MatrixRead node for $p")
          val mnr = MatrixNativeReader(ctx.fs, p, Some(opts))
          MatrixRead(mnr.fullMatrixTypeWithoutUIDs, false, false, mnr): MatrixIR
        }.asJava

      log.info("pyReadMultipleMatrixTables: returning N matrix tables")
      matrixReaders
    }._1

  def pyAddReference(jsonConfig: String): Unit =
    synchronized(addReference(ReferenceGenome.fromJSON(jsonConfig)))

  def pyRemoveReference(name: String): Unit =
    synchronized(removeReference(name))

  def pyAddLiftover(name: String, chainFile: String, destRGName: String): Unit =
    synchronized {
      references(name).addLiftover(
        references(destRGName),
        LiftOver(tmpFileManager.fs, chainFile),
      )
    }

  def pyRemoveLiftover(name: String, destRGName: String): Unit =
    synchronized(references(name).removeLiftover(destRGName))

  def parse_blockmatrix_ir(s: String): BlockMatrixIR =
    withExecuteContext(selfContainedExecution = false) { ctx =>
      IRParser.parse_blockmatrix_ir(ctx, s)
    }._1

  private[this] def addReference(rg: ReferenceGenome): Unit =
    ReferenceGenome.addFatalOnCollision(references, FastSeq(rg))

  override def close(): Unit =
    synchronized {
      blockMatrixCache.clear()
      compiledCodeCache.clear()
      irCache.clear()
      coercerCache.clear()
    }

  private[this] def removeReference(name: String): Unit =
    references -= name

  private[this] def withExecuteContext[T](
    selfContainedExecution: Boolean = true
  )(
    f: ExecuteContext => T
  )(implicit E: Enclosing
  ): (T, Timings) =
    synchronized {
      ExecutionTimer.time { timer =>
        ExecuteContext.scoped(
          tmpdir = tmpdir,
          localTmpdir = localTmpdir,
          backend = backend,
          references = references.toMap,
          fs = tmpFileManager.fs,
          timer = timer,
          tempFileManager =
            if (!selfContainedExecution) NonOwningTempFileManager(tmpFileManager)
            else new OwningTempFileManager(tmpFileManager.fs),
          theHailClassLoader = hcl,
          flags = flags,
          irMetadata = new IrMetadata(),
          blockMatrixCache = blockMatrixCache,
          codeCache = compiledCodeCache,
          irCache = irCache,
          coercerCache = coercerCache,
        )(f)
      }
    }

  private[this] def newFs(cloudfsConfig: CloudStorageFSConfig): FS =
    backend match {
      case s: SparkBackend =>
        val conf = new Configuration(s.sc.hadoopConfiguration)
        cloudfsConfig.google.flatMap(_.requester_pays_config).foreach {
          case RequesterPaysConfig(prj, bkts) =>
            bkts
              .map { buckets =>
                conf.set("fs.gs.requester.pays.mode", "CUSTOM")
                conf.set("fs.gs.requester.pays.project.id", prj)
                conf.set("fs.gs.requester.pays.buckets", buckets.mkString(","))
              }
              .getOrElse {
                conf.set("fs.gs.requester.pays.mode", "AUTO")
                conf.set("fs.gs.requester.pays.project.id", prj)
              }
        }
        new HadoopFS(new SerializableHadoopConfiguration(conf))

      case _ =>
        RouterFS.buildRoutes(cloudfsConfig)
    }

  private[this] def nextIRID(): Int = {
    irID += 1
    irID
  }

  private[this] def addJavaIR(ctx: ExecuteContext, ir: BaseIR): Int = {
    val id = nextIRID()
    ctx.PersistedIrCache += (id -> ir)
    id
  }

  def pyHttpServer: HttpLikeRpc with Closeable =
    new HttpLikeRpc with Closeable {

      override type Env = HttpExchange

      implicit object Request extends HttpLikeRequest {
        override def route(req: HttpExchange): Route =
          req.getRequestURI.getPath match {
            case "/value/type" => Routes.TypeOf(Value)
            case "/table/type" => Routes.TypeOf(Table)
            case "/matrixtable/type" => Routes.TypeOf(Matrix)
            case "/blockmatrix/type" => Routes.TypeOf(BlockMatrix)
            case "/execute" => Routes.Execute
            case "/vcf/metadata/parse" => Routes.ParseVcfMetadata
            case "/fam/import" => Routes.ImportFam
            case "/references/load" => Routes.LoadReferencesFromDataset
            case "/references/from_fasta" => Routes.LoadReferencesFromFASTA
          }

        override def payload(req: HttpExchange): JValue =
          using(req.getRequestBody)(JsonMethods.parse(_))

        override def timings(req: HttpExchange, t: Timings): Unit = {
          val ts = Serialization.write(Map("timings" -> t))
          req.getResponseHeaders.add("X-Hail-Timings", ts)
        }

        override def result(req: HttpExchange, result: Array[Byte]): Unit =
          respond(req, HttpStatusCodes.STATUS_CODE_OK, result)

        override def failure(req: HttpExchange, t: Throwable): Unit =
          respond(
            req,
            HttpStatusCodes.STATUS_CODE_SERVER_ERROR,
            jsonToBytes {
              val (shortMessage, expandedMessage, errorId) = handleForPython(t)
              JObject(
                "short" -> JString(shortMessage),
                "expanded" -> JString(expandedMessage),
                "error_id" -> JInt(errorId),
              )
            },
          )

        private[this] def respond(req: HttpExchange, code: Int, payload: Array[Byte]): Unit = {
          req.sendResponseHeaders(code, payload.length)
          using(req.getResponseBody)(_.write(payload))
        }
      }

      implicit object Context extends Context {
        override def scoped[A](req: HttpExchange)(f: ExecuteContext => A): (A, Timings) =
          withExecuteContext()(f)

        override def putReferences(req: HttpExchange)(refs: Iterable[ReferenceGenome]): Unit =
          refs.foreach(addReference)
      }

      // 0 => let the OS pick an available port
      private[this] val httpServer = HttpServer.create(new InetSocketAddress(0), 10)

      @nowarn def port: Int = httpServer.getAddress.getPort
      override def close(): Unit = httpServer.stop(10)

      // This HTTP server *must not* start non-daemon threads because such threads keep the JVM
      // alive. A living JVM indicates to Spark that the job is incomplete. This does not manifest
      // when you run jobs in a local pyspark (because you'll Ctrl-C out of Python regardless of
      // the JVM's state) nor does it manifest in a Notebook (again, you'll kill the Notebook kernel
      // explicitly regardless of the JVM). It *does* manifest when submitting jobs with
      //
      //     gcloud dataproc submit ...
      //
      // or
      //
      //     spark-submit
      httpServer.createContext("/", runRpc(_: HttpExchange))
      httpServer.setExecutor(null) // ensures the server creates no new threads
      httpServer.start()
    }
}
