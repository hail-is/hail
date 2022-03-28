package is.hail.backend.service

import java.io._
import java.nio.charset._
import java.net._
import java.nio.charset.StandardCharsets
import java.util.concurrent._

import is.hail.{HAIL_REVISION, HailContext, HailFeatureFlags}
import is.hail.annotations._
import is.hail.asm4s._
import is.hail.backend.{Backend, BackendContext, BroadcastValue, ExecuteContext, HailTaskContext}
import is.hail.expr.JSONAnnotationImpex
import is.hail.expr.ir.lowering._
import is.hail.expr.ir.{Compile, IR, IRParser, MakeTuple, SortField}
import is.hail.expr.ir.functions.IRFunctionRegistry
import is.hail.io.{BufferSpec, TypedCodecSpec}
import is.hail.io.bgen.IndexBgen
import is.hail.io.fs._
import is.hail.io.bgen.IndexBgen
import is.hail.io.plink.LoadPlink
import is.hail.io.vcf.LoadVCF
import is.hail.linalg.BlockMatrix
import is.hail.services._
import is.hail.services.batch_client.BatchClient
import is.hail.types._
import is.hail.types.physical._
import is.hail.types.physical.stypes.PTypeReferenceSingleCodeType
import is.hail.types.virtual._
import is.hail.types.encoded._
import is.hail.utils._
import is.hail.variant.ReferenceGenome
import org.apache.commons.io.IOUtils
import org.apache.log4j.Logger
import org.json4s.Extraction
import org.json4s.JsonAST._
import org.json4s.jackson.JsonMethods
import org.json4s.{DefaultFormats, Formats}
import org.newsclub.net.unix.{AFUNIXServerSocket, AFUNIXSocketAddress}

import scala.annotation.switch
import scala.reflect.ClassTag
import scala.{concurrent => scalaConcurrent}
import scala.collection.mutable


class ServiceBackendContext(
  @transient val sessionID: String,
  val billingProject: String,
  val remoteTmpDir: String
) extends BackendContext with Serializable {
  def tokens(): Tokens =
    new Tokens(Map((DeployConfig.get.defaultNamespace, sessionID)))
}

object ServiceBackend {
  private val log = Logger.getLogger(getClass.getName())
}

class ServiceBackend(
  val jarLocation: String,
  var name: String,
  val theHailClassLoader: HailClassLoader,
  val scratchDir: String = sys.env.get("HAIL_WORKER_SCRATCH_DIR").getOrElse("")
) extends Backend {
  import ServiceBackend.log

  private[this] var batchCount = 0
  private[this] implicit val ec = scalaConcurrent.ExecutionContext.fromExecutorService(
    Executors.newCachedThreadPool())
  private[this] val MAX_AVAILABLE_GCS_CONNECTIONS = 100
  private[this] val availableGCSConnections = new Semaphore(MAX_AVAILABLE_GCS_CONNECTIONS, true)

  def defaultParallelism: Int = 10

  def broadcast[T: ClassTag](_value: T): BroadcastValue[T] = {
    using(new ObjectOutputStream(new ByteArrayOutputStream())) { os =>
      try {
        os.writeObject(_value)
      } catch {
        case e: Exception =>
          fatal(_value.toString, e)
      }
    }
    new BroadcastValue[T] with Serializable {
      def value: T = _value
    }
  }

  private[this] def readString(in: DataInputStream): String = {
    val n = in.readInt()
    val bytes = new Array[Byte](n)
    in.read(bytes)
    new String(bytes, StandardCharsets.UTF_8)
  }

  def parallelizeAndComputeWithIndex(
    _backendContext: BackendContext,
    _fs: FS,
    collection: Array[Array[Byte]],
    dependency: Option[TableStageDependency] = None
  )(f: (Array[Byte], HailTaskContext, HailClassLoader, FS) => Array[Byte]
  ): Array[Array[Byte]] = {
    val backendContext = _backendContext.asInstanceOf[ServiceBackendContext]
    val fs = _fs.asInstanceOf[ServiceCacheableFS]
    val n = collection.length
    val token = tokenUrlSafe(32)
    val root = s"${ backendContext.remoteTmpDir }parallelizeAndComputeWithIndex/$token"

    // FIXME: HACK
    val (open, create) = if (n <= 50) {
      (fs.openCachedNoCompression _, fs.createCachedNoCompression _)
    } else {
      (fs.openNoCompression _, fs.createNoCompression _)
    }

    log.info(s"parallelizeAndComputeWithIndex: $token: nPartitions $n")
    log.info(s"parallelizeAndComputeWithIndex: $token: writing f and contexts")

    val uploadFunction = scalaConcurrent.Future {
      retryTransientErrors {
        using(new ObjectOutputStream(create(s"$root/f"))) { os =>
          os.writeObject(f)
        }
      }
    }

    val uploadContexts = scalaConcurrent.Future {
      retryTransientErrors {
        using(create(s"$root/contexts")) { os =>
          var o = 12L * n
          var i = 0
          while (i < n) {
            val len = collection(i).length
            os.writeLong(o)
            os.writeInt(len)
            i += 1
            o += len
          }
          log.info(s"parallelizeAndComputeWithIndex: $token: writing contexts")
          collection.foreach { context =>
            os.write(context)
          }
        }
      }
    }

    scalaConcurrent.Await.result(uploadFunction, scalaConcurrent.duration.Duration.Inf)
    scalaConcurrent.Await.result(uploadContexts, scalaConcurrent.duration.Duration.Inf)

    val batchClient = BatchClient.fromSessionID(backendContext.sessionID)
    val jobs = new Array[JObject](n)
    var i = 0
    while (i < n) {
      jobs(i) = JObject(
        "always_run" -> JBool(false),
        "job_id" -> JInt(i + 1),
        "parent_ids" -> JArray(List()),
        "process" -> JObject(
          "jar_spec" -> JObject(
            "type" -> JString("jar_url"),
            "value" -> JString(jarLocation)
          ),
          "command" -> JArray(List(
            JString(Main.WORKER),
            JString(root),
            JString(s"$i"),
            JString(s"$n"))),
          "type" -> JString("jvm")),
        "mount_tokens" -> JBool(true),
        "resources" -> JObject("preemptible" -> JBool(true))
      )
      i += 1
    }

    log.info(s"parallelizeAndComputeWithIndex: $token: running job")

    val batchId = batchClient.create(
      JObject(
        "billing_project" -> JString(backendContext.billingProject),
        "n_jobs" -> JInt(n),
        "token" -> JString(token),
        "attributes" -> JObject("name" -> JString(name + "_" + batchCount))),
      jobs)

    val batch = batchClient.waitForBatch(batchId)
    batchCount += 1
    implicit val formats: Formats = DefaultFormats
    val batchID = (batch \ "id").extract[Int]
    val batchState = (batch \ "state").extract[String]
    if (batchState != "success") {
      throw new HailBatchFailure(s"$batchID")
    }

    log.info(s"parallelizeAndComputeWithIndex: $token: reading results")

    val r = new Array[Array[Byte]](n)

    def resultOrHailException(is: DataInputStream): Array[Byte] = {
      val success = is.readBoolean()
      if (success) {
        IOUtils.toByteArray(is)
      } else {
        val shortMessage = readString(is)
        val expandedMessage = readString(is)
        val errorId = is.readInt()
        throw new HailWorkerException(shortMessage, expandedMessage, errorId)
      }
    }

    def readResult(i: Int): scalaConcurrent.Future[Unit] = scalaConcurrent.Future {
      availableGCSConnections.acquire()
      try {
        r(i) = retryTransientErrors {
          using(open(s"$root/result.$i")) { is =>
            resultOrHailException(new DataInputStream(is))
          }
        }
        log.info(s"result $i complete")
      } finally {
        availableGCSConnections.release()
      }
    }

    scalaConcurrent.Await.result(
      scalaConcurrent.Future.sequence(
        Array.tabulate(n)(readResult).toFastIndexedSeq),
      scalaConcurrent.duration.Duration.Inf)

    log.info(s"all results complete")
    r
  }

  def stop(): Unit = ()

  def valueType(
    ctx: ExecuteContext,
    s: String
  ): String = {
    val x = IRParser.parse_value_ir(ctx, s)
    x.typ.toString
  }

  def tableType(
    ctx: ExecuteContext,
    s: String
  ): String =  {
    val x = IRParser.parse_table_ir(ctx, s)
    val t = x.typ
    val jv = JObject("global" -> JString(t.globalType.toString),
      "row" -> JString(t.rowType.toString),
      "row_key" -> JArray(t.key.map(f => JString(f)).toList))
    JsonMethods.compact(jv)
  }

  def matrixTableType(
    ctx: ExecuteContext,
    s: String
  ): String = {
    val x = IRParser.parse_matrix_ir(ctx, s)
    val t = x.typ
    val jv = JObject("global" -> JString(t.globalType.toString),
      "col" -> JString(t.colType.toString),
      "col_key" -> JArray(t.colKey.map(f => JString(f)).toList),
      "row" -> JString(t.rowType.toString),
      "row_key" -> JArray(t.rowKey.map(f => JString(f)).toList),
      "entry" -> JString(t.entryType.toString))
    JsonMethods.compact(jv)
  }

  def blockMatrixType(
    ctx: ExecuteContext,
    s: String
  ): String = {
    val x = IRParser.parse_blockmatrix_ir(ctx, s)
    val t = x.typ
    val jv = JObject("element_type" -> JString(t.elementType.toString),
      "shape" -> JArray(t.shape.map(s => JInt(s)).toList),
      "is_row_vector" -> JBool(t.isRowVector),
      "block_size" -> JInt(t.blockSize))
    JsonMethods.compact(jv)
  }

  def referenceGenome(
    ctx: ExecuteContext,
    name: String
  ): String = {
    ReferenceGenome.getReference(name).toJSONString
  }

  private[this] def execute(ctx: ExecuteContext, _x: IR, bufferSpecString: String): Array[Byte] = {
    // FIXME: do we need Validate(_x)?
    val x = LoweringPipeline.darrayLowerer(true)(DArrayLowering.All).apply(ctx, _x)
      .asInstanceOf[IR]
    if (x.typ == TVoid) {
      val (_, f) = Compile[AsmFunction1RegionUnit](ctx,
        FastIndexedSeq(),
        FastIndexedSeq[TypeInfo[_]](classInfo[Region]), UnitInfo,
        x,
        optimize = true)

      f(ctx.theHailClassLoader, ctx.fs, 0, ctx.r)(ctx.r)
      Array()
    } else {
      val (Some(PTypeReferenceSingleCodeType(pt)), f) = Compile[AsmFunction1RegionLong](ctx,
        FastIndexedSeq(),
        FastIndexedSeq[TypeInfo[_]](classInfo[Region]), LongInfo,
        MakeTuple.ordered(FastIndexedSeq(x)),
        optimize = true)
      val retPType = pt.asInstanceOf[PBaseStruct]
      val off = f(ctx.theHailClassLoader, ctx.fs, 0, ctx.r)(ctx.r)
      val codec = TypedCodecSpec(
        EType.fromTypeAllOptional(retPType.virtualType),
        retPType.virtualType,
        BufferSpec.parseOrDefault(bufferSpecString)
      )
      codec.encode(ctx, retPType, off)
    }
  }

  def execute(
    ctx: ExecuteContext,
    code: String,
    token: String,
    bufferSpecString: String
  ): Array[Byte] = {
    log.info(s"executing: ${token}")

    execute(ctx, IRParser.parse_value_ir(ctx, code), bufferSpecString)
  }

  def lowerDistributedSort(
    ctx: ExecuteContext,
    stage: TableStage,
    sortFields: IndexedSeq[SortField],
    relationalLetsAbove: Map[String, IR],
    rowTypeRequiredness: RStruct
  ): TableStage = {
    if (ctx.getFlag("use_new_shuffle") != null) {
      LowerDistributedSort.distributedSort(ctx, stage, sortFields, relationalLetsAbove, rowTypeRequiredness)
    } else {
      LowerDistributedSort.localSort(ctx, stage, sortFields, relationalLetsAbove)
    }
  }

  def persist(backendContext: BackendContext, id: String, value: BlockMatrix, storageLevel: String): Unit = ???

  def unpersist(backendContext: BackendContext, id: String): Unit = ???

  def getPersistedBlockMatrix(backendContext: BackendContext, id: String): BlockMatrix = ???

  def getPersistedBlockMatrixType(backendContext: BackendContext, id: String): BlockMatrixType = ???

  def loadReferencesFromDataset(
    ctx: ExecuteContext,
    path: String
  ): String = ReferenceGenome.fromHailDataset(ctx.fs, path)

  def parseVCFMetadata(
    ctx: ExecuteContext,
    path: String
  ): String = {
    val metadata = LoadVCF.parseHeaderMetadata(ctx.fs, Set.empty, TFloat64, path)
    implicit val formats = defaultJSONFormats
    JsonMethods.compact(Extraction.decompose(metadata))
  }

  def importFam(
    ctx: ExecuteContext,
    path: String,
    quantPheno: Boolean,
    delimiter: String,
    missing: String
  ): String = {
    LoadPlink.importFamJSON(ctx.fs, path, quantPheno, delimiter, missing)
  }

  def indexBgen(
    ctx: ExecuteContext,
    files: Array[String],
    indexFileMap: Map[String, String],
    referenceGenomeName: Option[String],
    contigRecoding: Map[String, String],
    skipInvalidLoci: Boolean
  ): String = {
    IndexBgen(ctx, files, indexFileMap, referenceGenomeName, contigRecoding, skipInvalidLoci)
    info(s"Number of BGEN files indexed: ${ files.size }")
    "null"
  }
}

class EndOfInputException extends RuntimeException
class HailBatchFailure(message: String) extends RuntimeException(message)

object ServiceBackendSocketAPI2 {
  def main(argv: Array[String]): Unit = {
    assert(argv.length == 7, argv.toFastIndexedSeq)

    val scratchDir = argv(0)
    val logFile = argv(1)
    val jarLocation = argv(2)
    val kind = argv(3)
    assert(kind == Main.DRIVER)
    val name = argv(4)
    val input = argv(5)
    val output = argv(6)

    // FIXME: when can the classloader be shared? (optimizer benefits!)
    val backend = new ServiceBackend(
      jarLocation, name, new HailClassLoader(getClass().getClassLoader()), scratchDir)
    if (HailContext.isInitialized) {
      HailContext.get.backend = backend
    } else {
      HailContext(backend, "hail.log", false, false, 50, skipLoggingConfiguration = true, 3)
    }
    val fs = retryTransientErrors {
      using(new FileInputStream(s"$scratchDir/secrets/gsa-key/key.json")) { is =>
        new GoogleStorageFS(Some(IOUtils.toString(is, Charset.defaultCharset().toString()))).asCacheable()
      }
    }
    val deployConfig = DeployConfig.fromConfigFile(
      s"$scratchDir/secrets/deploy-config/deploy-config.json")
    DeployConfig.set(deployConfig)
    val userTokens = Tokens.fromFile(s"$scratchDir/secrets/user-tokens/tokens.json")
    Tokens.set(userTokens)
    tls.setSSLConfigFromDir(s"$scratchDir/secrets/ssl-config")

    val sessionId = userTokens.namespaceToken(deployConfig.defaultNamespace)
    retryTransientErrors {
      using(fs.openNoCompression(input)) { in =>
        retryTransientErrors {
          using(fs.createNoCompression(output)) { out =>
            new ServiceBackendSocketAPI2(backend, in, out, sessionId).executeOneCommand()
            out.flush()
          }
        }
      }
    }
  }
}

class ServiceBackendSocketAPI2(
  private[this] val backend: ServiceBackend,
  private[this] val in: InputStream,
  private[this] val out: OutputStream,
  private[this] val sessionId: String
) extends Thread {
  private[this] val LOAD_REFERENCES_FROM_DATASET = 1
  private[this] val VALUE_TYPE = 2
  private[this] val TABLE_TYPE = 3
  private[this] val MATRIX_TABLE_TYPE = 4
  private[this] val BLOCK_MATRIX_TYPE = 5
  private[this] val REFERENCE_GENOME = 6
  private[this] val EXECUTE = 7
  private[this] val PARSE_VCF_METADATA = 8
  private[this] val INDEX_BGEN = 9
  private[this] val IMPORT_FAM = 10

  private[this] val dummy = new Array[Byte](8)

  def read(bytes: Array[Byte], off: Int, n: Int): Unit = {
    assert(off + n <= bytes.length)
    var read = 0
    while (read < n) {
      val r = in.read(bytes, off + read, n - read)
      if (r < 0) {
        throw new EndOfInputException
      } else {
        read += r
      }
    }
  }

  def readBool(): Boolean = {
    read(dummy, 0, 1)
    Memory.loadByte(dummy, 0) != 0.toByte
  }

  def readInt(): Int = {
    read(dummy, 0, 4)
    Memory.loadInt(dummy, 0)
  }

  def readLong(): Long = {
    read(dummy, 0, 8)
    Memory.loadLong(dummy, 0)
  }

  def readBytes(): Array[Byte] = {
    val n = readInt()
    val bytes = new Array[Byte](n)
    read(bytes, 0, n)
    bytes
  }

  def readString(): String = new String(readBytes(), StandardCharsets.UTF_8)

  def writeBool(b: Boolean): Unit = {
    out.write(if (b) 1 else 0)
  }

  def writeInt(v: Int): Unit = {
    Memory.storeInt(dummy, 0, v)
    out.write(dummy, 0, 4)
  }

  def writeLong(v: Long): Unit = {
    Memory.storeLong(dummy, 0, v)
    out.write(dummy)
  }

  def writeBytes(bytes: Array[Byte]): Unit = {
    writeInt(bytes.length)
    out.write(bytes)
  }

  def writeString(s: String): Unit = writeBytes(s.getBytes(StandardCharsets.UTF_8))

  def executeOneCommand(): Unit = {
    var nFlagsRemaining = readInt()
    val flags = mutable.Map[String, String]()
    while (nFlagsRemaining > 0) {
      val flagName = readString()
      val flagValue = readString()
      flags.update(flagName, flagValue)
      nFlagsRemaining -= 1
    }

    val cmd = readInt()

    val tmpdir = readString()
    val billingProject = readString()
    val remoteTmpDir = readString()

    def withExecuteContext(methodName: String, method: ExecuteContext => Array[Byte]): Array[Byte] = ExecutionTimer.logTime(methodName) { timer =>
      val fs = retryTransientErrors {
        using(new FileInputStream(s"${backend.scratchDir}/secrets/gsa-key/key.json")) { is =>
          new GoogleStorageFS(Some(IOUtils.toString(is, Charset.defaultCharset().toString()))).asCacheable()
        }
      }
      ExecuteContext.scoped(
        tmpdir,
        "file:///tmp",
        backend,
        fs,
        timer,
        null,
        backend.theHailClassLoader,
        HailFeatureFlags.fromMap(flags)
      ) { ctx =>
        ctx.backendContext = new ServiceBackendContext(sessionId, billingProject, remoteTmpDir)
        method(ctx)
      }
    }

    try {
      val result = (cmd: @switch) match {
        case LOAD_REFERENCES_FROM_DATASET =>
          val path = readString()
          withExecuteContext(
            "ServiceBackend.loadReferencesFromDataset",
            backend.loadReferencesFromDataset(_, path).getBytes(StandardCharsets.UTF_8)
          )
        case VALUE_TYPE =>
          val s = readString()
          withExecuteContext(
            "ServiceBackend.valueType",
            backend.valueType(_, s).getBytes(StandardCharsets.UTF_8)
          )
        case TABLE_TYPE =>
          val s = readString()
          withExecuteContext(
            "ServiceBackend.tableType",
            backend.tableType(_, s).getBytes(StandardCharsets.UTF_8)
          )
        case MATRIX_TABLE_TYPE =>
          val s = readString()
          withExecuteContext(
            "ServiceBackend.matrixTableType",
            backend.matrixTableType(_, s).getBytes(StandardCharsets.UTF_8)
          )
        case BLOCK_MATRIX_TYPE =>
          val s = readString()
          withExecuteContext(
            "ServiceBackend.blockMatrixType",
            backend.blockMatrixType(_, s).getBytes(StandardCharsets.UTF_8)
          )
        case REFERENCE_GENOME =>
          val name = readString()
          withExecuteContext(
            "ServiceBackend.referenceGenome",
            backend.referenceGenome(_, name).getBytes(StandardCharsets.UTF_8)
          )
        case EXECUTE =>
          val code = readString()
          val token = readString()
          withExecuteContext(
            "ServiceBackend.execute",
            { ctx =>
              withIRFunctionsReadFromInput(ctx) { () =>
                val bufferSpecString = readString()
                backend.execute(ctx, code, token, bufferSpecString)
              }
            }
          )
        case PARSE_VCF_METADATA =>
          val path = readString()
          withExecuteContext(
            "ServiceBackend.parseVCFMetadata",
            backend.parseVCFMetadata(_, path).getBytes(StandardCharsets.UTF_8)
          )
        case IMPORT_FAM =>
          val path = readString()
          val quantPheno = readBool()
          val delimiter = readString()
          val missing = readString()
          withExecuteContext(
            "ServiceBackend.importFam",
            backend.importFam(_, path, quantPheno, delimiter, missing).getBytes(StandardCharsets.UTF_8)
          )
        case INDEX_BGEN =>
          val nFiles = readInt()
          val files = new Array[String](nFiles)
          var i = 0
          while (i < nFiles) {
            files(i) = readString()
            i += 1
          }
          val nIndexFiles = readInt()
          val indexFileMap = mutable.Map[String, String]()
          i = 0
          while (i < nIndexFiles) {
            val k = readString()
            val v = readString()
            indexFileMap(k) = v
            i += 1
          }
          val hasReferenceGenome = readBool()
          val referenceGenomeName = hasReferenceGenome match {
            case true => Some(readString())
            case false => None
          }
          val nContigRecoding = readInt()
          val contigRecoding = mutable.Map[String, String]()
          i = 0
          while (i < nContigRecoding) {
            val k = readString()
            val v = readString()
            contigRecoding(k) = v
            i += 1
          }
          val skipInvalidLoci = readBool()
          withExecuteContext(
            "ServiceBackend.indexBgen",
            backend.indexBgen(
              _,
              files,
              indexFileMap.toMap,
              referenceGenomeName,
              contigRecoding.toMap,
              skipInvalidLoci
            ).getBytes(StandardCharsets.UTF_8)
          )
      }
      writeBool(true)
      writeBytes(result)
    } catch {
      case exc: HailWorkerException =>
        writeBool(false)
        writeString(exc.shortMessage)
        writeString(exc.expandedMessage)
        writeInt(exc.errorId)
      case t: Throwable =>
        val (shortMessage, expandedMessage, errorId) = handleForPython(t)
        writeBool(false)
        writeString(shortMessage)
        writeString(expandedMessage)
        writeInt(errorId)
    }
  }

  def withIRFunctionsReadFromInput(ctx: ExecuteContext)(body: () => Array[Byte]): Array[Byte] = {
    try {
      var nFunctionsRemaining = readInt()
      while (nFunctionsRemaining > 0) {
        val name = readString()

        val nTypeParametersRemaining = readInt()
        val typeParameters = new Array[String](nTypeParametersRemaining)
        var i = 0
        while (i < nTypeParametersRemaining) {
          typeParameters(i) = readString()
          i += 1
        }

        val nValueParameterNamesRemaining = readInt()
        val valueParameterNames = new Array[String](nValueParameterNamesRemaining)
        i = 0
        while (i < nValueParameterNamesRemaining) {
          valueParameterNames(i) = readString()
          i += 1
        }

        val nValueParameterTypesRemaining = readInt()
        val valueParameterTypes = new Array[String](nValueParameterTypesRemaining)
        i = 0
        while (i < nValueParameterTypesRemaining) {
          valueParameterTypes(i) = readString()
          i += 1
        }

        val returnType = readString()

        val renderedBody = readString()

        IRFunctionRegistry.pyRegisterIRForServiceBackend(
          ctx,
          name,
          typeParameters,
          valueParameterNames,
          valueParameterTypes,
          returnType,
          renderedBody
        )
        nFunctionsRemaining -= 1
      }
      body()
    } finally {
      IRFunctionRegistry.clearUserFunctions()
    }
  }
}
