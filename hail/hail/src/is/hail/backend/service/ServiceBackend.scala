package is.hail.backend.service

import is.hail.{HailContext, HailFeatureFlags}
import is.hail.annotations._
import is.hail.asm4s._
import is.hail.backend._
import is.hail.expr.Validate
import is.hail.expr.ir.{
  IR, IRParser, IRSize, LoweringAnalyses, SortField, TableIR, TableReader, TypeCheck,
}
import is.hail.expr.ir.analyses.SemanticHash
import is.hail.expr.ir.compile.Compile
import is.hail.expr.ir.defs.MakeTuple
import is.hail.expr.ir.functions.IRFunctionRegistry
import is.hail.expr.ir.lowering._
import is.hail.io.fs._
import is.hail.io.reference.{IndexedFastaSequenceFile, LiftOver}
import is.hail.services.{BatchClient, JobGroupRequest, _}
import is.hail.services.JobGroupStates.{Cancelled, Failure, Success}
import is.hail.types._
import is.hail.types.physical._
import is.hail.types.physical.stypes.PTypeReferenceSingleCodeType
import is.hail.types.virtual._
import is.hail.utils._
import is.hail.variant.ReferenceGenome

import scala.annotation.switch
import scala.collection.mutable
import scala.reflect.ClassTag

import java.io._
import java.nio.charset.StandardCharsets
import java.nio.file.Path
import java.util.concurrent._

import org.apache.log4j.Logger
import org.json4s.{DefaultFormats, Formats}
import org.json4s.JsonAST._
import org.json4s.jackson.{JsonMethods, Serialization}
import sourcecode.Enclosing

class ServiceBackendContext(
  val billingProject: String,
  val remoteTmpDir: String,
  val workerCores: String,
  val workerMemory: String,
  val storageRequirement: String,
  val regions: Array[String],
  val cloudfuseConfig: Array[CloudfuseConfig],
  val profile: Boolean,
  val executionCache: ExecutionCache,
) extends BackendContext with Serializable {}

object ServiceBackend {

  def apply(
    jarLocation: String,
    name: String,
    theHailClassLoader: HailClassLoader,
    batchClient: BatchClient,
    batchConfig: BatchConfig,
    scratchDir: String = sys.env.getOrElse("HAIL_WORKER_SCRATCH_DIR", ""),
    rpcConfig: ServiceBackendRPCPayload,
    env: Map[String, String],
  ): ServiceBackend = {

    val flags = HailFeatureFlags.fromEnv(rpcConfig.flags)
    val shouldProfile = flags.get("profile") != null
    val fs = RouterFS.buildRoutes(
      CloudStorageFSConfig.fromFlagsAndEnv(
        Some(Path.of(scratchDir, "secrets/gsa-key/key.json")),
        flags,
        env,
      )
    )

    val backendContext = new ServiceBackendContext(
      rpcConfig.billing_project,
      rpcConfig.remote_tmpdir,
      rpcConfig.worker_cores,
      rpcConfig.worker_memory,
      rpcConfig.storage,
      rpcConfig.regions,
      rpcConfig.cloudfuse_configs,
      shouldProfile,
      ExecutionCache.fromFlags(flags, fs, rpcConfig.remote_tmpdir),
    )

    val references = mutable.Map.empty[String, ReferenceGenome]
    references ++= ReferenceGenome.builtinReferences()
    ReferenceGenome.addFatalOnCollision(
      references,
      rpcConfig.custom_references.map(ReferenceGenome.fromJSON),
    )

    rpcConfig.liftovers.foreach { case (sourceGenome, liftoversForSource) =>
      liftoversForSource.foreach { case (destGenome, chainFile) =>
        references(sourceGenome).addLiftover(references(destGenome), LiftOver(fs, chainFile))
      }
    }
    rpcConfig.sequences.foreach { case (rg, seq) =>
      references(rg).addSequence(IndexedFastaSequenceFile(fs, seq.fasta, seq.index))
    }

    new ServiceBackend(
      JarUrl(jarLocation),
      name,
      theHailClassLoader,
      references,
      batchClient,
      batchConfig,
      flags,
      rpcConfig.tmp_dir,
      fs,
      backendContext,
      scratchDir,
    )
  }
}

class ServiceBackend(
  val jarSpec: JarSpec,
  var name: String,
  val theHailClassLoader: HailClassLoader,
  val references: mutable.Map[String, ReferenceGenome],
  val batchClient: BatchClient,
  val batchConfig: BatchConfig,
  val flags: HailFeatureFlags,
  val tmpdir: String,
  val fs: FS,
  val serviceBackendContext: ServiceBackendContext,
  val scratchDir: String,
) extends Backend with Logging {

  private[this] var stageCount = 0
  private[this] val MAX_AVAILABLE_GCS_CONNECTIONS = 1000
  private[this] val executor = Executors.newFixedThreadPool(MAX_AVAILABLE_GCS_CONNECTIONS)

  def defaultParallelism: Int = 4

  def broadcast[T: ClassTag](_value: T): BroadcastValue[T] =
    new BroadcastValue[T] with Serializable {
      def value: T = _value
    }

  private[this] def readString(in: DataInputStream): String = {
    val n = in.readInt()
    val bytes = new Array[Byte](n)
    in.read(bytes)
    new String(bytes, StandardCharsets.UTF_8)
  }

  private[this] def submitJobGroupAndWait(
    backendContext: ServiceBackendContext,
    collection: IndexedSeq[Array[Byte]],
    token: String,
    root: String,
    stageIdentifier: String,
  ): (JobGroupResponse, Int) = {
    val defaultProcess =
      JvmJob(
        command = null,
        spec = jarSpec,
        profile = flags.get("profile") != null,
      )

    val defaultJob =
      JobRequest(
        always_run = false,
        process = null,
        resources = Some(
          JobResources(
            preemptible = true,
            cpu = Some(backendContext.workerCores).filter(_ != "None"),
            memory = Some(backendContext.workerMemory).filter(_ != "None"),
            storage = Some(backendContext.storageRequirement).filter(_ != "0Gi"),
          )
        ),
        regions = Some(backendContext.regions).filter(_.nonEmpty),
        cloudfuse = Some(backendContext.cloudfuseConfig).filter(_.nonEmpty),
      )

    val jobs =
      collection.indices.map { i =>
        defaultJob.copy(
          attributes = Map(
            "name" -> s"${name}_stage${stageCount}_${stageIdentifier}_job$i",
            "idx" -> i.toString,
          ),
          process = defaultProcess.copy(
            command = Array(Main.WORKER, root, s"$i", s"${collection.length}")
          ),
        )
      }

    /* When we create a JobGroup with n jobs, Batch gives us the absolute JobGroupId, and the
     * startJobId for the first job.
     * This means that all JobId's in the JobGroup will have values in range (startJobId, startJobId
     * + n).
     * Therefore, we know the partition index for a given job by using this startJobId offset.
     *
     * Why do we do this?
     * Consider a situation where we're submitting thousands of jobs in a job group.
     * If one of those jobs fails, we don't want to make thousands of requests to batch to get a
     * partition index that that job corresponds to. */

    val (jobGroupId, startJobId) =
      batchClient.newJobGroup(
        JobGroupRequest(
          batch_id = batchConfig.batchId,
          absolute_parent_id = batchConfig.jobGroupId,
          token = token,
          cancel_after_n_failures = Some(1),
          attributes = Map("name" -> stageIdentifier),
          jobs = jobs,
        )
      )

    stageCount += 1

    Thread.sleep(600) // it is not possible for the batch to be finished in less than 600ms
    val response = batchClient.waitForJobGroup(batchConfig.batchId, jobGroupId)
    (response, startJobId)
  }

  private[this] def readPartitionResult(root: String, i: Int): Array[Byte] = {
    val file = s"$root/result.$i"
    val bytes = fs.readNoCompression(file)
    assert(bytes(0) != 0, s"$file is not a valid result.")
    bytes.slice(1, bytes.length)
  }

  private[this] def readPartitionError(root: String, i: Int): HailWorkerException = {
    val file = s"$root/result.$i"
    val bytes = fs.readNoCompression(file)
    assert(bytes(0) == 0, s"$file did not contain an error")
    val errorInformationBytes = bytes.slice(1, bytes.length)
    val is = new DataInputStream(new ByteArrayInputStream(errorInformationBytes))
    val shortMessage = readString(is)
    val expandedMessage = readString(is)
    val errorId = is.readInt()
    new HailWorkerException(i, shortMessage, expandedMessage, errorId)
  }

  override def parallelizeAndComputeWithIndex(
    _backendContext: BackendContext,
    fs: FS,
    contexts: IndexedSeq[Array[Byte]],
    stageIdentifier: String,
    dependency: Option[TableStageDependency],
    partitions: Option[IndexedSeq[Int]],
  )(
    f: (Array[Byte], HailTaskContext, HailClassLoader, FS) => Array[Byte]
  ): (Option[Throwable], IndexedSeq[(Array[Byte], Int)]) = {

    val backendContext = _backendContext.asInstanceOf[ServiceBackendContext]

    val token = tokenUrlSafe
    val root = s"${backendContext.remoteTmpDir}/parallelizeAndComputeWithIndex/$token"
    log.info(s"parallelizeAndComputeWithIndex: token='$token', nPartitions=${contexts.length}")

    val uploadFunction = executor.submit[Unit](() =>
      retryTransientErrors {
        fs.writePDOS(s"$root/f") { fos =>
          using(new ObjectOutputStream(fos))(oos => oos.writeObject(f))
          log.info(s"parallelizeAndComputeWithIndex: $token: uploaded f")
        }
      }
    )

    val (partIdxs, parts) =
      partitions
        .map(ps => (ps, ps.map(contexts)))
        .getOrElse((contexts.indices, contexts))

    val uploadContexts = executor.submit[Unit](() =>
      retryTransientErrors {
        fs.writePDOS(s"$root/contexts") { os =>
          var o = 12L * parts.length // 12L = sizeof(Long) + sizeof(Int)
          parts.foreach { context =>
            val len = context.length
            os.writeLong(o)
            os.writeInt(len)
            o += len
          }
          parts.foreach(os.write)
          log.info(s"parallelizeAndComputeWithIndex: $token: wrote ${parts.length} contexts")
        }
      }
    )

    uploadFunction.get()
    uploadContexts.get()

    val (jobGroup, startJobId) =
      submitJobGroupAndWait(backendContext, parts, token, root, stageIdentifier)
    log.info(s"parallelizeAndComputeWithIndex: $token: reading results")
    val startTime = System.nanoTime()

    def streamSuccessfulJobResults: Stream[(Array[Byte], Int)] =
      for {
        successes <- batchClient.getJobGroupJobs(
          jobGroup.batch_id,
          jobGroup.job_group_id,
          Some(JobStates.Success),
        )
        job <- successes
        partIdx = job.job_id - startJobId
      } yield (readPartitionResult(root, partIdx), partIdx)

    val r @ (_, results) =
      jobGroup.state match {
        case Success =>
          runAllKeepFirstError(executor) {
            (partIdxs, parts.indices).zipped.map { (partIdx, jobIndex) =>
              (() => readPartitionResult(root, jobIndex), partIdx)
            }
          }
        case Failure =>
          val failedEntries = batchClient.getJobGroupJobs(
            jobGroup.batch_id,
            jobGroup.job_group_id,
            Some(JobStates.Failed),
          )
          assert(
            failedEntries.nonEmpty,
            s"Job group ${jobGroup.job_group_id} for batch ${batchConfig.batchId} failed, but no failed jobs found.",
          )
          val error = readPartitionError(root, failedEntries.head.head.job_id - startJobId)

          (Some(error), streamSuccessfulJobResults.toIndexedSeq)
        case Cancelled =>
          val error =
            new CancellationException(
              s"Job group ${jobGroup.job_group_id} for batch ${batchConfig.batchId} was cancelled"
            )

          (Some(error), streamSuccessfulJobResults.toIndexedSeq)
      }

    val resultsReadingSeconds = (System.nanoTime() - startTime) / 1000000000.0
    val rate = results.length / resultsReadingSeconds
    val byterate = results.map(_._1.length).sum / resultsReadingSeconds / 1024 / 1024
    log.info(s"all results read. $resultsReadingSeconds s. $rate result/s. $byterate MiB/s.")
    r
  }

  override def close(): Unit = {
    executor.shutdownNow()
    batchClient.close()
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

  private[this] def _jvmLowerAndExecute(ctx: ExecuteContext, ir: IR): Either[Unit, (PTuple, Long)] =
    ctx.time {
      val x = LoweringPipeline.darrayLowerer(true)(DArrayLowering.All)(ctx, ir).asInstanceOf[IR]

      x.typ match {
        case TVoid =>
          val (_, f) = Compile[AsmFunction1RegionUnit](
            ctx,
            FastSeq(),
            FastSeq[TypeInfo[_]](classInfo[Region]),
            UnitInfo,
            x,
            optimize = true,
          )

          Left(ctx.scopedExecution((hcl, fs, htc, r) => f(hcl, fs, htc, r)(r)))
        case _ =>
          val (Some(PTypeReferenceSingleCodeType(pt: PTuple)), f) =
            Compile[AsmFunction1RegionLong](
              ctx,
              FastSeq(),
              FastSeq(classInfo[Region]),
              LongInfo,
              MakeTuple.ordered(FastSeq(x)),
              optimize = true,
            )

          Right((pt, ctx.scopedExecution((hcl, fs, htc, r) => f(hcl, fs, htc, r)(r))))
      }
    }

  override def lowerDistributedSort(
    ctx: ExecuteContext,
    inputStage: TableStage,
    sortFields: IndexedSeq[SortField],
    rt: RTable,
    nPartitions: Option[Int],
  ): TableReader =
    LowerDistributedSort.distributedSort(ctx, inputStage, sortFields, rt, nPartitions)

  def tableToTableStage(ctx: ExecuteContext, inputIR: TableIR, analyses: LoweringAnalyses)
    : TableStage =
    LowerTableIR.applyTable(inputIR, DArrayLowering.All, ctx, analyses)

  override def withExecuteContext[T](f: ExecuteContext => T)(implicit E: Enclosing): T =
    ExecutionTimer.logTime { timer =>
      ExecuteContext.scoped(
        tmpdir,
        "file:///tmp",
        this,
        references.toMap,
        fs,
        timer,
        null,
        theHailClassLoader,
        flags,
        serviceBackendContext,
        new IrMetadata(),
        ImmutableMap.empty,
        ImmutableMap.empty,
        ImmutableMap.empty,
        ImmutableMap.empty,
      )(f)
    }

  override def loadReferencesFromDataset(path: String): Array[Byte] =
    withExecuteContext { ctx =>
      val rgs = ReferenceGenome.fromHailDataset(ctx.fs, path)
      ReferenceGenome.addFatalOnCollision(references, rgs)
      implicit val formats: Formats = defaultJSONFormats
      Serialization.write(rgs.map(_.toJSON).toFastSeq).getBytes(StandardCharsets.UTF_8)
    }
}

class EndOfInputException extends RuntimeException
class HailBatchFailure(message: String) extends RuntimeException(message)

object ServiceBackendAPI {
  private[this] val log = Logger.getLogger(getClass.getName())

  def main(argv: Array[String]): Unit = {
    assert(argv.length == 7, argv.toFastSeq)

    val scratchDir = argv(0)
    // val logFile = argv(1)
    val jarLocation = argv(2)
    val kind = argv(3)
    assert(kind == Main.DRIVER)
    val name = argv(4)
    val inputURL = argv(5)
    val outputURL = argv(6)

    implicit val formats: Formats = DefaultFormats

    val fs = RouterFS.buildRoutes(
      CloudStorageFSConfig.fromFlagsAndEnv(
        Some(Path.of(scratchDir, "secrets/gsa-key/key.json")),
        HailFeatureFlags.fromEnv(),
      )
    )
    val deployConfig = DeployConfig.fromConfigFile("/deploy-config/deploy-config.json")
    DeployConfig.set(deployConfig)
    sys.env.get("HAIL_SSL_CONFIG_DIR").foreach(tls.setSSLConfigFromDir)

    val batchClient = BatchClient(deployConfig, Path.of(scratchDir, "secrets/gsa-key/key.json"))
    log.info("BatchClient allocated.")

    val batchConfig =
      BatchConfig.fromConfigFile(Path.of(scratchDir, "batch-config/batch-config.json"))
    log.info("BatchConfig parsed.")

    val input = using(fs.openNoCompression(inputURL))(JsonMethods.parse(_))
    val rpcConfig = (input \ "config").extract[ServiceBackendRPCPayload]

    // FIXME: when can the classloader be shared? (optimizer benefits!)
    val backend = ServiceBackend(
      jarLocation,
      name,
      new HailClassLoader(getClass().getClassLoader()),
      batchClient,
      batchConfig,
      scratchDir,
      rpcConfig,
      sys.env,
    )
    log.info("ServiceBackend allocated.")
    if (HailContext.isInitialized) {
      HailContext.get.backend = backend
      log.info("Default references added to already initialized HailContexet.")
    } else {
      HailContext(backend, 50, 3)
      log.info("HailContexet initialized.")
    }

    val action = (input \ "action").extract[Int]
    val payload = (input \ "payload")
    new ServiceBackendAPI(backend, fs, outputURL).executeOneCommand(action, payload)
  }
}

private class HailSocketAPIOutputStream(
  private[this] val out: OutputStream
) extends AutoCloseable {
  private[this] var closed: Boolean = false
  private[this] val dummy = new Array[Byte](8)

  def writeBool(b: Boolean): Unit =
    out.write(if (b) 1 else 0)

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

  def close(): Unit =
    if (!closed) {
      out.close()
      closed = true
    }
}

case class SequenceConfig(fasta: String, index: String)

case class ServiceBackendRPCPayload(
  tmp_dir: String,
  remote_tmpdir: String,
  billing_project: String,
  worker_cores: String,
  worker_memory: String,
  storage: String,
  cloudfuse_configs: Array[CloudfuseConfig],
  regions: Array[String],
  flags: Map[String, String],
  custom_references: Array[String],
  liftovers: Map[String, Map[String, String]],
  sequences: Map[String, SequenceConfig],
)

case class ServiceBackendExecutePayload(
  functions: Array[SerializedIRFunction],
  idempotency_token: String,
  payload: ExecutePayload,
)

case class SerializedIRFunction(
  name: String,
  type_parameters: Array[String],
  value_parameter_names: Array[String],
  value_parameter_types: Array[String],
  return_type: String,
  rendered_body: String,
)

class ServiceBackendAPI(
  private[this] val backend: ServiceBackend,
  private[this] val fs: FS,
  private[this] val outputURL: String,
) extends Thread {
  private[this] val LOAD_REFERENCES_FROM_DATASET = 1
  private[this] val VALUE_TYPE = 2
  private[this] val TABLE_TYPE = 3
  private[this] val MATRIX_TABLE_TYPE = 4
  private[this] val BLOCK_MATRIX_TYPE = 5
  private[this] val EXECUTE = 6
  private[this] val PARSE_VCF_METADATA = 7
  private[this] val IMPORT_FAM = 8
  private[this] val FROM_FASTA_FILE = 9

  private[this] val log = Logger.getLogger(getClass.getName())

  private[this] def doAction(action: Int, payload: JValue): Array[Byte] = retryTransientErrors {
    implicit val formats: Formats = DefaultFormats
    (action: @switch) match {
      case LOAD_REFERENCES_FROM_DATASET =>
        val path = payload.extract[LoadReferencesFromDatasetPayload].path
        backend.loadReferencesFromDataset(path)
      case VALUE_TYPE =>
        val ir = payload.extract[IRTypePayload].ir
        backend.valueType(ir)
      case TABLE_TYPE =>
        val ir = payload.extract[IRTypePayload].ir
        backend.tableType(ir)
      case MATRIX_TABLE_TYPE =>
        val ir = payload.extract[IRTypePayload].ir
        backend.matrixTableType(ir)
      case BLOCK_MATRIX_TYPE =>
        val ir = payload.extract[IRTypePayload].ir
        backend.blockMatrixType(ir)
      case EXECUTE =>
        val qobExecutePayload = payload.extract[ServiceBackendExecutePayload]
        val bufferSpecString = qobExecutePayload.payload.stream_codec
        val code = qobExecutePayload.payload.ir
        backend.withExecuteContext { ctx =>
          withIRFunctionsReadFromInput(qobExecutePayload.functions, ctx) { () =>
            val ir = IRParser.parse_value_ir(ctx, code)
            backend.execute(ctx, ir) match {
              case Left(()) =>
                Array()
              case Right((pt, off)) =>
                using(new ByteArrayOutputStream()) { os =>
                  Backend.encodeToOutputStream(ctx, pt, off, bufferSpecString, os)
                  os.toByteArray
                }
            }
          }
        }
      case PARSE_VCF_METADATA =>
        val path = payload.extract[ParseVCFMetadataPayload].path
        backend.parseVCFMetadata(path)
      case IMPORT_FAM =>
        val famPayload = payload.extract[ImportFamPayload]
        val path = famPayload.path
        val quantPheno = famPayload.quant_pheno
        val delimiter = famPayload.delimiter
        val missing = famPayload.missing
        backend.importFam(path, quantPheno, delimiter, missing)
      case FROM_FASTA_FILE =>
        val fastaPayload = payload.extract[FromFASTAFilePayload]
        backend.fromFASTAFile(
          fastaPayload.name,
          fastaPayload.fasta_file,
          fastaPayload.index_file,
          fastaPayload.x_contigs,
          fastaPayload.y_contigs,
          fastaPayload.mt_contigs,
          fastaPayload.par,
        )
    }
  }

  private[this] def withIRFunctionsReadFromInput(
    serializedFunctions: Array[SerializedIRFunction],
    ctx: ExecuteContext,
  )(
    body: () => Array[Byte]
  ): Array[Byte] = {
    try {
      serializedFunctions.foreach { func =>
        IRFunctionRegistry.registerIR(
          ctx,
          func.name,
          func.type_parameters,
          func.value_parameter_names,
          func.value_parameter_types,
          func.return_type,
          func.rendered_body,
        )
      }
      body()
    } finally
      IRFunctionRegistry.clearUserFunctions()
  }

  def executeOneCommand(action: Int, payload: JValue): Unit = {
    try {
      val result = doAction(action, payload)
      retryTransientErrors {
        using(fs.createNoCompression(outputURL)) { outputStream =>
          val output = new HailSocketAPIOutputStream(outputStream)
          output.writeBool(true)
          output.writeBytes(result)
        }
      }
    } catch {
      case exc: HailWorkerException =>
        retryTransientErrors {
          using(fs.createNoCompression(outputURL)) { outputStream =>
            val output = new HailSocketAPIOutputStream(outputStream)
            output.writeBool(false)
            output.writeString(exc.shortMessage)
            output.writeString(exc.expandedMessage)
            output.writeInt(exc.errorId)
          }
        }
        log.error(
          "A worker failed. The exception was written for Python but we will also throw an exception to fail this driver job."
        )
        throw exc
      case t: Throwable =>
        val (shortMessage, expandedMessage, errorId) = handleForPython(t)
        retryTransientErrors {
          using(fs.createNoCompression(outputURL)) { outputStream =>
            val output = new HailSocketAPIOutputStream(outputStream)
            output.writeBool(false)
            output.writeString(shortMessage)
            output.writeString(expandedMessage)
            output.writeInt(errorId)
          }
        }
        log.error(
          "An exception occurred in the driver. The exception was written for Python but we will re-throw to fail this driver job."
        )
        throw t
    }
  }
}
