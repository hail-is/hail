package is.hail.backend.service

import is.hail.{CancellingExecutorService, HailContext, HailFeatureFlags}
import is.hail.annotations._
import is.hail.asm4s._
import is.hail.backend._
import is.hail.backend.service.ServiceBackend.MaxAvailableGcsConnections
import is.hail.expr.Validate
import is.hail.expr.ir.{
  IR, IRSize, LoweringAnalyses, MakeTuple, SortField, TableIR, TableReader, TypeCheck,
}
import is.hail.expr.ir.analyses.SemanticHash
import is.hail.expr.ir.compile.Compile
import is.hail.expr.ir.lowering._
import is.hail.io.fs._
import is.hail.io.reference.{IndexedFastaSequenceFile, LiftOver}
import is.hail.services._
import is.hail.services.JobGroupStates.Failure
import is.hail.types._
import is.hail.types.physical._
import is.hail.types.physical.stypes.PTypeReferenceSingleCodeType
import is.hail.types.virtual.{Kinds, TVoid}
import is.hail.utils._
import is.hail.utils.ExecutionTimer.Timings
import is.hail.variant.ReferenceGenome

import scala.collection.mutable
import scala.reflect.ClassTag

import java.io._
import java.nio.charset.StandardCharsets
import java.nio.file.Path
import java.util.concurrent._

import org.json4s.{DefaultFormats, Formats}
import org.json4s.JsonAST._
import org.json4s.jackson.JsonMethods
import sourcecode.Enclosing

case class ServiceBackendContext(
  remoteTmpDir: String,
  jobConfig: BatchJobConfig,
  override val executionCache: ExecutionCache,
) extends BackendContext with Serializable

object ServiceBackend {
  val MaxAvailableGcsConnections = 1000
}

class ServiceBackend(
  val name: String,
  batchClient: BatchClient,
  jarSpec: JarSpec,
  theHailClassLoader: HailClassLoader,
  val batchConfig: BatchConfig,
  rpcConfig: ServiceBackendRPCPayload,
  jobConfig: BatchJobConfig,
  flags: HailFeatureFlags,
  val fs: FS,
  references: mutable.Map[String, ReferenceGenome],
) extends Backend with Logging {

  private[this] var stageCount = 0
  private[this] val executor = Executors.newFixedThreadPool(MaxAvailableGcsConnections)

  override def shouldCacheQueryInfo: Boolean = false

  def defaultParallelism: Int = 4

  def broadcast[T: ClassTag](_value: T): BroadcastValue[T] = {
    using(new ObjectOutputStream(new ByteArrayOutputStream())) { os =>
      try
        os.writeObject(_value)
      catch {
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

  private[this] def submitJobGroupAndWait(
    collection: IndexedSeq[Array[Byte]],
    token: String,
    root: String,
    stageIdentifier: String,
  ): JobGroupResponse = {
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
            cpu = Some(jobConfig.worker_cores).filter(_ != "None"),
            memory = Some(jobConfig.worker_memory).filter(_ != "None"),
            storage = Some(jobConfig.storage).filter(_ != "0Gi"),
          )
        ),
        regions = Some(jobConfig.regions).filter(_.nonEmpty),
        cloudfuse = Some(jobConfig.cloudfuse_configs).filter(_.nonEmpty),
      )

    val jobs =
      collection.indices.map { i =>
        defaultJob.copy(
          attributes = Map("name" -> s"${name}_stage${stageCount}_${stageIdentifier}_job$i"),
          process = defaultProcess.copy(
            command = Array(Main.WORKER, root, s"$i", s"${collection.length}")
          ),
        )
      }

    val jobGroupId =
      batchClient.newJobGroup(
        JobGroupRequest(
          batch_id = batchConfig.batchId,
          absolute_parent_id = batchConfig.jobGroupId,
          token = token,
          attributes = Map("name" -> stageIdentifier),
          jobs = jobs,
        )
      )

    stageCount += 1

    Thread.sleep(600) // it is not possible for the batch to be finished in less than 600ms
    batchClient.waitForJobGroup(batchConfig.batchId, jobGroupId)
  }

  private[this] def readResult(root: String, i: Int): Array[Byte] = {
    val bytes = fs.readNoCompression(s"$root/result.$i")
    if (bytes(0) != 0) {
      bytes.slice(1, bytes.length)
    } else {
      val errorInformationBytes = bytes.slice(1, bytes.length)
      val is = new DataInputStream(new ByteArrayInputStream(errorInformationBytes))
      val shortMessage = readString(is)
      val expandedMessage = readString(is)
      val errorId = is.readInt()
      throw new HailWorkerException(i, shortMessage, expandedMessage, errorId)
    }
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

    val jobGroup = submitJobGroupAndWait(parts, token, root, stageIdentifier)

    log.info(s"parallelizeAndComputeWithIndex: $token: reading results")
    val startTime = System.nanoTime()
    val r @ (error, results) = runAllKeepFirstError(new CancellingExecutorService(executor)) {
      (partIdxs, parts.indices).zipped.map { (partIdx, jobIndex) =>
        (() => readResult(root, jobIndex), partIdx)
      }
    }

    error.foreach(throw _)

    if (jobGroup.state == Failure) {
      throw new HailBatchFailure(
        s"Job group ${jobGroup.job_group_id} for batch ${batchConfig.batchId} failed with an unknown error"
      )
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

  override def withExecuteContext[T](f: ExecuteContext => T)(implicit E: Enclosing): (T, Timings) =
    ExecutionTimer.time { timer =>
      ExecuteContext.scoped(
        rpcConfig.tmp_dir,
        rpcConfig.remote_tmpdir,
        this,
        fs,
        timer,
        null,
        theHailClassLoader,
        flags,
        ServiceBackendContext(
          rpcConfig.remote_tmpdir,
          jobConfig,
          ExecutionCache.fromFlags(flags, fs, rpcConfig.remote_tmpdir),
        ),
        new IrMetadata(),
        references,
        ImmutableMap.empty,
        mutable.Map.empty,
        ImmutableMap.empty,
      )(f)
    }
}

class EndOfInputException extends RuntimeException
class HailBatchFailure(message: String) extends RuntimeException(message)

case class Request(
  backend: ServiceBackend,
  fs: FS,
  outputUrl: String,
  action: Int,
  payload: JValue,
)

object ServiceBackendAPI extends HttpLikeBackendRpc[Request] with Logging {

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

    implicit val fmts: Formats = DefaultFormats

    val deployConfig = DeployConfig.fromConfigFile("/deploy-config/deploy-config.json")
    DeployConfig.set(deployConfig)
    sys.env.get("HAIL_SSL_CONFIG_DIR").foreach(tls.setSSLConfigFromDir)

    var fs = RouterFS.buildRoutes(
      CloudStorageFSConfig.fromFlagsAndEnv(
        Some(Path.of(scratchDir, "secrets/gsa-key/key.json")),
        HailFeatureFlags.fromEnv(),
      )
    )

    val (rpcConfig, jobConfig, action, payload) =
      using(fs.openNoCompression(inputURL)) { is =>
        val input = JsonMethods.parse(is)
        (
          (input \ "config").extract[ServiceBackendRPCPayload],
          (input \ "job_config").extract[BatchJobConfig],
          (input \ "action").extract[Int],
          input \ "payload",
        )
      }

    // requester pays config is conveyed in feature flags currently
    val featureFlags = HailFeatureFlags.fromEnv(rpcConfig.flags)
    fs = RouterFS.buildRoutes(
      CloudStorageFSConfig.fromFlagsAndEnv(
        Some(Path.of(scratchDir, "secrets/gsa-key/key.json")),
        featureFlags,
      )
    )

    val references = mutable.Map[String, ReferenceGenome]()
    references ++= ReferenceGenome.builtinReferences()
    rpcConfig.custom_references.toFastSeq.view.map(ReferenceGenome.fromJSON).foreach { rg =>
      references += rg.name -> rg
    }

    rpcConfig.liftovers.foreach { case (sourceGenome, liftoversForSource) =>
      liftoversForSource.foreach { case (destGenome, chainFile) =>
        references(sourceGenome).addLiftover(references(destGenome), LiftOver(fs, chainFile))
      }
    }

    rpcConfig.sequences.foreach { case (rg, seq) =>
      references(rg).addSequence(IndexedFastaSequenceFile(fs, seq.fasta, seq.index))
    }

    // FIXME: when can the classloader be shared? (optimizer benefits!)
    val backend = new ServiceBackend(
      name,
      BatchClient(deployConfig, Path.of(scratchDir, "secrets/gsa-key/key.json")),
      JarUrl(jarLocation),
      new HailClassLoader(getClass.getClassLoader),
      BatchConfig.fromConfigFile(Path.of(scratchDir, "batch-config/batch-config.json")),
      rpcConfig,
      jobConfig,
      featureFlags,
      fs,
      references,
    )

    log.info("ServiceBackend allocated.")
    if (HailContext.isInitialized) {
      HailContext.get.backend = backend
      log.info("Default references added to already initialized HailContexet.")
    } else {
      HailContext(backend, 50, 3)
      log.info("HailContexet initialized.")
    }

    runRpc(Request(backend, fs, outputURL, action, payload))
  }

  implicit override protected object Ask extends Routing {
    import Routes._

    override def route(a: Request): Route =
      a.action match {
        case 2 => TypeOf(Kinds.Value)
        case 3 => TypeOf(Kinds.Table)
        case 4 => TypeOf(Kinds.Matrix)
        case 5 => TypeOf(Kinds.BlockMatrix)
        case 6 => Execute
        case 7 => ParseVcfMetadata
        case 8 => ImportFam
        case 1 => LoadReferencesFromDataset
        case 9 => LoadReferencesFromFASTA
      }

    override def payload(a: Request): JValue = a.payload
  }

  implicit override protected object Write extends Write[Request] {

    // service backend doesn't support sending timings back to the python client
    override def timings(env: Request)(t: Timings): Unit =
      ()

    override def result(env: Request)(result: Array[Byte]): Unit =
      retryTransientErrors {
        using(env.fs.createNoCompression(env.outputUrl)) { outputStream =>
          val output = new HailSocketAPIOutputStream(outputStream)
          output.writeBool(true)
          output.writeBytes(result)
        }
      }

    override def error(env: Request)(t: Throwable): Unit =
      retryTransientErrors {
        val (shortMessage, expandedMessage, errorId) =
          t match {
            case t: HailWorkerException =>
              log.error(
                "A worker failed. The exception was written for Python but we will also throw an exception to fail this driver job.",
                t,
              )
              (t.shortMessage, t.expandedMessage, t.errorId)
            case _ =>
              log.error(
                "An exception occurred in the driver. The exception was written for Python but we will re-throw to fail this driver job.",
                t,
              )
              handleForPython(t)
          }

        using(env.fs.createNoCompression(env.outputUrl)) { outputStream =>
          val output = new HailSocketAPIOutputStream(outputStream)
          output.writeBool(false)
          output.writeString(shortMessage)
          output.writeString(expandedMessage)
          output.writeInt(errorId)
        }

        throw t
      }
  }

  implicit override protected object Context extends Context[Request] {
    override def scoped[A](env: Request)(f: ExecuteContext => A): (A, Timings) =
      env.backend.withExecuteContext(f)
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
  flags: Map[String, String],
  custom_references: Array[String],
  liftovers: Map[String, Map[String, String]],
  sequences: Map[String, SequenceConfig],
)

case class BatchJobConfig(
  billing_project: String,
  worker_cores: String,
  worker_memory: String,
  storage: String,
  cloudfuse_configs: Array[CloudfuseConfig],
  regions: Array[String],
)
