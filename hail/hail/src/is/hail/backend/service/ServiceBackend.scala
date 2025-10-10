package is.hail.backend.service

import is.hail.backend._
import is.hail.backend.Backend.PartitionFn
import is.hail.backend.local.LocalTaskContext
import is.hail.backend.service.ServiceBackend.MaxAvailableGcsConnections
import is.hail.expr.Validate
import is.hail.expr.ir.{
  CompileAndEvaluate, IR, IRSize, LoweringAnalyses, SortField, TableIR, TableReader, TypeCheck,
}
import is.hail.expr.ir.analyses.SemanticHash
import is.hail.expr.ir.lowering._
import is.hail.io.fs._
import is.hail.services._
import is.hail.services.JobGroupStates.{Cancelled, Failure, Success}
import is.hail.types._
import is.hail.types.physical._
import is.hail.utils._

import scala.collection.compat._
import scala.reflect.ClassTag
import scala.util.control.NonFatal

import java.io._
import java.nio.charset.StandardCharsets
import java.util.concurrent._

object ServiceBackend {
  val MaxAvailableGcsConnections = 1000
}

case class BatchJobConfig(
  worker_cores: String,
  worker_memory: String,
  storage: String,
  cloudfuse_configs: Array[CloudfuseConfig],
  regions: Array[String],
)

class ServiceBackend(
  val name: String,
  batchClient: BatchClient,
  jarSpec: JarSpec,
  val batchConfig: BatchConfig,
  jobConfig: BatchJobConfig,
) extends Backend {

  logHailVersion()

  private[this] var stageCount = 0

  private[this] val executor =
    lazily {
      Executors.newFixedThreadPool(MaxAvailableGcsConnections)
    }

  def defaultParallelism: Int = 4

  def broadcast[T: ClassTag](_value: T): BroadcastValue[T] =
    new BroadcastValue[T] with Serializable {
      def value: T = _value
    }

  override def runtimeContext(ctx: ExecuteContext): DriverRuntimeContext =
    new DriverRuntimeContext {

      override val executionCache: ExecutionCache =
        ExecutionCache.fromFlags(ctx.flags, ctx.fs, ctx.tmpdir)

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
      ): (JobGroupResponse, Int) = {
        val defaultProcess =
          JvmJob(
            command = null,
            spec = jarSpec,
            profile = ctx.flags.get("profile") != null,
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
         * This means that all JobId's in the JobGroup will have values in range (startJobId,
         * startJobId + n).
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

      private[this] def readPartitionResult(fs: FS, root: String, i: Int): Array[Byte] = {
        val file = s"$root/result.$i"
        val bytes = fs.readNoCompression(file)
        assert(bytes(0) != 0, s"$file is not a valid result.")
        bytes.slice(1, bytes.length)
      }

      private[this] def readPartitionError(fs: FS, root: String, i: Int): HailWorkerException = {
        val file = s"$root/result.$i"
        val bytes = fs.readNoCompression(file)
        assert(bytes(0) == 0, s"$file did not contain an error")
        val errorInformationBytes = bytes.slice(1, bytes.length)
        val is = new DataInputStream(new ByteArrayInputStream(errorInformationBytes))
        val shortMessage = readString(is)
        val expandedMessage = readString(is)
        val errorId = is.readInt()
        HailWorkerException(i, shortMessage, expandedMessage, errorId)
      }

      override def mapCollectPartitions(
        globals: Array[Byte],
        contexts: IndexedSeq[Array[Byte]],
        stageIdentifier: String,
        dependency: Option[TableStageDependency],
        partitions: Option[IndexedSeq[Int]],
      )(
        f: PartitionFn
      ): (Option[Throwable], IndexedSeq[(Array[Byte], Int)]) =
        partitions.getOrElse(contexts.indices) match {
          case Seq(k) =>
            try
              using(new LocalTaskContext(k, stageCount)) { htc =>
                (None, FastSeq(f(globals, contexts(k), htc, ctx.theHailClassLoader, ctx.fs) -> k))
              }
            catch {
              case NonFatal(t) => (Some(t), IndexedSeq.empty)
            } finally stageCount += 1

          case todo =>
            val token = tokenUrlSafe
            val root = s"${ctx.tmpdir}/mapCollectPartitions/$token"
            log.info(s"mapCollectPartitions: token='$token', nPartitions=${todo.length}")

            val uploadGlobals =
              executor.submit[Unit] { () =>
                retryTransientErrors {
                  ctx.fs.writePDOS(s"$root/globals")(_.write(globals))
                  log.info(s"mapCollectPartitions: $token: uploaded globals")
                }
              }

            val uploadFunction =
              executor.submit[Unit] { () =>
                val fsConfig: Any =
                  ctx.fs.getConfiguration()

                val partial: PartitionFn = {
                  (globals, context, htc, hcl, fs) =>
                    fs.setConfiguration(fsConfig)
                    f(globals, context, htc, hcl, fs)
                }

                retryTransientErrors {
                  ctx.fs.writePDOS(s"$root/f") { fos =>
                    using(new ObjectOutputStream(fos))(_.writeObject(partial))
                    log.info(s"mapCollectPartitions: $token: uploaded function")
                  }
                }
              }

            val parts = todo.map(contexts(_))

            val uploadContexts =
              executor.submit[Unit] { () =>
                retryTransientErrors {
                  ctx.fs.writePDOS(s"$root/contexts") { os =>
                    var o = 12L * parts.length // 12L = sizeof(Long) + sizeof(Int)

                    for (p <- parts) {
                      val len = p.length
                      os.writeLong(o)
                      os.writeInt(len)
                      o += len
                    }

                    for (p <- parts) os.write(p)

                    log.info(s"mapCollectPartitions: $token: wrote ${parts.length} contexts")
                  }
                }
              }

            uploadGlobals.get()
            uploadFunction.get()
            uploadContexts.get()

            val (jobGroup, startJobId) = submitJobGroupAndWait(parts, token, root, stageIdentifier)
            log.info(s"mapCollectPartitions: $token: reading results")
            val startTime = System.nanoTime()

            def streamSuccessfulPartitionResults: Stream[(Array[Byte], Int)] =
              for {
                successes <- batchClient.getJobGroupJobs(
                  jobGroup.batch_id,
                  jobGroup.job_group_id,
                  Some(JobStates.Success),
                )
                job <- successes
                partIdx = job.job_id - startJobId
              } yield (readPartitionResult(ctx.fs, root, partIdx), partIdx)

            val (failureOpt, results) =
              jobGroup.state match {
                case Success =>
                  runAllKeepFirstError(executor) {
                    todo.lazyZip(parts.indices).map { (partIdx, jobIndex) =>
                      (() => readPartitionResult(ctx.fs, root, jobIndex), partIdx)
                    }
                  }
                case Failure =>
                  val failedEntries =
                    batchClient.getJobGroupJobs(
                      jobGroup.batch_id,
                      jobGroup.job_group_id,
                      Some(JobStates.Failed),
                    )

                  assert(
                    failedEntries.nonEmpty,
                    s"Job group ${jobGroup.job_group_id} for batch ${batchConfig.batchId} failed, but no failed jobs found.",
                  )
                  val error =
                    readPartitionError(ctx.fs, root, failedEntries.head.head.job_id - startJobId)
                  (Some(error), streamSuccessfulPartitionResults.toIndexedSeq)
                case Cancelled =>
                  val error =
                    new CancellationException(
                      s"Job group ${jobGroup.job_group_id} for batch ${batchConfig.batchId} was cancelled"
                    )

                  (Some(error), streamSuccessfulPartitionResults.toIndexedSeq)
              }

            val end = (System.nanoTime() - startTime) / 1000000000.0
            val rate = results.length / end
            val byterate = results.view.map(_._1.length).sum / end / 1024 / 1024
            log.info(s"all results read. $end s. $rate result/s. $byterate MiB/s.")
            (failureOpt, results.sortBy(_._2))
        }
    }

  override def close(): Unit = {
    if (executor.isEvaluated) executor.shutdownNow()
    if (batchClient != null) batchClient.close() // see Worker
  }

  override def execute(ctx: ExecuteContext, ir: IR): Either[Unit, (PTuple, Long)] =
    ctx.time {
      TypeCheck(ctx, ir)
      Validate(ir)
      val queryID = Backend.nextID()
      log.info(s"starting execution of query $queryID of initial size ${IRSize(ir)}")
      if (ctx.flags.isDefined(ExecutionCache.Flags.UseFastRestarts))
        ctx.irMetadata.semhash = SemanticHash(ctx, ir)
      val res = _jvmLowerAndExecute(ctx, ir)
      log.info(s"finished execution of query $queryID")
      res
    }

  private[this] def _jvmLowerAndExecute(ctx: ExecuteContext, ir: IR): Either[Unit, (PTuple, Long)] =
    CompileAndEvaluate._apply(
      ctx,
      ir,
      lower = LoweringPipeline.darrayLowerer(DArrayLowering.All),
    )

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
}
