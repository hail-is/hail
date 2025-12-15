package is.hail.backend.service

import is.hail.Revision
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
import is.hail.services._
import is.hail.services.JobGroupStates.{Cancelled, Failure, Success}
import is.hail.services.oauth2.{CloudCredentials, HailCredentials}
import is.hail.types._
import is.hail.types.physical._
import is.hail.utils._
import is.hail.utils.compat.immutable.ArraySeq

import scala.collection.compat._
import scala.concurrent.{Await, CancellationException, ExecutionContext, Future}
import scala.concurrent.duration.Duration
import scala.reflect.ClassTag
import scala.util.control.NonFatal

import java.io._
import java.util.concurrent.Executors

import com.fasterxml.jackson.core.StreamReadConstraints

object ServiceBackend {
  val MaxAvailableGcsConnections = 1000

  // See https://github.com/hail-is/hail/issues/14580
  StreamReadConstraints.overrideDefaultStreamReadConstraints(
    StreamReadConstraints.builder().maxStringLength(Integer.MAX_VALUE).build()
  )

  def pyServiceBackend(
    name: String,
    batchId_ : Integer,
    billingProject: String,
    deployConfigFile: String,
    workerCores: String,
    workerMemory: String,
    storage: String,
    cloudfuse: Array[CloudfuseConfig],
    regions: Array[String],
  ): ServiceBackend = {
    val credentials: CloudCredentials =
      HailCredentials().getOrElse(CloudCredentials(keyPath = None))

    val client =
      BatchClient(
        DeployConfig.fromConfigFile(deployConfigFile),
        credentials,
      )

    val batchId =
      Option(batchId_).map(_.toInt).getOrElse {
        client.newBatch(
          BatchRequest(
            billing_project = billingProject,
            token = tokenUrlSafe,
            n_jobs = 0,
            attributes = Map("name" -> name),
          )
        )
      }

    val workerConfig =
      BatchJobConfig(
        workerCores,
        workerMemory,
        storage,
        cloudfuse,
        regions,
      )

    new ServiceBackend(
      name,
      client,
      GitRevision(Revision),
      BatchConfig(batchId, 0),
      workerConfig,
    )
  }
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
) extends Backend with Logging {

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

      private[this] def submitJobGroupAndWait(
        partitions: IndexedSeq[Int],
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
          partitions.zipWithIndex.map { case (partitionId, idx) =>
            defaultJob.copy(
              attributes = Map(
                "name" -> s"${name}_stage${stageCount}_${stageIdentifier}_partition$partitionId",
                "partition" -> partitionId.toString,
                "outfile" -> s"$root/result.$idx",
              ),
              process = defaultProcess.copy(
                command = Array(Main.WORKER, root, s"$partitionId", s"$idx")
              ),
            )
          }

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
        try {
          Thread.sleep(600) // it is not possible for the batch to be finished in less than 600ms
          val response = batchClient.waitForJobGroup(batchConfig.batchId, jobGroupId)
          (response, startJobId)
        } catch {
          case _: InterruptedException =>
            batchClient.cancelJobGroup(batchConfig.batchId, jobGroupId)
            Thread.currentThread().interrupt()
            throw new CancellationException()
        }
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
              case NonFatal(t) => (Some(t), ArraySeq.empty)
            } finally stageCount += 1

          case todo =>
            val token = tokenUrlSafe
            val root = s"${ctx.tmpdir}/mapCollectPartitions/$token"
            logger.info(s"mapCollectPartitions: token='$token', nPartitions=${todo.length}")

            implicit val ec: ExecutionContext =
              ExecutionContext.fromExecutor(executor)

            val uploadGlobals = Future {
              retryTransientErrors {
                ctx.fs.writePDOS(s"$root/globals")(_.write(globals))
                logger.info(s"mapCollectPartitions: $token: uploaded globals")
              }
            }

            val uploadContexts = Future {
              val partInputs = todo.map(contexts)
              retryTransientErrors {
                ctx.fs.writePDOS(s"$root/contexts") { os =>
                  var o = 12L * partInputs.length // 12L = sizeof(Long) + sizeof(Int)

                  for (p <- partInputs) {
                    val len = p.length
                    os.writeLong(o)
                    os.writeInt(len)
                    o += len
                  }

                  for (p <- partInputs) os.write(p)

                  logger.info(s"mapCollectPartitions: $token: wrote ${partInputs.length} contexts")
                }
              }
            }

            val uploadPartFn = Future {
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
                  logger.info(s"mapCollectPartitions: $token: uploaded function")
                }
              }
            }

            Await.result(uploadGlobals zip uploadContexts zip uploadPartFn, Duration.Inf): Unit

            val (jobGroup, startJobId) = submitJobGroupAndWait(todo, token, root, stageIdentifier)
            logger.info(s"mapCollectPartitions: $token: reading results")
            val startTime = System.nanoTime()

            def readPartitionOutputs(indices: IndexedSeq[Int]) =
              Future.traverse(indices) { idx =>
                Future {
                  val filename = s"$root/result.$idx"
                  try using(ctx.fs.openNoCompression(filename))(WireProtocol.read)
                  catch {
                    case NonFatal(e) =>
                      throw new HailException(
                        msg = f"Failed to read partition output '$filename'." +
                          f"See the batch ui at ${batchClient.req.url}/batches/${jobGroup.batch_id} for details.",
                        logMsg = None,
                        cause = e,
                      )
                  }
                }
              }

            val (failureOpt, results) =
              jobGroup.state match {
                case Success =>
                  val (failures, successes) =
                    Await.result(readPartitionOutputs(todo.indices), Duration.Inf).partitionMap(
                      identity
                    )

                  if (failures.nonEmpty)
                    logger.error(
                      f"Job group ${jobGroup.job_group_id} in batch ${jobGroup.batch_id} " +
                        f"completed successfully yet found errors in partition outputs."
                    )

                  (failures.headOption, successes)

                case Failure =>
                  val failedJobs =
                    batchClient.getJobGroupJobs(
                      jobGroup.batch_id,
                      jobGroup.job_group_id,
                      Some(JobStates.Failed),
                    )

                  val succeededJobs =
                    batchClient.getJobGroupJobs(
                      jobGroup.batch_id,
                      jobGroup.job_group_id,
                      Some(JobStates.Success),
                    )

                  val (failures, successes) =
                    Await.result(
                      Future
                        .traverse(failedJobs.map(_.take(1)) lazyAppendedAll succeededJobs) { jobs =>
                          readPartitionOutputs(jobs.map(_.job_id - startJobId))
                        },
                      Duration.Inf,
                    )
                      .flatten
                      .partitionMap(identity)

                  val error: Throwable =
                    failures.headOption.getOrElse {
                      new HailException(
                        f"An unknown error occurred. " +
                          f"Job group ${jobGroup.job_group_id} in batch ${jobGroup.batch_id} failed " +
                          f"yet found zero errors in partition outputs. " +
                          f"See the batch ui at ${batchClient.req.url}/batches/${jobGroup.batch_id} for details."
                      )
                    }

                  (Some(error), successes.to(ArraySeq))
                case Cancelled =>
                  val error =
                    new CancellationException(
                      s"Job group ${jobGroup.job_group_id} in batch ${batchConfig.batchId} was cancelled"
                    )

                  val succeededJobs =
                    batchClient.getJobGroupJobs(
                      jobGroup.batch_id,
                      jobGroup.job_group_id,
                      Some(JobStates.Success),
                    )

                  val (_, successes) =
                    Await.result(
                      Future.traverse(succeededJobs) { jobs =>
                        readPartitionOutputs(jobs.map(_.job_id - startJobId))
                      },
                      Duration.Inf,
                    )
                      .flatten
                      .partitionMap(identity)

                  (Some(error), successes.to(ArraySeq))
              }

            val end = (System.nanoTime() - startTime) / 1000000000.0
            val rate = results.length / end
            val byterate = results.view.map(_._1.length).sum / end / 1024 / 1024
            logger.info(s"all results read. $end s. $rate result/s. $byterate MiB/s.")
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
      logger.info(s"starting execution of query $queryID of initial size ${IRSize(ir)}")
      if (ctx.flags.isDefined(ExecutionCache.Flags.UseFastRestarts))
        ctx.irMetadata.semhash = SemanticHash(ctx, ir)
      val res = _jvmLowerAndExecute(ctx, ir)
      logger.info(s"finished execution of query $queryID")
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
