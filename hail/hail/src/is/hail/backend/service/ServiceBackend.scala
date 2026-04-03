package is.hail.backend.service

import is.hail.Revision
import is.hail.backend._
import is.hail.backend.Backend.PartitionFn
import is.hail.backend.ExecutionCache.Flags.UseFastRestarts
import is.hail.backend.local.LocalTaskContext
import is.hail.backend.service.ServiceBackend.Flags._
import is.hail.collection.FastSeq
import is.hail.collection.compat.immutable.ArraySeq
import is.hail.expr.Validate
import is.hail.expr.ir.{
  CompileAndEvaluate, IR, IRSize, LoweringAnalyses, NormalizeNames, SortField, TableIR, TableReader,
  TypeCheck,
}
import is.hail.expr.ir.analyses.SemanticHash
import is.hail.expr.ir.lowering._
import is.hail.services._
import is.hail.services.JobGroupStates.{Cancelled, Failure, Running, Success}
import is.hail.services.oauth2.{CloudCredentials, HailCredentials}
import is.hail.types._
import is.hail.types.physical._
import is.hail.utils._

import scala.concurrent._
import scala.concurrent.duration.Duration
import scala.math.Ordered.orderingToOrdered
import scala.reflect.ClassTag
import scala.util.control.NonFatal

import java.io._
import java.util.concurrent.{ExecutorCompletionService, Executors}

import com.fasterxml.jackson.core.StreamReadConstraints

object ServiceBackend {
  object Flags {
    val UseAsyncProfiler = "profile"
  }

  // Chosen (somewhat arbitrarily) to limit the amount of memory the driver
  // needs to execute range_table(10_000).repartition(10_000)._force_count()
  // so that it can run on "standard" workers without performance overhead.
  val DefaultMaxReadParallelism = 50

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
    maxReadParallelism: Integer,
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
        Option(workerCores),
        Option(workerMemory),
        Option(storage),
        Option(cloudfuse),
        Option(regions),
      )

    new ServiceBackend(
      name,
      client,
      GitRevision(Revision),
      BatchConfig(batchId, 0),
      workerConfig,
      Option(maxReadParallelism).map(_.toInt).getOrElse(DefaultMaxReadParallelism),
    )
  }
}

case class BatchJobConfig(
  worker_cores: Option[String],
  worker_memory: Option[String],
  storage: Option[String],
  cloudfuse_configs: Option[Array[CloudfuseConfig]],
  regions: Option[Array[String]],
)

class ServiceBackend(
  val name: String,
  batchClient: BatchClient,
  jarSpec: JarSpec,
  val batchConfig: BatchConfig,
  jobConfig: BatchJobConfig,
  maxReadParallelism: Int,
) extends Backend with Logging {

  private[this] val batchId = batchConfig.batchId
  private[this] var stageCount = 0

  private[this] val executor =
    lazily {
      Executors.newFixedThreadPool(maxReadParallelism)
    }

  override def defaultParallelism: Int = 4

  override def broadcast[T: ClassTag](_value: T): BroadcastValue[T] =
    new BroadcastValue[T] with Serializable {
      override def value: T = _value
    }

  override def runtimeContext(ctx: ExecuteContext): DriverRuntimeContext =
    new DriverRuntimeContext {

      override val executionCache: ExecutionCache =
        ExecutionCache.fromFlags(ctx.flags, ctx.fs, ctx.tmpdir)

      private[this] def submitJobGroup(
        partitions: IndexedSeq[Int],
        token: String,
        root: String,
        stageIdentifier: String,
      ): (Int, Int) = {
        val defaultProcess =
          JvmJob(
            command = null,
            spec = jarSpec,
            profile = ctx.flags.get(UseAsyncProfiler) != null,
          )

        val defaultJob =
          JobRequest(
            always_run = false,
            process = null,
            resources = Some(
              JobResources(
                preemptible = true,
                cpu = jobConfig.worker_cores,
                memory = jobConfig.worker_memory,
                storage = jobConfig.storage,
              )
            ),
            regions = jobConfig.regions,
            cloudfuse = jobConfig.cloudfuse_configs,
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

        stageCount += 1

        batchClient.newJobGroup(
          JobGroupRequest(
            batch_id = batchId,
            absolute_parent_id = batchConfig.jobGroupId,
            token = token,
            cancel_after_n_failures = Some(1),
            attributes = Map("name" -> stageIdentifier),
            jobs = jobs,
          )
        )
      }

      private[this] def readPartitionResult(root: String, partition: Int) = {
        val filename = s"$root/result.$partition"
        try using(ctx.fs.openNoCompression(filename))(WireProtocol.read)
        catch {
          case t: Throwable =>
            throw new HailException(
              msg = f"Failed to read partition output '$filename'." +
                f"See the batch ui at ${batchClient.req.url}/batches/$batchId for details.",
              logMsg = None,
              cause = t,
            )
        }
      }

      private[this] def collect(root: String, jobGroupId: Int, startJobId: Int, nJobs: Int)
        : (Option[Throwable], IndexedSeq[(Array[Byte], Int)]) = {

        // #15288 Write partition results directly to avoid copying unnecessarily.
        // We'll clean up nulls if the stage fails and the user has enabled call-caching.
        val results = new Array[(Array[Byte], Int)](nJobs)

        // While the job group is running, we'll poll the batch service for jobs
        // that have finished after this time. As we collect jobs, we'll update
        // this var to be that of the last job to finish.
        var endTime = Option.empty[String]

        // Delays in the cloud are inevitable and batch doesn't enforce end_time
        // ordering when jobs are marked complete. Consider the following timeline:
        //
        // 1. [query] request completed jobs from batch. let T = max(jobs.end_time).
        // 2. [batch] mark job(s) complete with an end_time < T due to some latency
        // 3. [query] request jobs that have completed after T
        //
        // In this case, we miss the jobs from 2. Recording the number of jobs
        // we've processed gives a cheap way to test if we've missed any.
        var read = 0

        def drain(jobs: Iterable[(Int, Option[String])]): Unit = {
          // Important: an ExecutorCompletionService maintains its own queue of
          // complete jobs. It's unsafe to reuse cs since `drain` throws before
          // emptying queue.
          val cs = new ExecutorCompletionService[Unit](executor)
          val iter = jobs.iterator

          def push(): Unit = {
            val (partId, time) = iter.next()
            if (time > endTime) endTime = time
            cs.submit { () =>
              readPartitionResult(root, partId) match {
                case Left(t) => throw t
                case Right(r) => results(r._2) = r
              }
            }
          }

          // The sliding window maintains min(MaxConcurrentPartitionReads, remainingJobs)
          // concurrent reads of partition results. It reduces peak memory consumption
          // by limiting the number of tasks that can added to the executor's queue.
          var inFlight = 0
          while (inFlight < maxReadParallelism && iter.hasNext) {
            push()
            inFlight += 1
          }

          while (inFlight > 0) {
            cs.take().get()
            read += 1
            if (iter.hasNext) push() else inFlight -= 1
          }
        }

        def nextSuccesses =
          batchClient
            .getJobGroupJobs(batchId, jobGroupId, Some(JobStates.Success), endTime)
            .flatMap(_.view.map(e => (e.job_id - startJobId, e.end_time)))

        def gather: IndexedSeq[(Array[Byte], Int)] =
          retryable { attempt =>
            batchClient.getJobGroup(batchId, jobGroupId).state match {
              case Running =>
                drain(nextSuccesses)
                logger.info(f"Read $read of $nJobs partition results")
                Thread.sleep(delayMsForTry(attempt, 1000, 10000))
                retry
              case Success =>
                drain(nextSuccesses)

                if (read < nJobs) {
                  logger.info(s"Reading ${nJobs - read} missed partition results")
                  val stragglers = results.indices.view.filter(results(_) == null)
                  drain(stragglers.map((_, endTime)))
                }

                assert(read == nJobs, f"Read $read of $nJobs partition results")
                ArraySeq.unsafeWrapArray(results)
              case Failure =>
                drain(
                  batchClient
                    .getJobGroupJobs(batchId, jobGroupId, Some(JobStates.Failed))
                    .flatMap(_.view.map(e => (e.job_id - startJobId, e.end_time)))
                    .take(3) // should only need one failure, but to be safe...
                ): Unit

                throw new HailException(
                  f"An unknown error occurred. " +
                    f"Job group $jobGroupId in batch $batchId failed " +
                    f"yet found zero errors in partition outputs. " +
                    f"See the batch ui at ${batchClient.req.url}/batches/$batchId for details."
                )
              case Cancelled =>
                val msg = s"Job group $jobGroupId in batch $batchId was cancelled."
                throw new CancellationException(msg)
            }
          }

        try ((Option.empty, gather)) // scala 2.12 needs these seemingly-redundant parens
        catch {
          case t: Throwable =>
            try
              batchClient.cancelJobGroup(batchId, jobGroupId)
            catch {
              case NonFatal(f) =>
                logger.warn(s"Failed to cancel job group $jobGroupId in batch $batchId", f)
            }

            val failure =
              t match {
                case _: InterruptedException =>
                  new CancellationException("Cancelled by user").fillInStackTrace()
                case e: ExecutionException =>
                  e.getCause
                case _ =>
                  t
              }

            val successes =
              if (!ctx.flags.isDefined(UseFastRestarts)) ArraySeq.empty
              else {
                try drain(nextSuccesses)
                catch {
                  case t: Throwable =>
                    logger.warn(s"Failed to collect final successful partition results", t)
                }

                val pruned = ArraySeq.newBuilder[(Array[Byte], Int)]
                pruned.sizeHint(read)
                for (r <- results) if (r != null) pruned += r
                pruned.result()
              }

            (Some(failure), successes)
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
                None -> htc.getRegionPool().scopedRegion { r =>
                  FastSeq(f(ctx.theHailClassLoader, ctx.fs, htc, r)(globals, contexts(k)) -> k)
                }
              }
            catch {
              case NonFatal(t) => Some(t) -> ArraySeq.empty
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
                logger.info("uploaded globals")
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

                  logger.info(s"wrote ${partInputs.length} contexts")
                }
              }
            }

            val uploadPartFn = Future {
              val fsConfig: Any =
                ctx.fs.getConfiguration()

              val partial: PartitionFn = { (hcl, fs, htc, r) =>
                fs.setConfiguration(fsConfig)
                f(hcl, fs, htc, r)
              }

              retryTransientErrors {
                ctx.fs.writePDOS(s"$root/f") { fos =>
                  using(new ObjectOutputStream(fos))(_.writeObject(partial))
                  logger.info("uploaded function")
                }
              }
            }

            Await.result(uploadGlobals zip uploadContexts zip uploadPartFn, Duration.Inf): Unit

            val (jobGroupId, startJobId) = submitJobGroup(todo, token, root, stageIdentifier)

            logger.info("reading results")
            val startTime = System.nanoTime()
            val results @ (_, bytes) = collect(root, jobGroupId, startJobId, todo.length)
            val end = (System.nanoTime() - startTime) / 1000000000.0
            val rate = bytes.length / end
            val byterate = bytes.view.map(_._1.length).sum / end / 1024 / 1024
            logger.info(s"all results read. $end s. $rate result/s. $byterate MiB/s.")
            results
        }
    }

  override def close(): Unit = {
    if (executor.isEvaluated) executor.shutdownNow()
    batchClient.close()
  }

  override def execute(ctx: ExecuteContext, ir0: IR): Either[Unit, (PTuple, Long)] =
    ctx.time {
      TypeCheck(ctx, ir0)
      Validate(ir0)
      val ir = NormalizeNames()(ctx, ir0)

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

  override def tableToTableStage(ctx: ExecuteContext, inputIR: TableIR, analyses: LoweringAnalyses)
    : TableStage =
    LowerTableIR.applyTable(inputIR, DArrayLowering.All, ctx, analyses)
}
