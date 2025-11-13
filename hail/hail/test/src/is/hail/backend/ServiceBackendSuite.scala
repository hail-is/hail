package is.hail.backend

import is.hail.HailSuite
import is.hail.backend.service.{BatchJobConfig, ServiceBackend, WireProtocol}
import is.hail.services._
import is.hail.services.JobGroupStates.{Cancelled, Failure, Success}
import is.hail.utils.{handleForPython, tokenUrlSafe, using, FastSeq, HailWorkerException}

import scala.collection.compat.immutable.LazyList
import scala.concurrent.CancellationException
import scala.reflect.io.{Directory, Path}
import scala.util.Random

import java.io.Closeable
import java.nio.charset.StandardCharsets

import org.mockito.ArgumentMatchersSugar.any
import org.mockito.IdiomaticMockito
import org.mockito.MockitoSugar.when
import org.scalatest.OptionValues
import org.scalatest.matchers.should.Matchers.{a, convertToAnyShouldWrapper}
import org.testng.annotations.Test

class ServiceBackendSuite extends HailSuite with IdiomaticMockito with OptionValues {

  @Test def testExecutesSinglePartitionLocally(): Unit =
    runMock { (ctx, _, batchClient, backend) =>
      val contexts = Array.tabulate(10)(_.toString.getBytes)

      val (failure, _) =
        backend.runtimeContext(ctx).mapCollectPartitions(
          Array.emptyByteArray,
          contexts,
          "stage1",
          partitions = Some(FastSeq(0)),
        )((_, bytes, _, _, _) => bytes)

      failure.foreach(throw _)

      batchClient.newJobGroup(any[JobGroupRequest]) wasNever called
    }

  @Test def testCreateJobPayload(): Unit =
    runMock { (ctx, jobConfig, batchClient, backend) =>
      val contexts = Array.tabulate(2)(_.toString.getBytes)

      // verify that
      // - the number of jobs matches the number of partitions, and
      // - each job is created in the specified region, and
      // - each job's resource configuration matches the rpc config

      when(batchClient.newJobGroup(any[JobGroupRequest])) thenAnswer {
        jobGroup: JobGroupRequest =>
          jobGroup.batch_id shouldBe backend.batchConfig.batchId
          jobGroup.absolute_parent_id shouldBe backend.batchConfig.jobGroupId
          val jobs = jobGroup.jobs
          jobs.length shouldEqual contexts.length
          jobs.foreach { payload =>
            payload.regions.value shouldBe jobConfig.regions
            payload.resources.value shouldBe JobResources(
              preemptible = true,
              cpu = Some(jobConfig.worker_cores),
              memory = Some(jobConfig.worker_memory),
              storage = Some(jobConfig.storage),
            )
          }

          (backend.batchConfig.jobGroupId + 1, 1)
      }

      // the service backend expects that each job write its output to a well-known
      // location when it finishes.
      when(batchClient.waitForJobGroup(any[Int], any[Int])) thenAnswer {
        (id: Int, jobGroupId: Int) =>
          id shouldEqual backend.batchConfig.batchId
          jobGroupId shouldEqual backend.batchConfig.jobGroupId + 1

          val resultsDir = Path(ctx.tmpdir) / "mapCollectPartitions" / tokenUrlSafe
          resultsDir.createDirectory(): Unit

          for (i <- contexts.indices)
            ctx.fs.writePDOS((resultsDir / f"result.$i").toString()) {
              os => WireProtocol.write(os, i, Right(Array.emptyByteArray))
            }

          JobGroupResponse(
            batch_id = id,
            job_group_id = jobGroupId,
            state = Success,
            complete = true,
            n_jobs = contexts.length,
            n_completed = contexts.length,
            n_succeeded = contexts.length,
            n_failed = 0,
            n_cancelled = 0,
          )
      }

      val (failure, _) =
        backend.runtimeContext(ctx).mapCollectPartitions(
          Array.emptyByteArray,
          contexts,
          "stage1",
        )((_, bytes, _, _, _) => bytes)

      failure.foreach(throw _)

      batchClient.newJobGroup(any[JobGroupRequest]) wasCalled once
      batchClient.waitForJobGroup(any[Int], any[Int]) wasCalled once
    }

  @Test def testFailedJobGroup(): Unit =
    runMock { (ctx, _, batchClient, backend) =>
      val contexts = (0 until 100).map(_ => Array.emptyByteArray)
      val startJobGroupId = 2356
      when(batchClient.newJobGroup(any[JobGroupRequest])) thenAnswer {
        _: JobGroupRequest => (backend.batchConfig.jobGroupId + 1, startJobGroupId)
      }
      val successes = Array(13, 34, 65, 81) // arbitrary indices
      val failures = Array(21, 44)
      val expectedCause = new NoSuchMethodError("")
      when(batchClient.waitForJobGroup(any[Int], any[Int])) thenAnswer {
        (id: Int, jobGroupId: Int) =>
          val resultsDir = Path(ctx.tmpdir) / "mapCollectPartitions" / tokenUrlSafe
          resultsDir.createDirectory(): Unit

          for (i <- successes)
            ctx.fs.writePDOS((resultsDir / f"result.$i").toString()) {
              os => WireProtocol.write(os, i, Right(i.toString.getBytes(StandardCharsets.UTF_8)))
            }

          for (i <- failures)
            ctx.fs.writePDOS((resultsDir / f"result.$i").toString()) {
              os => WireProtocol.write(os, i, Left(expectedCause))
            }

          JobGroupResponse(
            batch_id = id,
            job_group_id = jobGroupId,
            state = Failure,
            complete = false,
            n_jobs = contexts.length,
            n_completed = successes.length + failures.length,
            n_succeeded = successes.length,
            n_failed = failures.length,
            n_cancelled = contexts.length - failures.length - successes.length,
          )
      }
      when(batchClient.getJobGroupJobs(any[Int], any[Int], any[Option[JobState]])) thenAnswer {
        (batchId: Int, _: Int, s: Option[JobState]) =>
          s match {
            case Some(JobStates.Failed) =>
              LazyList(failures.toIndexedSeq.map(i =>
                JobListEntry(batchId, i + startJobGroupId, JobStates.Failed, 1)
              ))

            case Some(JobStates.Success) =>
              LazyList(successes.toIndexedSeq.map(i =>
                JobListEntry(batchId, i + startJobGroupId, JobStates.Success, 1)
              ))
          }
      }

      val (failure, result) =
        backend.runtimeContext(ctx).mapCollectPartitions(
          Array.emptyByteArray,
          contexts,
          "stage1",
        )((_, bytes, _, _, _) => bytes)

      val (shortMessage, expanded, id) = handleForPython(expectedCause)
      failure.value shouldBe HailWorkerException(failures.head, shortMessage, expanded, id)
      result.map(_._2) shouldBe successes
    }

  @Test def testCancelledJobGroup(): Unit =
    runMock { (ctx, _, batchClient, backend) =>
      val contexts = Array.tabulate(100)(_.toString.getBytes)
      val startJobGroupId = 2356
      when(batchClient.newJobGroup(any[JobGroupRequest])) thenAnswer {
        _: JobGroupRequest => (backend.batchConfig.jobGroupId + 1, startJobGroupId)
      }

      val successes = Array(13, 34, 65, 81) // arbitrary indices
      when(batchClient.waitForJobGroup(any[Int], any[Int])) thenAnswer {
        (id: Int, jobGroupId: Int) =>
          val resultsDir = Path(ctx.tmpdir) / "mapCollectPartitions" / tokenUrlSafe
          resultsDir.createDirectory(): Unit

          for (i <- successes)
            ctx.fs.writePDOS((resultsDir / f"result.$i").toString()) {
              os => WireProtocol.write(os, i, Right(Array.emptyByteArray))
            }

          JobGroupResponse(
            batch_id = id,
            job_group_id = jobGroupId,
            state = Cancelled,
            complete = true,
            n_jobs = contexts.length,
            n_completed = contexts.length,
            n_succeeded = successes.length,
            n_failed = 0,
            n_cancelled = contexts.length - successes.length,
          )
      }

      when(batchClient.getJobGroupJobs(any[Int], any[Int], any[Option[JobState]])) thenAnswer {
        (batchId: Int, _: Int, s: Option[JobState]) =>
          s match {
            case Some(JobStates.Success) =>
              LazyList(successes.map(i =>
                JobListEntry(batchId, i + startJobGroupId, JobStates.Success, 1)
              ).toIndexedSeq)
          }
      }

      val (failure, result) =
        backend.runtimeContext(ctx).mapCollectPartitions(
          Array.emptyByteArray,
          contexts,
          "stage1",
        )((_, bytes, _, _, _) => bytes)

      failure.value shouldBe a[CancellationException]
      result.map(_._2) shouldBe successes
    }

  def runMock(test: (ExecuteContext, BatchJobConfig, BatchClient, ServiceBackend) => Any): Unit =
    withObjectSpied[is.hail.utils.UtilsType] {
      // not obvious how to pull out `tokenUrlSafe` and inject this directory
      // using a spy is a hack and i don't particularly like it.
      when(is.hail.utils.tokenUrlSafe) thenAnswer "TOKEN"

      val jobConfig =
        BatchJobConfig(
          worker_cores = "128",
          worker_memory = "a lot.",
          storage = "a big ssd?",
          cloudfuse_configs = Array(),
          regions = Array("lunar1"),
        )

      // no idea why this trigger the missing override rule
      @SuppressWarnings(Array("org.wartremover.contrib.warts.MissingOverride"))
      val batchClient =
        mock[BatchClient]

      def serviceBackend =
        new ServiceBackend(
          name = "name",
          batchClient = batchClient,
          jarSpec = GitRevision("123"),
          batchConfig =
            BatchConfig(batchId = Random.nextInt(100), jobGroupId = Random.nextInt(100)),
          jobConfig = jobConfig,
        )

      def localTmpDirectory: Directory with Closeable =
        new Directory(Directory.makeTemp("hail-testing-tmp").jfile) with Closeable {
          override def close(): Unit =
            deleteRecursively(): Unit
        }

      using(serviceBackend) { backend =>
        using(localTmpDirectory) { tmp =>
          ctx.local(tmpdir = tmp.toString())(ctx => test(ctx, jobConfig, batchClient, backend))
        }
      }
    }
}
