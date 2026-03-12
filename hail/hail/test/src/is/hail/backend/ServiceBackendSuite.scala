package is.hail.backend

import is.hail.HailSuite
import is.hail.backend.ExecutionCache.Flags.UseFastRestarts
import is.hail.backend.service.{BatchJobConfig, ServiceBackend, WireProtocol}
import is.hail.collection.FastSeq
import is.hail.collection.compat.immutable.ArraySeq
import is.hail.services._
import is.hail.services.JobGroupStates._
import is.hail.utils.{handleForPython, tokenUrlSafe, using, HailWorkerException}

import scala.collection.compat.immutable.LazyList
import scala.concurrent.CancellationException
import scala.reflect.io.{Directory, Path}
import scala.util.Random

import java.io.Closeable
import java.util.concurrent.CountDownLatch

import org.mockito.ArgumentMatchersSugar.any
import org.mockito.IdiomaticMockito
import org.mockito.MockitoSugar.when
import org.scalatest.OptionValues
import org.scalatest.matchers.should.Matchers.{a, convertToAnyShouldWrapper}
import org.testng.annotations.{DataProvider, Test}

class ServiceBackendSuite extends HailSuite with IdiomaticMockito with OptionValues {

  @Test def testExecutesSinglePartitionLocally(): Unit =
    runMock { (ctx, _, batchClient, backend) =>
      val contexts = ArraySeq.tabulate(10)(_ => Array.emptyByteArray)

      val (failure, _) =
        backend.runtimeContext(ctx).mapCollectPartitions(
          Array.emptyByteArray,
          contexts,
          "stage1",
          partitions = Some(FastSeq(0)),
        )((_, _, _, _) => (_, bytes) => bytes)

      failure.foreach(throw _)

      batchClient.newJobGroup(any[JobGroupRequest]) wasNever called
    }

  @Test def testCollectIncrementally(): Unit =
    runMock { (ctx, jobConfig, batchClient, backend) =>
      // the service backend expects that each job write its output to a well-known
      // location when it finishes.
      val resultsDir = Path(ctx.tmpdir) / "mapCollectPartitions" / tokenUrlSafe
      resultsDir.createDirectory(): Unit

      val contexts = ArraySeq.tabulate(5)(_ => Array.emptyByteArray)

      for (i <- contexts.indices)
        ctx.fs.writePDOS((resultsDir / f"result.$i").toString()) {
          os => WireProtocol.write(os, i, Right(Array.emptyByteArray))
        }

      // verify that
      // - the number of jobs matches the number of partitions, and
      // - each job is created in the specified region, and
      // - each job's resource configuration matches the rpc config

      val startJobId = Random.nextInt()

      when(batchClient.newJobGroup(any[JobGroupRequest])) thenAnswer {
        jobGroup: JobGroupRequest =>
          jobGroup.batch_id shouldBe backend.batchConfig.batchId
          jobGroup.absolute_parent_id shouldBe backend.batchConfig.jobGroupId
          val jobs = jobGroup.jobs
          jobs.length shouldEqual contexts.length
          jobs.foreach { payload =>
            payload.regions shouldBe jobConfig.regions
            payload.resources.value shouldBe JobResources(
              preemptible = true,
              cpu = jobConfig.worker_cores,
              memory = jobConfig.worker_memory,
              storage = jobConfig.storage,
            )
          }

          (backend.batchConfig.jobGroupId + 1, startJobId)
      }

      val endTime = Some("")
      var getJobGroupJobsCalled = 0
      when(batchClient.getJobGroupJobs(
        any[Int],
        any[Int],
        any[Option[JobState]],
        any[Option[String]],
      )) thenAnswer {
        (batchId: Int, _: Int, s: Option[JobState], t: Option[String]) =>
          s shouldBe Some(JobStates.Success)
          t shouldBe (if (getJobGroupJobsCalled > 0) endTime else None)

          // require more than one call
          // withhold one job to simulate delays in marking a job complete
          val jobs =
            contexts.indices.take(4).slice(
              getJobGroupJobsCalled * 2,
              getJobGroupJobsCalled * 2 + 2,
            ).map { i =>
              JobListEntry(
                batch_id = batchId,
                job_id = startJobId + i,
                state = JobStates.Success,
                exit_code = Some(0),
                end_time = endTime,
              )
            }

          getJobGroupJobsCalled += 1
          LazyList(jobs)
      }

      // make the driver poll for results while the job group is running
      when(batchClient.getJobGroup(any[Int], any[Int])) thenAnswer {
        (id: Int, jobGroupId: Int) =>
          id shouldEqual backend.batchConfig.batchId
          jobGroupId shouldEqual backend.batchConfig.jobGroupId + 1
          val complete = getJobGroupJobsCalled >= 2
          JobGroupResponse(
            batch_id = id,
            job_group_id = jobGroupId,
            state = if (complete) Success else Running,
            complete = complete,
            n_jobs = contexts.length,
            n_completed = getJobGroupJobsCalled * 2,
            n_succeeded = getJobGroupJobsCalled * 2,
            n_failed = 0,
            n_cancelled = 0,
          )
      }

      val (failure, results) =
        backend.runtimeContext(ctx).mapCollectPartitions(
          Array.emptyByteArray,
          contexts,
          "stage1",
        )((_, _, _, _) => (_, bytes) => bytes)

      failure.foreach(throw _)

      results.length shouldBe contexts.length
      batchClient.newJobGroup(any[JobGroupRequest]) wasCalled once
      batchClient.getJobGroup(any[Int], any[Int]) wasCalled thrice
      batchClient.getJobGroupJobs(
        any[Int],
        any[Int],
        any[Option[JobState]],
        any[Option[String]],
      ) wasCalled thrice
    }

  @DataProvider(name = "UseFastRestarts")
  def useFastRestarts: Array[Array[Any]] =
    Array(Array(null), Array("1"))

  @Test(dataProvider = "UseFastRestarts")
  def testFailedJobGroup(useFastRestarts: String): Unit =
    runMock { (ctx, _, batchClient, backend) =>
      ctx.local(flags = ctx.flags + (UseFastRestarts -> useFastRestarts)) { ctx =>
        val contexts = ArraySeq.tabulate(100)(_ => Array.emptyByteArray)
        val startJobId = 2356
        when(batchClient.newJobGroup(any[JobGroupRequest])) thenAnswer {
          _: JobGroupRequest => (backend.batchConfig.jobGroupId + 1, startJobId)
        }

        val resultsDir = Path(ctx.tmpdir) / "mapCollectPartitions" / tokenUrlSafe
        resultsDir.createDirectory(): Unit

        val successes = ArraySeq(13, 34, 81) // arbitrary indices
        if (ctx.flags.isDefined(UseFastRestarts))
          for (i <- successes)
            ctx.fs.writePDOS((resultsDir / f"result.$i").toString()) {
              os => WireProtocol.write(os, i, Right(i.toString.getBytes()))
            }

        val failures = ArraySeq(21)
        val expectedCause = new NoSuchMethodError("")
        for (i <- failures)
          ctx.fs.writePDOS((resultsDir / f"result.$i").toString()) {
            os => WireProtocol.write(os, i, Left(expectedCause))
          }

        when(batchClient.getJobGroup(any[Int], any[Int])) thenAnswer {
          (id: Int, jobGroupId: Int) =>
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

        when(batchClient.getJobGroupJobs(
          any[Int],
          any[Int],
          any[Option[JobState]],
          any[Option[String]],
        )) thenAnswer {
          (batchId: Int, _: Int, s: Option[JobState], _: Option[String]) =>
            s match {
              case Some(JobStates.Failed) =>
                LazyList(failures.map(i =>
                  JobListEntry(
                    batch_id = batchId,
                    job_id = i + startJobId,
                    state = JobStates.Failed,
                    exit_code = Some(1),
                    end_time = Some(""),
                  )
                ))

              case Some(JobStates.Success) =>
                ctx.flags.isDefined(UseFastRestarts) shouldBe true
                LazyList(successes.map(i =>
                  JobListEntry(
                    batch_id = batchId,
                    job_id = i + startJobId,
                    state = JobStates.Success,
                    exit_code = Some(0),
                    end_time = Some(""),
                  )
                ))
            }
        }

        val (failure, result) =
          backend.runtimeContext(ctx).mapCollectPartitions(
            Array.emptyByteArray,
            contexts,
            "stage1",
          )((_, _, _, _) => (_, bytes) => bytes)

        val (shortMessage, expanded, id) = handleForPython(expectedCause)
        failure.value shouldBe HailWorkerException(failures.head, shortMessage, expanded, id)
        if (ctx.flags.isDefined(UseFastRestarts)) result.map(_._2) shouldBe successes
      }
    }

  @Test def testCancelledJobGroup(): Unit =
    runMock { (ctx, _, batchClient, backend) =>
      val contexts = Array.tabulate(2)(_ => Array.emptyByteArray)
      val startJobId = 2356
      when(batchClient.newJobGroup(any[JobGroupRequest])) thenAnswer {
        _: JobGroupRequest => (backend.batchConfig.jobGroupId + 1, startJobId)
      }

      val successes = Array(13, 34, 65, 81) // arbitrary indices
      when(batchClient.getJobGroup(any[Int], any[Int])) thenAnswer {
        (id: Int, jobGroupId: Int) =>
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

      val (failure, _) =
        backend.runtimeContext(ctx).mapCollectPartitions(
          Array.emptyByteArray,
          contexts,
          "stage1",
        )((_, _, _, _) => (_, bytes) => bytes)

      failure.value shouldBe a[CancellationException]
    }

  @Test def testInterrupt(): Unit =
    runMock { (ctx, _, batchClient, backend) =>
      val contexts = Array.tabulate(2)(_ => Array.emptyByteArray)
      val jobGroupId = Random.nextInt()

      when(batchClient.newJobGroup(any[JobGroupRequest])) thenAnswer {
        _: JobGroupRequest => (jobGroupId, 2356)
      }

      when(batchClient.getJobGroup(any[Int], any[Int])) thenAnswer {
        (id: Int, jobGroupId: Int) =>
          JobGroupResponse(
            batch_id = id,
            job_group_id = jobGroupId,
            state = Running,
            complete = false,
            n_jobs = contexts.length,
            n_completed = 0,
            n_succeeded = 0,
            n_failed = 0,
            n_cancelled = 0,
          )
      }

      val latch = new CountDownLatch(1)

      when(batchClient.getJobGroupJobs(
        any[Int],
        any[Int],
        any[Option[JobState]],
        any[Option[String]],
      )) thenAnswer {
        (_: Int, _: Int, _: Option[JobState], _: Option[String]) =>
          latch.countDown()
          LazyList()
      }

      when(batchClient.cancelJobGroup(any[Int], any[Int])) thenAnswer {
        (batchId: Int, jgId: Int) =>
          batchId shouldBe backend.batchConfig.batchId
          jgId shouldBe jobGroupId
      }

      @volatile var failure: Option[Throwable] =
        None

      val t =
        new Thread(() =>
          failure =
            backend.runtimeContext(ctx).mapCollectPartitions(
              Array.emptyByteArray,
              contexts,
              "stage1",
            )((_, _, _, _) => (_, bytes) => bytes)._1
        )

      t.start()
      latch.await()
      t.interrupt()
      t.join()

      failure.value shouldBe a[CancellationException]
      batchClient.cancelJobGroup(any[Int], any[Int]) wasCalled once
    }

  def runMock(test: (ExecuteContext, BatchJobConfig, BatchClient, ServiceBackend) => Any): Unit =
    withObjectSpied[is.hail.utils.UtilsType] {
      // not obvious how to pull out `tokenUrlSafe` and inject this directory
      // using a spy is a hack and i don't particularly like it.
      when(is.hail.utils.tokenUrlSafe) thenAnswer "TOKEN"

      val jobConfig =
        BatchJobConfig(
          worker_cores = Some("128"),
          worker_memory = Some("a lot."),
          storage = Some("a big ssd?"),
          cloudfuse_configs = None,
          regions = Some(Array("lunar1")),
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
          maxReadParallelism = 2,
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
