package is.hail.backend

import is.hail.HailSuite
import is.hail.backend.api.BatchJobConfig
import is.hail.backend.service.ServiceBackend
import is.hail.services._
import is.hail.services.JobGroupStates.Success
import is.hail.utils.{tokenUrlSafe, using}

import scala.reflect.io.Directory
import scala.util.Random

import java.io.Closeable

import org.mockito.ArgumentMatchersSugar.any
import org.mockito.IdiomaticMockito
import org.mockito.MockitoSugar.when
import org.scalatest.OptionValues
import org.scalatest.matchers.should.Matchers.convertToAnyShouldWrapper
import org.testng.annotations.Test

class ServiceBackendSuite extends HailSuite with IdiomaticMockito with OptionValues {

  @Test def testCreateJobPayload(): Unit =
    withObjectSpied[is.hail.utils.UtilsType] {
      // not obvious how to pull out `tokenUrlSafe` and inject this directory
      // using a spy is a hack and i don't particularly like it.
      when(is.hail.utils.tokenUrlSafe) thenAnswer "TOKEN"

      val jobConfig = BatchJobConfig(
        token = tokenUrlSafe,
        billing_project = "fancy",
        worker_cores = "128",
        worker_memory = "a lot.",
        storage = "a big ssd?",
        cloudfuse_configs = Array(),
        regions = Array("lunar1"),
      )

      val batchClient = mock[BatchClient]
      using(ServiceBackend(batchClient, jobConfig)) { backend =>
        using(LocalTmpFolder) { tmp =>
          val contexts = Array.tabulate(1)(_.toString.getBytes)

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

              backend.batchConfig.jobGroupId + 1
          }

          // the service backend expects that each job write its output to a well-known
          // location when it finishes.
          when(batchClient.waitForJobGroup(any[Int], any[Int])) thenAnswer {
            (id: Int, jobGroupId: Int) =>
              id shouldEqual backend.batchConfig.batchId
              jobGroupId shouldEqual backend.batchConfig.jobGroupId + 1

              val resultsDir = tmp / "parallelizeAndComputeWithIndex" / tokenUrlSafe

              resultsDir.createDirectory()
              for (i <- contexts.indices) (resultsDir / f"result.$i").toFile.writeAll("11")
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

          ctx.local(tmpdir = tmp.toString()) { ctx =>
            val (failure, _) =
              backend.parallelizeAndComputeWithIndex(
                backend.backendContext(ctx),
                ctx.fs,
                contexts,
                "stage1",
              )((bytes, _, _, _) => bytes)

            failure.foreach(throw _)
          }
          batchClient.newJobGroup(any) wasCalled once
          batchClient.waitForJobGroup(any, any) wasCalled once

        }
      }
    }

  def ServiceBackend(client: BatchClient, jobConfig: BatchJobConfig): ServiceBackend =
    new ServiceBackend(
      name = "name",
      batchClient = client,
      jarSpec = GitRevision("123"),
      batchConfig = BatchConfig(batchId = Random.nextInt(), jobGroupId = Random.nextInt()),
      jobConfig = jobConfig,
    )

  def LocalTmpFolder: Directory with Closeable =
    new Directory(Directory.makeTemp("hail-testing-tmp").jfile) with Closeable {
      override def close(): Unit = deleteRecursively()
    }
}
