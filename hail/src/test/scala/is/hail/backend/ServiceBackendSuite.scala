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

          // verify that the service backend
          // - creates the batch with the correct billing project, and
          // - the number of jobs matches the number of partitions, and
          // - each job is created in the specified region, and
          // - each job's resource configuration matches the rpc config
          val batchId = Random.nextInt()

          when(batchClient.newBatch(any[BatchRequest])) thenAnswer {
            (batchRequest: BatchRequest) =>
              batchRequest.billing_project shouldEqual jobConfig.billing_project
              batchRequest.n_jobs shouldBe 0
              batchRequest.attributes.get("name").value shouldBe backend.name
              batchId
          }

          when(batchClient.newJobGroup(
            any[Int],
            any[String],
            any[JobGroupRequest],
            any[IndexedSeq[JobRequest]],
          )) thenAnswer {
            (id: Int, _: String, jobGroup: JobGroupRequest, jobs: IndexedSeq[JobRequest]) =>
              id shouldBe batchId
              jobGroup.job_group_id shouldBe 1
              jobGroup.absolute_parent_id shouldBe 0
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

              (37, 1)
          }

          // the service backend expects that each job write its output to a well-known
          // location when it finishes.
          when(batchClient.waitForJobGroup(any[Int], any[Int])) thenAnswer {
            (id: Int, jobGroupId: Int) =>
              id shouldEqual batchId
              jobGroupId shouldEqual 1

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

          batchClient.newBatch(any) wasCalled once
          batchClient.newJobGroup(any, any, any, any) wasCalled once
        }
      }
    }

  def ServiceBackend(client: BatchClient, jobConfig: BatchJobConfig): ServiceBackend =
    new ServiceBackend(
      name = "name",
      batchClient = client,
      jarLocation = "us-docker.pkg.dev/hail-vdc/hail/hailgenetics/hail@sha256:fake",
      batchConfig = None,
      jobConfig = jobConfig,
    )

  def LocalTmpFolder: Directory with Closeable =
    new Directory(Directory.makeTemp("hail-testing-tmp").jfile) with Closeable {
      override def close(): Unit = deleteRecursively()
    }
}
