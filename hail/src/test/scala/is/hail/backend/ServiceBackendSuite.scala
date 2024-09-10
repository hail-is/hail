package is.hail.backend

import is.hail.asm4s.HailClassLoader
import is.hail.backend.service.{ServiceBackend, ServiceBackendRPCPayload}
import is.hail.services.JobGroupStates.Success
import is.hail.services._
import is.hail.utils.tokenUrlSafe
import org.mockito.ArgumentMatchersSugar.any
import org.mockito.IdiomaticMockito
import org.mockito.MockitoSugar.when
import org.scalatest.matchers.should.Matchers.convertToAnyShouldWrapper
import org.scalatestplus.testng.TestNGSuite
import org.testng.annotations.Test

import scala.reflect.io.{Directory, Path}
import scala.util.Random

class ServiceBackendSuite extends TestNGSuite with IdiomaticMockito {

  @Test def testCreateJobPayload(): Unit =
    withMockDriverContext { rpcPayload =>
      val batchClient = mock[BatchClient]

      val backend =
        ServiceBackend(
          jarLocation = "us-docker.pkg.dev/hail-vdc/hail/hailgenetics/hail@sha256:fake",
          name = "name",
          theHailClassLoader = new HailClassLoader(getClass.getClassLoader),
          batchClient,
          batchId = None,
          jobGroupId = None,
          scratchDir = rpcPayload.remote_tmpdir,
          rpcConfig = rpcPayload,
          sys.env + ("HAIL_CLOUD" -> "gcp"),
        )

      val contexts = Array.tabulate(1)(_.toString.getBytes)

      // verify that the service backend
      // - creates the batch with the correct billing project, and
      // - the number of jobs matches the number of partitions, and
      // - each job is created in the specified region, and
      // - each job's resource configuration matches the rpc config
      val batchId = Random.nextInt()

      when(batchClient.newBatch(any[BatchRequest])) thenAnswer {
        (batchRequest: BatchRequest) =>
          batchRequest.billing_project shouldEqual rpcPayload.billing_project
          batchId
      }

      when(batchClient.newJobGroup(any[Int], any[String], any[JobGroupRequest], any[IndexedSeq[JobRequest]])) thenAnswer {
        (_: Int, _: String, jobGroup: JobGroupRequest, jobs: IndexedSeq[JobRequest]) =>
          jobGroup.job_group_id shouldBe 1
          jobGroup.absolute_parent_id shouldBe 0

          jobs.length shouldEqual contexts.length
          jobs.foreach { payload =>
            payload.regions shouldBe rpcPayload.regions
            payload.resources shouldBe Some(
              JobResources(
                preemptible = true,
                cpu = Some(rpcPayload.worker_cores),
                memory = Some(rpcPayload.worker_memory),
                storage = Some(rpcPayload.storage)
              )
            )
          }

          (batchId, 37)
      }

      // the service backend expects that each job write its output to a well-known
      // location when it finishes.
      when(batchClient.waitForJobGroup(any[Int], any[Int])) thenAnswer {
        (batchId: Int, jobGroupId: Int) =>
          batchId shouldEqual 37L
          jobGroupId shouldEqual 1L

          val resultsDir =
            Path(backend.serviceBackendContext.remoteTmpDir) /
              "parallelizeAndComputeWithIndex" /
              tokenUrlSafe

          resultsDir.createDirectory()
          for (i <- contexts.indices) (resultsDir / f"result.$i").toFile.writeAll("11")
          JobGroupResponse(
            batch_id = batchId,
            job_group_id = jobGroupId,
            state = Success,
            complete = true,
            n_jobs = contexts.length,
            n_completed = contexts.length,
            n_succeeded = contexts.length,
            n_failed = 0,
            n_cancelled = 0
          )
      }

      val (failure, _) =
        backend.parallelizeAndComputeWithIndex(
          backend.serviceBackendContext,
          backend.fs,
          contexts,
          "stage1",
        )((bytes, _, _, _) => bytes)

      failure.foreach(throw _)

      batchClient.newBatch(any) wasCalled once
      batchClient.newJobGroup(any, any, any, any) wasCalled once
    }

  def withMockDriverContext(test: ServiceBackendRPCPayload => Any): Any =
    withNewLocalTmpFolder { tmp =>
      // The `ServiceBackend` assumes credentials are installed to a well known location
      val gcsKeyDir = tmp / "secrets" / "gsa-key"
      gcsKeyDir.createDirectory()
      (gcsKeyDir / "key.json").toFile.writeAll("password1234")

      withObjectSpied[is.hail.utils.UtilsType] {
        // not obvious how to pull out `tokenUrlSafe` and inject this directory
        // using a spy is a hack and i don't particularly like it.
        when(is.hail.utils.tokenUrlSafe) thenAnswer "TOKEN"

        test {
          ServiceBackendRPCPayload(
            tmp_dir = "",
            remote_tmpdir = tmp.path + "/", // because raw strings...
            billing_project = "fancy",
            worker_cores = "128",
            worker_memory = "a lot.",
            storage = "a big ssd?",
            cloudfuse_configs = Array(),
            regions = Array("lunar1"),
            flags = Map(),
            custom_references = Array(),
            liftovers = Map(),
            sequences = Map(),
          )
        }
      }
    }

  def withNewLocalTmpFolder[A](f: Directory => A): A = {
    val tmp = Directory.makeTemp("hail-testing-tmp", "")
    try f(tmp)
    finally tmp.deleteRecursively()
  }

}
