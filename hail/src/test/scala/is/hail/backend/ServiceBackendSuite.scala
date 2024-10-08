package is.hail.backend

import is.hail.HailFeatureFlags
import is.hail.asm4s.HailClassLoader
import is.hail.backend.service.{
  BatchJobConfig, ServiceBackend, ServiceBackendContext, ServiceBackendRPCPayload,
}
import is.hail.io.fs.{CloudStorageFSConfig, RouterFS}
import is.hail.services._
import is.hail.services.JobGroupStates.Success
import is.hail.utils.{tokenUrlSafe, using}

import scala.collection.mutable
import scala.reflect.io.{Directory, Path}
import scala.util.Random

import java.io.Closeable

import org.mockito.ArgumentMatchersSugar.any
import org.mockito.IdiomaticMockito
import org.mockito.MockitoSugar.when
import org.scalatest.OptionValues
import org.scalatest.matchers.should.Matchers.convertToAnyShouldWrapper
import org.scalatestplus.testng.TestNGSuite
import org.testng.annotations.Test

class ServiceBackendSuite extends TestNGSuite with IdiomaticMockito with OptionValues {

  @Test def testCreateJobPayload(): Unit =
    withMockDriverContext { case (rpcConfig, jobConfig) =>
      val batchClient = mock[BatchClient]
      using(ServiceBackend(batchClient, rpcConfig, jobConfig)) { backend =>
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

            val resultsDir =
              Path(rpcConfig.remote_tmpdir) /
                "parallelizeAndComputeWithIndex" /
                tokenUrlSafe

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

        val (failure, _) =
          backend.parallelizeAndComputeWithIndex(
            ServiceBackendContext(
              remoteTmpDir = rpcConfig.remote_tmpdir,
              jobConfig = jobConfig,
              executionCache = ExecutionCache.noCache,
            ),
            backend.fs,
            contexts,
            "stage1",
          )((bytes, _, _, _) => bytes)

        failure.foreach(throw _)

        batchClient.newBatch(any) wasCalled once
        batchClient.newJobGroup(any, any, any, any) wasCalled once
      }
    }

  def ServiceBackend(
    client: BatchClient,
    rpcConfig: ServiceBackendRPCPayload,
    jobConfig: BatchJobConfig,
  ): ServiceBackend = {
    val flags = HailFeatureFlags.fromEnv()
    val fs = RouterFS.buildRoutes(CloudStorageFSConfig())
    new ServiceBackend(
      name = "name",
      batchClient = client,
      jarLocation = "us-docker.pkg.dev/hail-vdc/hail/hailgenetics/hail@sha256:fake",
      theHailClassLoader = new HailClassLoader(getClass.getClassLoader),
      batchConfig = None,
      rpcConfig = rpcConfig,
      jobConfig = jobConfig,
      flags = flags,
      fs = fs,
      references = mutable.Map.empty,
    )
  }

  def withMockDriverContext(test: (ServiceBackendRPCPayload, BatchJobConfig) => Any): Any =
    using(LocalTmpFolder) { tmp =>
      withObjectSpied[is.hail.utils.UtilsType] {
        // not obvious how to pull out `tokenUrlSafe` and inject this directory
        // using a spy is a hack and i don't particularly like it.
        when(is.hail.utils.tokenUrlSafe) thenAnswer "TOKEN"

        test(
          ServiceBackendRPCPayload(
            tmp_dir = tmp.path,
            remote_tmpdir = tmp.path,
            flags = Map(),
            custom_references = Array(),
            liftovers = Map(),
            sequences = Map(),
          ),
          BatchJobConfig(
            token = tokenUrlSafe,
            billing_project = "fancy",
            worker_cores = "128",
            worker_memory = "a lot.",
            storage = "a big ssd?",
            cloudfuse_configs = Array(),
            regions = Array("lunar1"),
          ),
        )
      }
    }

  def LocalTmpFolder: Directory with Closeable =
    new Directory(Directory.makeTemp("hail-testing-tmp").jfile) with Closeable {
      override def close(): Unit = deleteRecursively()
    }
}
