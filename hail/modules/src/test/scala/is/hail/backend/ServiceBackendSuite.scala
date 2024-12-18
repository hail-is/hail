package is.hail.backend

import is.hail.HailFeatureFlags
import is.hail.asm4s.HailClassLoader
import is.hail.backend.service.{ServiceBackend, ServiceBackendContext, ServiceBackendRPCPayload}
import is.hail.io.fs.{CloudStorageFSConfig, RouterFS}
import is.hail.services._
import is.hail.services.JobGroupStates.Success
import is.hail.utils.{tokenUrlSafe, using}

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
    withMockDriverContext { rpcConfig =>
      val batchClient = mock[BatchClient]
      using(ServiceBackend(batchClient, rpcConfig)) { backend =>
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
              payload.regions.value shouldBe rpcConfig.regions
              payload.resources.value shouldBe JobResources(
                preemptible = true,
                cpu = Some(rpcConfig.worker_cores),
                memory = Some(rpcConfig.worker_memory),
                storage = Some(rpcConfig.storage),
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

            val resultsDir =
              Path(backend.serviceBackendContext.remoteTmpDir) /
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
            backend.serviceBackendContext,
            backend.fs,
            contexts,
            "stage1",
          )((bytes, _, _, _) => bytes)

        failure.foreach(throw _)

        batchClient.newJobGroup(any) wasCalled once
        batchClient.waitForJobGroup(any, any) wasCalled once
      }
    }

  def ServiceBackend(client: BatchClient, rpcConfig: ServiceBackendRPCPayload): ServiceBackend = {
    val flags = HailFeatureFlags.fromEnv()
    val fs = RouterFS.buildRoutes(CloudStorageFSConfig())
    new ServiceBackend(
      jarSpec = GitRevision("123"),
      name = "name",
      theHailClassLoader = new HailClassLoader(getClass.getClassLoader),
      batchClient = client,
      batchConfig = BatchConfig(batchId = Random.nextInt(), jobGroupId = Random.nextInt()),
      flags = flags,
      tmpdir = rpcConfig.tmp_dir,
      fs = fs,
      serviceBackendContext =
        new ServiceBackendContext(
          rpcConfig.billing_project,
          rpcConfig.remote_tmpdir,
          rpcConfig.worker_cores,
          rpcConfig.worker_memory,
          rpcConfig.storage,
          rpcConfig.regions,
          rpcConfig.cloudfuse_configs,
          profile = false,
          ExecutionCache.fromFlags(flags, fs, rpcConfig.remote_tmpdir),
        ),
      scratchDir = rpcConfig.remote_tmpdir,
    )
  }

  def withMockDriverContext(test: ServiceBackendRPCPayload => Any): Any =
    using(LocalTmpFolder) { tmp =>
      withObjectSpied[is.hail.utils.UtilsType] {
        // not obvious how to pull out `tokenUrlSafe` and inject this directory
        // using a spy is a hack and i don't particularly like it.
        when(is.hail.utils.tokenUrlSafe) thenAnswer "TOKEN"

        test {
          ServiceBackendRPCPayload(
            tmp_dir = tmp.path,
            remote_tmpdir = tmp.path,
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

  def LocalTmpFolder: Directory with Closeable =
    new Directory(Directory.makeTemp("hail-testing-tmp").jfile) with Closeable {
      override def close(): Unit = deleteRecursively()
    }
}
