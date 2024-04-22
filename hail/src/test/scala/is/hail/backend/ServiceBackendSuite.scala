package is.hail.backend

import is.hail.asm4s.HailClassLoader
import is.hail.backend.service.{ServiceBackend, ServiceBackendRPCPayload}
import is.hail.services.batch_client.BatchClient
import is.hail.utils.tokenUrlSafe
import org.json4s.{JObject, JString}
import org.mockito.ArgumentMatchersSugar.{any, eqTo}
import org.mockito.MockitoSugar.{mock, when}
import org.scalatestplus.testng.TestNGSuite
import org.testng.annotations.Test

class ServiceBackendSuite extends TestNGSuite {

  @Test def testWorkerJobsHaveCorrectMemory(): Unit = {
    val batchClient = mock[BatchClient]

    val config =
      ServiceBackendRPCPayload(
        tmp_dir = "",
        remote_tmpdir = "",
        billing_project = "",
        worker_cores = "128",
        worker_memory = "a lot.",
        storage = "",
        cloudfuse_configs = Array(),
        regions = Array(),
        flags = Map(),
        custom_references = Array(),
        liftovers = Map(),
        sequences = Map(),
      )

    val backend =
      ServiceBackend(
        jarLocation = "/path/to/jar",
        name = "name",
        theHailClassLoader = new HailClassLoader(getClass.getClassLoader),
        batchClient,
        batchId = None,
        jobGroupId = None,
        scratchDir = "/path/to/scratch",
        rpcConfig = config,
      )

    // configure mocks
    when(batchClient.create(any[JObject], any[IndexedSeq[JObject]])) thenReturn 0L
    when(batchClient.waitForJobGroup(eqTo(0L), any[Long])) thenAnswer {
      val hideous =
        f"${backend.serviceBackendContext.remoteTmpDir}parallelizeAndComputeWithIndex/" +
          f"${tokenUrlSafe(32)}/result.0"

      backend.fs.write(hideous)(_.write(Array(0xFF, 0xFF).map(_.toByte)))
      JObject("state" -> JString("success"))
    }

    backend.parallelizeAndComputeWithIndex(
      backend.serviceBackendContext,
      backend.fs,
      Array.tabulate(1)(_.toString.getBytes),
      "stage1",
    )((bytes, _, _, _) => bytes)

    assert(true)
  }

}
