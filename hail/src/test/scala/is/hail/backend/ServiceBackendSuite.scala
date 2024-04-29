package is.hail.backend

import is.hail.asm4s.HailClassLoader
import is.hail.backend.service.{ServiceBackend, ServiceBackendRPCPayload}
import is.hail.services.batch_client.BatchClient
import is.hail.utils.tokenUrlSafe
import org.json4s.{JBool, JObject, JString}
import org.mockito.ArgumentMatchersSugar.{any, eqTo}
import org.mockito.IdiomaticMockito
import org.mockito.MockitoSugar.when
import org.scalatestplus.testng.TestNGSuite
import org.testng.annotations.Test

import scala.reflect.io.{Directory, Path}

class ServiceBackendSuite extends TestNGSuite with IdiomaticMockito {

  def withNewLocalTmpFolder[A](f: Directory => A): A = {
    val tmp = Directory.makeTemp("hail-testing-tmp", "")
    try f(tmp)
    finally tmp.deleteRecursively()
  }

  @Test def testWorkerJobsHaveCorrectResources(): Unit =
    withNewLocalTmpFolder { scratchDir =>
      withObjectSpied[is.hail.utils.UtilsType] {

        val batchClient = mock[BatchClient]

        val config =
          ServiceBackendRPCPayload(
            tmp_dir = scratchDir.path,
            remote_tmpdir = f"$scratchDir/",
            billing_project = "",
            worker_cores = "128",
            worker_memory = "a lot.",
            storage = "a big ssd?",
            cloudfuse_configs = Array(),
            regions = Array(),
            flags = Map(),
            custom_references = Array(),
            liftovers = Map(),
            sequences = Map(),
          )

        // Workers have cloud credentials installed to a well-known location
        val gcsKeyDir = scratchDir / "secrets" / "gsa-key"
        gcsKeyDir.createDirectory()
        (gcsKeyDir / "key.json").toFile.writeAll("password1234")

        val backend =
          ServiceBackend(
            jarLocation =
              classOf[ServiceBackend].getProtectionDomain.getCodeSource.getLocation.getPath,
            name = "name",
            theHailClassLoader = new HailClassLoader(getClass.getClassLoader),
            batchClient,
            batchId = None,
            jobGroupId = None,
            scratchDir = scratchDir.toString,
            rpcConfig = config,
          )

        val contexts = Array.tabulate(1)(_.toString.getBytes)

        // configure mocks
        when(is.hail.utils.tokenUrlSafe(any)) thenAnswer "{random}"
        when(batchClient.create(any[JObject], any[IndexedSeq[JObject]])) thenAnswer {
          (_: JObject, jobs: IndexedSeq[JObject]) =>
            assert(jobs.length == contexts.length)
            jobs.foreach { payload =>
              assert(payload \ "resources" == JObject(
                "preemptible" -> JBool(true),
                "cpu" -> JString(config.worker_cores),
                "memory" -> JString(config.worker_memory),
                "storage" -> JString(config.storage),
              ))
            }
            0L
        }
        when(batchClient.waitForJobGroup(eqTo(0L), any[Long])) thenAnswer {
          val resultsDir =
            Path(backend.serviceBackendContext.remoteTmpDir) /
              "parallelizeAndComputeWithIndex" /
              tokenUrlSafe(32)

          resultsDir.createDirectory()
          for (i <- contexts.indices) (resultsDir / f"result.$i").toFile.writeAll("11")
          JObject("state" -> JString("success"))
        }

        backend.parallelizeAndComputeWithIndex(
          backend.serviceBackendContext,
          backend.fs,
          contexts,
          "stage1",
        )((bytes, _, _, _) => bytes)

        batchClient.create(any[JObject], any[IndexedSeq[JObject]]) wasCalled once
      }
    }

}
