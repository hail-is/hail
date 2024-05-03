package is.hail.backend

import is.hail.asm4s.HailClassLoader
import is.hail.backend.service.{ServiceBackend, ServiceBackendRPCPayload}
import is.hail.services.batch_client.BatchClient
import is.hail.utils.tokenUrlSafe
import org.json4s.{JArray, JBool, JInt, JObject, JString}
import org.mockito.ArgumentMatchersSugar.{any, eqTo}
import org.mockito.IdiomaticMockito
import org.mockito.MockitoSugar.when
import org.scalatest.matchers.should.Matchers.convertToAnyShouldWrapper
import org.scalatestplus.testng.TestNGSuite
import org.testng.annotations.Test

import scala.reflect.io.{Directory, Path}

class ServiceBackendSuite extends TestNGSuite with IdiomaticMockito {

  @Test def testCreateJobPayload(): Unit =
    withMockDriverContext { rpcPayload =>
      val batchClient = mock[BatchClient]

      val backend =
        ServiceBackend(
          jarLocation =
            classOf[ServiceBackend].getProtectionDomain.getCodeSource.getLocation.getPath,
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

      when(batchClient.create(any[JObject], any[IndexedSeq[JObject]])) thenAnswer {
        (batch: JObject, jobs: IndexedSeq[JObject]) =>
          batch \ "billing_project" shouldBe JString(rpcPayload.billing_project)
          batch \ "n_jobs" shouldBe JInt(contexts.length)

          jobs.length shouldEqual contexts.length
          jobs.foreach { payload =>
            payload \ "regions" shouldBe JArray(rpcPayload.regions.map(JString).toList)

            payload \ "resources" shouldBe JObject(
              "preemptible" -> JBool(true),
              "cpu" -> JString(rpcPayload.worker_cores),
              "memory" -> JString(rpcPayload.worker_memory),
              "storage" -> JString(rpcPayload.storage),
            )
          }

          37L
      }

      when(batchClient.waitForJobGroup(eqTo(37L), eqTo(1L))) thenAnswer {
        val resultsDir =
          Path(backend.serviceBackendContext.remoteTmpDir) /
            "parallelizeAndComputeWithIndex" /
            tokenUrlSafe

        resultsDir.createDirectory()
        for (i <- contexts.indices) (resultsDir / f"result.$i").toFile.writeAll("11")
        JObject("state" -> JString("success"))
      }

      val (failure, _) =
        backend.parallelizeAndComputeWithIndex(
          backend.serviceBackendContext,
          backend.fs,
          contexts,
          "stage1",
        )((bytes, _, _, _) => bytes)

      failure.foreach(throw _)

      batchClient.create(any[JObject], any[IndexedSeq[JObject]]) wasCalled once
    }

  @Test def testUpdateJobPayload(): Unit =
    withMockDriverContext { config =>
      val batchClient = mock[BatchClient]

      val backend =
        ServiceBackend(
          jarLocation =
            classOf[ServiceBackend].getProtectionDomain.getCodeSource.getLocation.getPath,
          name = "name",
          theHailClassLoader = new HailClassLoader(getClass.getClassLoader),
          batchClient,
          batchId = Some(23L),
          jobGroupId = None,
          scratchDir = config.remote_tmpdir,
          rpcConfig = config,
          sys.env + ("HAIL_CLOUD" -> "gcp"),
        )

      val contexts = Array.tabulate(1)(_.toString.getBytes)

      when(
        batchClient.update(any[Long], any[String], any[JObject], any[IndexedSeq[JObject]])
      ) thenAnswer {
        (batchId: Long, _: String, _: JObject, jobs: IndexedSeq[JObject]) =>
          batchId shouldEqual 23L

          jobs.length shouldEqual contexts.length
          jobs.foreach { payload =>
            payload \ "regions" shouldBe JArray(config.regions.map(JString).toList)

            payload \ "resources" shouldBe JObject(
              "preemptible" -> JBool(true),
              "cpu" -> JString(config.worker_cores),
              "memory" -> JString(config.worker_memory),
              "storage" -> JString(config.storage),
            )
          }

          (2L, 3L)
      }

      when(batchClient.waitForJobGroup(eqTo(23L), eqTo(3L))) thenAnswer {
        val resultsDir =
          Path(backend.serviceBackendContext.remoteTmpDir) /
            "parallelizeAndComputeWithIndex" /
            tokenUrlSafe

        resultsDir.createDirectory()
        for (i <- contexts.indices) (resultsDir / f"result.$i").toFile.writeAll("11")
        JObject("state" -> JString("success"))
      }

      val (failure, _) =
        backend.parallelizeAndComputeWithIndex(
          backend.serviceBackendContext,
          backend.fs,
          contexts,
          "stage1",
        )((bytes, _, _, _) => bytes)

      failure.foreach(throw _)

      batchClient.create(any[JObject], any[IndexedSeq[JObject]]) wasNever called
      batchClient.update(
        any[Long],
        any[String],
        any[JObject],
        any[IndexedSeq[JObject]],
      ) wasCalled once
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
