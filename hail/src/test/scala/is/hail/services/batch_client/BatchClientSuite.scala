package is.hail.services.batch_client

import org.json4s.JsonAST.{JArray, JBool, JInt, JObject, JString}
import org.scalatest.testng.TestNGSuite
import org.testng.annotations.Test
import java.util.{Base64, Random}

import is.hail.utils.FastIndexedSeq
import org.json4s.{DefaultFormats, Formats}

class BatchClientSuite extends TestNGSuite {
  @Test def testBasic(): Unit = {
    val client = new BatchClient()

    val bytes = new Array[Byte](32)
    val random = new java.util.Random()
    random.nextBytes(bytes)
    val token = Base64.getUrlEncoder.encodeToString(bytes)
    println(s"token $token")

    val resp = client.post(
      "/api/v1alpha/batches/create",
      json = JObject(
        "billing_project" -> JString("test"),
        "n_jobs" -> JInt(1),
        "token" -> JString(token)))

    implicit val formats: Formats = DefaultFormats
    val batchID = (resp \ "id").extract[Long]

    client.createJobs(
      batchID,
      FastIndexedSeq(
        JObject(
          "always_run" -> JBool(false),
          "image" -> JString("ubuntu:18.04"),
          "mount_docker_socket" -> JBool(false),
          "command" -> JArray(List(
            JString("/bin/bash"),
            JString("-c"),
            JString("echo 'Hello, world!'"))),
          "job_id" -> JInt(0),
          "parent_ids" -> JArray(List()))))

    client.patch(
      s"/api/v1alpha/batches/$batchID/close")

    val batch = client.waitForBatch(batchID)

    assert((batch \ "state").extract[String] == "Success")
  }
}
