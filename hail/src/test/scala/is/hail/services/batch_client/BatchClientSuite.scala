package is.hail.services.batch_client

import is.hail.utils._

import org.json4s.JsonAST.{JArray, JBool, JInt, JObject, JString}
import org.json4s.{DefaultFormats, Formats}
import org.scalatest.testng.TestNGSuite
import org.testng.annotations.Test

class BatchClientSuite extends TestNGSuite {
  @Test def testBasic(): Unit = {
    val client = new BatchClient()
    val token = tokenUrlSafe(32)
    val batch = client.run(
      JObject(
        "billing_project" -> JString("test"),
        "n_jobs" -> JInt(1),
        "token" -> JString(token)),
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
    implicit val formats: Formats = DefaultFormats
    assert((batch \ "state").extract[String] == "success")
  }
}
