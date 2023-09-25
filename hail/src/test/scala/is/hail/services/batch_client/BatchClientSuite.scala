package is.hail.services.batch_client

import is.hail.utils._
import org.json4s.JsonAST._
import org.json4s.{DefaultFormats, Formats}
import org.scalatest.testng.TestNGSuite
import org.testng.annotations.Test

class BatchClientSuite extends TestNGSuite {
  @Test def testBasic(): Unit = {
    val client = new BatchClient("/test-gsa-key/key.json")
    val token = tokenUrlSafe(32)
    val batch = client.run(
      JObject(
        "billing_project" -> JString("test"),
        "n_jobs" -> JInt(1),
        "token" -> JString(token)),
      FastSeq(
        JObject(
          "always_run" -> JBool(false),
          "job_id" -> JInt(0),
          "parent_ids" -> JArray(List()),
          "process" -> JObject(
            "image" -> JString("ubuntu:22.04"),
            "command" -> JArray(List(
              JString("/bin/bash"),
              JString("-c"),
              JString("echo 'Hello, world!'"))),
            "type" -> JString("docker"))
        )))
    implicit val formats: Formats = DefaultFormats
    assert((batch \ "state").extract[String] == "success")
  }
}
