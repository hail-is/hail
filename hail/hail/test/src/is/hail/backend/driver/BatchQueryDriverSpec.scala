package is.hail.backend.driver

import is.hail.io.fs.RequesterPaysConfig

import org.json4s._
import org.testng.annotations.{DataProvider, Test}

class BatchQueryDriverSpec {

  @DataProvider(name = "RequesterPaysConfig")
  def dataRequesterPaysConfig: Array[Array[Any]] =
    Array(
      Array(
        JNull,
        None,
      ),
      Array(
        JArray(List(JString("my-project"), JNull)),
        Some(RequesterPaysConfig("my-project", None)),
      ),
      Array(
        JArray(List(JString("my-project"), JArray(List(JString("my-bucket"))))),
        Some(RequesterPaysConfig("my-project", Some(Set("my-bucket")))),
      ),
    )

  @Test(dataProvider = "RequesterPaysConfig")
  def testRPCConfigDeserializer(rpjson: JValue, expected: Option[RequesterPaysConfig]): Unit = {
    val rpcConfig: JObject =
      JObject(
        "tmp_dir" -> JString("gs://tmp-bucket/tmp/hail"),
        "flags" -> JObject(),
        "requester_pays_config" -> rpjson,
        "custom_references" -> JArray(List()),
        "liftovers" -> JObject(List()),
        "sequences" -> JObject(List()),
        "max_read_parallelism" -> JNull,
      )

    implicit val fmts: Formats = DefaultFormats + RequesterPaysConfigFormats

    assert(rpcConfig.extract[ServiceBackendRPCConfig].requester_pays_config == expected)
  }

}
