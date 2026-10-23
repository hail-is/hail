package is.hail.backend.driver

import is.hail.ParameterizedTest
import is.hail.TestUtils.{assertEq, interceptFatal}
import is.hail.io.fs.RequesterPaysConfig
import is.hail.variant.{ReferenceGenome, ReferenceGenomeEnv}

import scala.collection.immutable.ArraySeq

import org.json4s._
import org.junit.jupiter.api.Test

class BatchQueryDriverSpec {

  def dataRequesterPaysConfig =
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

  @ParameterizedTest("dataRequesterPaysConfig")
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

    assertEq(rpcConfig.extract[ServiceBackendRPCConfig].requester_pays_config, expected)
  }

  private def envWith(references: ReferenceGenomeEnv): BatchQueryDriver.Env =
    BatchQueryDriver.Env(
      backend = null,
      flags = null,
      hcl = null,
      fs = null,
      tmpdir = null,
      references = references,
      liftovers = Map.empty,
      sequences = Map.empty,
      outputUrl = null,
      action = 0,
      payload = JNothing,
    )

  private def mk(name: String, length: Int): ReferenceGenome =
    ReferenceGenome(name, ArraySeq("1"), Map("1" -> length))

  @Test def testPutReferencesFailsOnACollision(): Unit = {
    val env = envWith(ReferenceGenomeEnv(Seq(mk("foo", 10))))

    BatchQueryDriver.Context.putReferences(env)(Seq(mk("bar", 10), mk("foo", 10)))
    interceptFatal("a different reference with that name already exists") {
      BatchQueryDriver.Context.putReferences(env)(Seq(mk("foo", 11)))
    }
    interceptFatal("a different reference with that name already exists") {
      BatchQueryDriver.Context.putReferences(env)(Seq(mk(ReferenceGenome.GRCh38, 10)))
    }
  }
}
