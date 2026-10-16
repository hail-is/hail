package is.hail.backend.driver

import is.hail.ParameterizedTest
import is.hail.TestUtils.{assertEq, getTestResource, interceptFatal}
import is.hail.backend.ExecuteContext
import is.hail.expr.ir.EmitFunctionBuilder
import is.hail.io.fs.RequesterPaysConfig
import is.hail.io.reference.RgAttachments
import is.hail.variant.{Locus, ReferenceGenome, ReferenceGenomeEnv}

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
      attachments = RgAttachments.empty,
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

  // Python may still record an attachment to a reference genome it has removed. Launching must not
  // fail every RPC for it: only a use of the attachment fails.
  @Test def testLaunchSkipsAttachmentsToAGenomeNotInScope(implicit ctx: ExecuteContext): Unit = {
    val grch37 = ReferenceGenome.builtinReferences()(ReferenceGenome.GRCh37)
    val grch38 = ReferenceGenome.builtinReferences()(ReferenceGenome.GRCh38)
    val removed =
      ReferenceGenome("removed", ArraySeq("chr12", "chr20"), Map("chr12" -> 10, "chr20" -> 10))
    val chain = getTestResource("grch37_to_grch38_chr20.over.chain.gz")

    val attachments =
      BatchQueryDriver.loadAttachments(
        ctx.fs,
        ReferenceGenomeEnv.empty,
        Map(ReferenceGenome.GRCh37 -> Map(ReferenceGenome.GRCh38 -> chain, "removed" -> chain)),
        Map("removed" -> SequenceConfig(
          getTestResource("fake_reference.fasta"),
          getTestResource("fake_reference.fasta.fai"),
        )),
      )

    assertEq(attachments.liftovers.keySet, Set((grch37, grch38)))
    assertEq(attachments.sequences, Map.empty)

    ctx.local(attachments = attachments) { ctx =>
      val fb = EmitFunctionBuilder[Locus, Double, (Locus, Boolean)](ctx, "lift_to_removed")
      interceptFatal("No liftover from reference genome 'GRCh37' to 'removed'") {
        fb.getLiftover(grch37, removed)
      }
    }
  }
}
