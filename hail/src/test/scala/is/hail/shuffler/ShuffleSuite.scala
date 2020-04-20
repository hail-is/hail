package is.hail.shuffler

import org.apache.log4j.Logger;
import is.hail.annotations._
import is.hail.expr.ir.ExecuteContext
import is.hail.expr.types.virtual._
import is.hail.expr.types.physical._
import is.hail.io.{ BufferSpec, TypedCodecSpec }
import is.hail.testUtils._
import is.hail.utils._
import is.hail.{HailSuite, TestUtils}
import org.scalatest.testng.TestNGSuite
import org.testng.annotations.Test

import scala.util.Random
import scala.language.implicitConversions

class ShuffleSuite extends TestNGSuite {
  val log = Logger.getLogger(this.getClass.getName());

  @Test def testShuffle() {
    val port = 8080
    val server = new ShuffleServer(sslContext(
      "src/test/resources/non-secret-key-and-trust-stores/server-keystore.p12",
      "hailhail",
      "src/test/resources/non-secret-key-and-trust-stores/server-truststore.p12",
      "hailhail"
    ),
      port)
    server.serveInBackground()
    try {
      val pt = PCanonicalStruct("x" -> PInt32())
      val t = pt.virtualType
      val key = Array("x")
      val c = new ShuffleClient(
        t,
        TypedCodecSpec(pt, BufferSpec.unblockedUncompressed),
        key,
        sslContext(
          "src/test/resources/non-secret-key-and-trust-stores/client-keystore.p12",
          "hailhail",
          "src/test/resources/non-secret-key-and-trust-stores/client-truststore.p12",
          "hailhail"),
        "localhost",
        port)
      val keyPType = pt.selectFields(key)

      val values = new ArrayBuilder[Long]()
      Region.scoped { region =>
        val rvb = new RegionValueBuilder(region)
        val max = 1000000
        val shuffled = Random.shuffle((0 until max).toIndexedSeq).toArray
        var i = 0
        while (i < max) {
          rvb.start(pt)
          rvb.startStruct()
          rvb.addInt(shuffled(i))
          rvb.endStruct()
          values += rvb.end()
          i += 1
        }

        c.start()
        c.put(values.result().iterator)

        rvb.start(keyPType)
        rvb.startStruct()
        rvb.addInt(0)
        rvb.endStruct()
        val left = rvb.end()
        rvb.start(keyPType)
        rvb.startStruct()
        rvb.addInt(max)
        rvb.endStruct()
        val right = rvb.end()

        val result = c.get(region, left, right)
        val readableArray = result.map(new UnsafeRow(c.keyedCodecSpec.pType, null, _)).toIndexedSeq
        i = 0
        while (i < max) {
          assert(c.keyedCodecSpec.pType.isFieldDefined(result(i), 0),
            s"first field is undefined ${readableArray(i)}")
          assert(Region.loadInt(c.keyedCodecSpec.pType.loadField(result(i), 0)) == i,
            s"first field should be ${i}: ${readableArray(i)}. Context: ${readableArray.slice(i-3, i+3)}. Length: ${result.length}")
          i += 1
        }
        assert(result.length == max)
      }
      c.stop()
    } finally {
      server.stop()
    }
  }
}
