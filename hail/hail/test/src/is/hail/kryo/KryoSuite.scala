package is.hail.kryo

import is.hail.ParameterizedTest
import is.hail.collection.compat.immutable.ArraySeq

import com.esotericsoftware.kryo.Kryo
import com.esotericsoftware.kryo.io.{Input, Output}
import org.junit.jupiter.api.BeforeAll

class KryoSuite {
  private var kryo: Kryo = _

  @BeforeAll def setupClass(): Unit = {
    kryo = new Kryo()
    new HailKryoRegistrator().registerClasses(kryo)
  }

  def ArraySeqSerialization() =
    ArraySeq(
      null,
      ArraySeq.empty[String],
      ArraySeq("a", "b", "c"),
      ArraySeq("a", null, "c"),
      ArraySeq('a', 'b'),
      ArraySeq(1.byteValue(), 2.byteValue),
      ArraySeq(1.shortValue, 2.shortValue),
      ArraySeq(1, 2, 3),
      ArraySeq(1L, 2L, 3L),
      ArraySeq(true, false),
    )

  @ParameterizedTest("ArraySeqSerialization")
  def testArraySeqSerialization(xs: ArraySeq[_]): Unit = {
    val output = new Output(1024, -1)
    kryo.writeClassAndObject(output, xs)
    val ys = kryo.readClassAndObject(new Input(output.toBytes))
    assert((xs == ys) && (xs == null || xs.getClass == ys.getClass))
  }
}
