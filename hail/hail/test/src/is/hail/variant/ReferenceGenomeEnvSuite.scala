package is.hail.variant

import is.hail.TestUtils._

import scala.collection.immutable.ArraySeq

import org.json4s.jackson.JsonMethods
import org.junit.jupiter.api.Assertions.assertTrue
import org.junit.jupiter.api.Test

class ReferenceGenomeEnvSuite {

  private val grch38 = ReferenceGenome.builtinReferences()(ReferenceGenome.GRCh38)

  private val collision = "a different reference with that name already exists"

  private def mk(name: String, lengths: (String, Int)*): ReferenceGenome =
    ReferenceGenome(name, ArraySeq.from(lengths.map(_._1)), lengths.toMap)

  // Equal to the cached GRCh38, but a different object.
  private def grch38Copy: ReferenceGenome =
    ReferenceGenome.fromJSON(JsonMethods.compact(grch38.toJSON))

  @Test def testAddingABuiltinIsANoOp(): Unit =
    assertEq(ReferenceGenomeEnv.empty.withReferences(Seq(grch38)), ReferenceGenomeEnv.empty)

  @Test def testAddingACopyOfABuiltinIsANoOp(): Unit = {
    val copy = grch38Copy
    assertTrue(copy ne grch38)
    assertEq(copy, grch38)

    val env = ReferenceGenomeEnv.empty.withReferences(Seq(copy))
    assertEq(env, ReferenceGenomeEnv.empty)
    assertTrue(env.resolve(ReferenceGenome.GRCh38) eq grch38)
  }

  @Test def testAddingADifferentGenomeUnderABuiltinNameFails(): Unit =
    interceptFatal(collision) {
      ReferenceGenomeEnv.empty.withReferences(Seq(mk(ReferenceGenome.GRCh38, "1" -> 10)))
    }

  @Test def testAddingADifferentGenomeUnderAUserDefinedNameFails(): Unit = {
    val env = ReferenceGenomeEnv.empty.withReferences(Seq(mk("foo", "1" -> 10)))
    interceptFatal(collision)(env.withReferences(Seq(mk("foo", "1" -> 11))))
  }

  @Test def testAddingAnEqualGenomeKeepsTheExistingOne(): Unit = {
    val existing = mk("foo", "1" -> 10)
    val equal = mk("foo", "1" -> 10)
    assertTrue(equal ne existing)

    val env = ReferenceGenomeEnv.empty.withReferences(Seq(existing)).withReferences(Seq(equal))
    assertTrue(env.resolve("foo") eq existing)
  }

  @Test def testWithoutRemovesAUserDefinedGenome(): Unit = {
    val env = ReferenceGenomeEnv.empty.withReferences(Seq(mk("foo", "1" -> 10)))
    assertEq(env.without("foo"), ReferenceGenomeEnv.empty)
    interceptFatal("not in scope")(env.without("foo").resolve("foo"))
  }

  @Test def testWithoutABuiltinFails(): Unit =
    interceptFatal("Cannot remove builtin reference genome 'GRCh38'") {
      ReferenceGenomeEnv.empty.without(ReferenceGenome.GRCh38)
    }

  @Test def testConstructingNeverStoresABuiltin(): Unit = {
    assertEq(ReferenceGenomeEnv(Seq(grch38, grch38Copy)), ReferenceGenomeEnv.empty)
    interceptFatal(collision)(ReferenceGenomeEnv(Seq(mk(ReferenceGenome.GRCh38, "1" -> 10))))
  }
}
