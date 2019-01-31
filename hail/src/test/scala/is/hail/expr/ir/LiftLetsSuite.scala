package is.hail.expr.ir

import is.hail.TestUtils._
import is.hail.expr.types.virtual._
import is.hail.utils._
import org.scalatest.testng.TestNGSuite
import org.testng.annotations.{DataProvider, Test}

class LiftLetsSuite extends TestNGSuite {
  @DataProvider(name = "nonLiftingOps")
  def nonLiftingOps(): Array[Array[IR]] = {
    val a = ArrayRange(I32(0), I32(10), I32(1))
    val x = Ref("x", TInt32())
    val y = Ref("y", TInt32())

    def let(t: Type, name: String = "x") = Let(name, ApplyBinaryPrimOp(Add(), I32(1), y), NA(t))

    Array(
      ArrayMap(a, "y", let(TInt32())),
      ArrayFilter(a, "y", let(TBoolean())),
      ArrayFlatMap(a, "y", let(TArray(TInt32()))),
      ArrayFold(a, I32(0), "acc", "y", let(TInt32())),
      ArrayFold(a, I32(0), "acc", "y", let(TInt32(), "acc")),
      ArrayScan(a, I32(0), "acc", "y", let(TInt32())),
      ArrayScan(a, I32(0), "acc", "y", let(TInt32(), "acc"))
    ).map(ir => Array[IR](ir))
  }

  @DataProvider(name = "liftingOps")
  def liftingOps(): Array[Array[IR]] = {
    val x = Ref("x", TInt32())
    val l = Let("x", I32(1), ApplyBinaryPrimOp(Add(), x, x))
    Array(
      MakeStruct(FastSeq("a" -> l)),
      MakeTuple(FastSeq(l)),
      ApplyBinaryPrimOp(Add(), l, I32(2)),
      ApplyUnaryPrimOp(Negate(), l),
      If(True(), l, NA(TInt32()))).map(ir => Array[IR](ir))
  }

  @Test def assertDataProvidersWork(): Unit = {
    nonLiftingOps()
    liftingOps()
  }

  @Test(dataProvider = "nonLiftingOps")
  def testNonLiftingOps(ir: IR): Unit = {
    val after = LiftLets(ir)
    assert(!after.isInstanceOf[Let])
    TypeCheck(ir)
  }

  @Test(dataProvider = "liftingOps")
  def testLiftingOps(ir: IR): Unit = {
    val after = LiftLets(ir)
    assert(after.isInstanceOf[Let])
    TypeCheck(ir)
  }

  @Test def testEquivalentLets(): Unit = {
    val ir = Let(
      "x1",
      I32(1),
      Let(
        "toElide",
        I32(1),
        ApplyUnaryPrimOp(Negate(), Ref("toElide", TInt32()))
      )
    )

    assert(LiftLets(ir) == Let("x1", I32(1), ApplyUnaryPrimOp(Negate(), Ref("x1", TInt32()))))
  }

  @Test def testNestedLetsBody(): Unit = {
    val ir = Let("bar", ApplyBinaryPrimOp(Add(), In(1, TInt32()), I32(1)),
      Let("baz", ApplyBinaryPrimOp(Add(), Ref("bar", TInt32()), I32(1)),
        ApplyBinaryPrimOp(Add(), I32(1), Ref("baz", TInt32()))))

    assert(LiftLets(ir) == ir)
  }

  @Test def testNestedLetsMixed() {
    val ir = Let(
      "x1",
      I32(1),
      Let("x2",
        Ref("x1", TInt32()),
        Let("x3",
          Let("x4", Ref("x1", TInt32()), Ref("x4", TInt32())),
          MakeArray(Array("x1", "x2", "x3").map(Ref(_, TInt32())), TArray(TInt32())))))
    TypeCheck(ir, Env.empty[Type], None)
    TypeCheck(LiftLets(ir).asInstanceOf[IR], Env.empty[Type], None)
  }

  @Test def testNestedLetsValue(): Unit = {
    val ir = Let(
      "bar",
      Let(
        "baz",
        ApplyBinaryPrimOp(Add(), In(1, TInt32()), I32(1)),
        ApplyBinaryPrimOp(Add(), I32(1), Ref("baz", TInt32()))),
      Let("qux",
        ApplyBinaryPrimOp(Add(), Ref("bar", TInt32()), I32(1)),
        Ref("qux", TInt32())
      ))
    TypeCheck(LiftLets(ir).asInstanceOf[IR], Env.empty[Type], None)

    val (_, lbs) = LiftLets.lift(ir)

    def traverseRefs(ir: IR, env: Set[String]): Unit = {
      ir match {
        case Ref(name, _) => if (!env.contains(name)) fatal(s"invalid ref $name")
        case _ => ir.children.foreach(c => traverseRefs(c.asInstanceOf[IR], env))
      }
    }

    def checkBindings(lbs: List[LetBinding], env: Set[String]): Unit = {
      lbs.foreach { lb =>
        traverseRefs(lb.value, env)
        checkBindings(lb.valueBindings, env)
        checkBindings(lb.bodyBindings, env + lb.name)
      }
    }
  }

  @Test def testDeeplyNested() {
    val refs = (0 until 10).map(i => Ref(s"x$i", TInt32()))
    val ir = refs.zipWithIndex.foldRight[IR](Let("foo", MakeArray(refs, TArray(TInt32())), Ref("foo", TArray(TInt32())))) { case ((ref, i), ir) => Let(ref.name, I32(i), ir) }
    TypeCheck(ir, Env.empty[Type], None)
    TypeCheck(LiftLets(ir).asInstanceOf[IR], Env.empty[Type], None)
  }

  @Test def testLetsInsideArrayMap(): Unit = {
    val ir = ArrayMap(
      In(0, TArray(TInt32())),
      "foo",
      Let("bar", ApplyBinaryPrimOp(Add(), In(1, TInt32()), I32(1)),
        Let("baz", ApplyBinaryPrimOp(Add(), Ref("bar", TInt32()), I32(1)),
          ApplyBinaryPrimOp(Add(), Ref("foo", TInt32()), Ref("baz", TInt32())))))

    assert(LiftLets(ir) == Let("bar", ApplyBinaryPrimOp(Add(), In(1, TInt32()), I32(1)),
      Let("baz", ApplyBinaryPrimOp(Add(), Ref("bar", TInt32()), I32(1)),
        ArrayMap(
          In(0, TArray(TInt32())),
          "foo",
          ApplyBinaryPrimOp(Add(), Ref("foo", TInt32()), Ref("baz", TInt32()))))))
  }
}
