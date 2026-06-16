package is.hail.expr.ir

import is.hail.{ExecStrategy, ParameterizedTest}
import is.hail.ExecStrategy.ExecStrategy
import is.hail.TestUtils._
import is.hail.backend.ExecuteContext
import is.hail.collection.FastSeq
import is.hail.collection.compat.immutable.ArraySeq
import is.hail.expr.ir.TestUtils._
import is.hail.expr.ir.defs._
import is.hail.types.virtual._

import org.junit.jupiter.api.Test

class ArrayFunctionsSuite {
  val naa = NA(TArray(TInt32))

  implicit val execStrats: Set[ExecStrategy] = ExecStrategy.javaOnly

  def basicData() = ArraySeq[IndexedSeq[Integer]](
    FastSeq(3, 7),
    null,
    FastSeq(3, null, 7, null),
    FastSeq(),
  )

  def extend() =
    basicData().combinations(2).map { case Seq(a, b) => (a, b) }.to(ArraySeq)

  @ParameterizedTest("basicData")
  def isEmpty(a: IndexedSeq[Integer])(implicit ctx: ExecuteContext): Unit =
    assertEvalsTo(invoke("isEmpty", TBoolean, toIRArray(a)), Option(a).map(_.isEmpty).orNull)

  @ParameterizedTest("basicData")
  def append(a: IndexedSeq[Integer])(implicit ctx: ExecuteContext): Unit =
    assertEvalsTo(
      invoke("append", TArray(TInt32), toIRArray(a), I32(1)),
      Option(a).map(_ :+ 1).orNull,
    )

  @ParameterizedTest("basicData")
  def appendNull(a: IndexedSeq[Integer])(implicit ctx: ExecuteContext): Unit =
    assertEvalsTo(
      invoke("append", TArray(TInt32), toIRArray(a), NA(TInt32)),
      Option(a).map(_ :+ null).orNull,
    )

  @ParameterizedTest("basicData")
  def sum(a: IndexedSeq[Integer])(implicit ctx: ExecuteContext): Unit = {
    assertEvalsTo(
      invoke("sum", TInt32, toIRArray(a)),
      Option(a).flatMap(_.foldLeft[Option[Int]](Some(0))((comb, x) =>
        comb.flatMap(c => Option(x).map(_ + c))
      )).orNull,
    )
  }

  @ParameterizedTest("basicData")
  def product(a: IndexedSeq[Integer])(implicit ctx: ExecuteContext): Unit = {
    assertEvalsTo(
      invoke("product", TInt32, toIRArray(a)),
      Option(a).flatMap(_.foldLeft[Option[Int]](Some(1))((comb, x) =>
        comb.flatMap(c => Option(x).map(_ * c))
      )).orNull,
    )
  }

  @Test def mean(implicit ctx: ExecuteContext): Unit = {
    assertEvalsTo(invoke("mean", TFloat64, IRArray(3, 7)), 5.0)
    assertEvalsTo(invoke("mean", TFloat64, IRArray(3, null, 7)), null)
    assertEvalsTo(invoke("mean", TFloat64, IRArray(3, 7, 11)), 7.0)
    assertEvalsTo(invoke("mean", TFloat64, IRArray()), Double.NaN)
    assertEvalsTo(invoke("mean", TFloat64, IRArray(null)), null)
    assertEvalsTo(invoke("mean", TFloat64, naa), null)
  }

  @Test def median(implicit ctx: ExecuteContext): Unit = {
    assertEvalsTo(invoke("median", TInt32, IRArray(5)), 5)
    assertEvalsTo(invoke("median", TInt32, IRArray(5, null, null)), 5)
    assertEvalsTo(invoke("median", TInt32, IRArray(3, 7)), 5)
    assertEvalsTo(invoke("median", TInt32, IRArray(3, null, 7, 1, null)), 3)
    assertEvalsTo(invoke("median", TInt32, IRArray(3, 7, 1)), 3)
    assertEvalsTo(invoke("median", TInt32, IRArray(3, null, 9, 6, 1, null)), 4)
    assertEvalsTo(invoke("median", TInt32, IRArray()), null)
    assertEvalsTo(invoke("median", TInt32, IRArray(null)), null)
    assertEvalsTo(invoke("median", TInt32, naa), null)
  }

  @ParameterizedTest
  def extend(a: IndexedSeq[Integer], b: IndexedSeq[Integer])(implicit ctx: ExecuteContext): Unit =
    assertEvalsTo(
      invoke("extend", TArray(TInt32), toIRArray(a), toIRArray(b)),
      Option(a).zip(Option(b)).headOption.map { case (x, y) => x ++ y }.orNull,
    )

  def sortData() = ArraySeq[(ArraySeq[Integer], ArraySeq[Integer], ArraySeq[Integer])](
    (ArraySeq(3, 9, 7), ArraySeq(3, 7, 9), ArraySeq(9, 7, 3)),
    (null, null, null),
    (ArraySeq(3, null, 1, null, 3), ArraySeq(1, 3, 3, null, null), ArraySeq(3, 3, 1, null, null)),
    (ArraySeq(1, null, 3, null, 1), ArraySeq(1, 1, 3, null, null), ArraySeq(3, 1, 1, null, null)),
    (ArraySeq(), ArraySeq(), ArraySeq()),
  )

  @ParameterizedTest("sortData")
  def min(
    a: IndexedSeq[Integer],
    asc: IndexedSeq[Integer],
    desc: IndexedSeq[Integer],
  )(implicit ctx: ExecuteContext
  ): Unit =
    assertEvalsTo(
      invoke("min", TInt32, toIRArray(a)),
      Option(asc).filter(!_.contains(null)).flatMap(_.headOption).orNull,
    )

  @Test def testMinMaxNans(implicit ctx: ExecuteContext): Unit = {
    assertAllEvalTo(
      (
        invoke(
          "min",
          TFloat32,
          MakeArray(
            FastSeq(F32(Float.NaN), F32(1.0f), F32(Float.NaN), F32(111.0f)),
            TArray(TFloat32),
          ),
        ),
        Float.NaN,
      ),
      (
        invoke(
          "max",
          TFloat32,
          MakeArray(
            FastSeq(F32(Float.NaN), F32(1.0f), F32(Float.NaN), F32(111.0f)),
            TArray(TFloat32),
          ),
        ),
        Float.NaN,
      ),
      (
        invoke(
          "min",
          TFloat64,
          MakeArray(
            FastSeq(F64(Double.NaN), F64(1.0), F64(Double.NaN), F64(111.0)),
            TArray(TFloat64),
          ),
        ),
        Double.NaN,
      ),
      (
        invoke(
          "max",
          TFloat64,
          MakeArray(
            FastSeq(F64(Double.NaN), F64(1.0), F64(Double.NaN), F64(111.0)),
            TArray(TFloat64),
          ),
        ),
        Double.NaN,
      ),
    )
  }

  @ParameterizedTest("sortData")
  def max(
    a: IndexedSeq[Integer],
    asc: IndexedSeq[Integer],
    desc: IndexedSeq[Integer],
  )(implicit ctx: ExecuteContext
  ): Unit =
    assertEvalsTo(
      invoke("max", TInt32, toIRArray(a)),
      Option(desc).filter(!_.contains(null)).flatMap(_.headOption).orNull,
    )

  def argMinMaxData() = ArraySeq[(ArraySeq[Integer], Integer, Integer)](
    (ArraySeq(3, 9, 7), 0, 1),
    (null, null, null),
    (ArraySeq(3, null, 1, null, 3), 2, 0),
    (ArraySeq(1, null, 3, null, 1), 0, 2),
    (ArraySeq(), null, null),
  )

  @ParameterizedTest("argMinMaxData")
  def argmin(a: IndexedSeq[Integer], argmin: Integer, argmax: Integer)(implicit ctx: ExecuteContext)
    : Unit =
    assertEvalsTo(invoke("argmin", TInt32, toIRArray(a)), argmin)

  @ParameterizedTest("argMinMaxData")
  def argmax(a: IndexedSeq[Integer], argmin: Integer, argmax: Integer)(implicit ctx: ExecuteContext)
    : Unit =
    assertEvalsTo(invoke("argmax", TInt32, toIRArray(a)), argmax)

  def uniqueMinMaxData() = ArraySeq[(ArraySeq[Integer], Integer, Integer)](
    (ArraySeq(3, 9, 7), 0, 1),
    (null, null, null),
    (ArraySeq(3, null, 1, null, 3), 2, null),
    (ArraySeq(1, null, 3, null, 1), null, 2),
    (ArraySeq(), null, null),
  )

  @ParameterizedTest("uniqueMinMaxData")
  def uniqueMinIndex(
    a: IndexedSeq[Integer],
    argmin: Integer,
    argmax: Integer,
  )(implicit ctx: ExecuteContext
  ): Unit =
    assertEvalsTo(invoke("uniqueMinIndex", TInt32, toIRArray(a)), argmin)

  @ParameterizedTest("uniqueMinMaxData")
  def uniqueMaxIndex(
    a: IndexedSeq[Integer],
    argmin: Integer,
    argmax: Integer,
  )(implicit ctx: ExecuteContext
  ): Unit =
    assertEvalsTo(invoke("uniqueMaxIndex", TInt32, toIRArray(a)), argmax)

  def arrayOpsData() =
    ArraySeq[IndexedSeq[Integer]](
      FastSeq(3, 9, 7, 1),
      FastSeq(null, 2, null, 8),
      FastSeq(5, 3, null, null),
      null,
    ).combinations(2).map { case Seq(a, b) => (a, b) }.to(ArraySeq)

  def arrayOpsOperations() = ArraySeq[(String, (Int, Int) => Int)](
    ("add", _ + _),
    ("sub", _ - _),
    ("mul", _ * _),
    ("floordiv", _ / _),
    ("mod", _ % _),
  )

  def arrayOps() =
    for {
      (a, b) <- arrayOpsData()
      (s, f) <- arrayOpsOperations()
    } yield (a, b, s, f)

  def lift(f: (Int, Int) => Int)
    : (IndexedSeq[Integer], IndexedSeq[Integer]) => IndexedSeq[Integer] = {
    case (a, b) =>
      Option(a).zip(Option(b)).headOption.map { case (a0, b0) =>
        a0.zip(b0).map { case (i, j) =>
          Option(i).zip(Option(j)).headOption.map[Integer] { case (m, n) => f(m, n) }.orNull
        }
      }.orNull
  }

  @ParameterizedTest
  def arrayOps(
    a: IndexedSeq[Integer],
    b: IndexedSeq[Integer],
    s: String,
    f: (Int, Int) => Int,
  )(implicit ctx: ExecuteContext
  ): Unit =
    assertEvalsTo(invoke(s, TArray(TInt32), toIRArray(a), toIRArray(b)), lift(f)(a, b))

  @ParameterizedTest("arrayOpsData")
  def arrayOpsFPDiv(a: IndexedSeq[Integer], b: IndexedSeq[Integer])(implicit ctx: ExecuteContext)
    : Unit = {
    assertEvalsTo(
      invoke("div", TArray(TFloat64), toIRArray(a), toIRArray(b)),
      Option(a).zip(Option(b)).headOption.map { case (a0, b0) =>
        a0.zip(b0).map { case (i, j) =>
          Option(i).zip(Option(j)).headOption.map[java.lang.Double] { case (m, n) =>
            m.toDouble / n
          }.orNull
        }
      }.orNull,
    )
  }

  @ParameterizedTest("arrayOpsData")
  def arrayOpsPow(a: IndexedSeq[Integer], b: IndexedSeq[Integer])(implicit ctx: ExecuteContext)
    : Unit = {
    assertEvalsTo(
      invoke("pow", TArray(TFloat64), toIRArray(a), toIRArray(b)),
      Option(a).zip(Option(b)).headOption.map { case (a0, b0) =>
        a0.zip(b0).map { case (i, j) =>
          Option(i).zip(Option(j)).headOption.map[java.lang.Double] { case (m, n) =>
            math.pow(m.toDouble, n.toDouble)
          }.orNull
        }
      }.orNull,
    )
  }

  @ParameterizedTest("arrayOpsOperations")
  def arrayOpsDifferentLength(s: String, f: (Int, Int) => Int)(implicit ctx: ExecuteContext)
    : Unit = {
    assertFatal(invoke(s, TArray(TInt32), IRArray(1, 2, 3), IRArray(1, 2)), "length mismatch")
    assertFatal(invoke(s, TArray(TInt32), IRArray(1, 2), IRArray(1, 2, 3)), "length mismatch")
  }

  @Test def indexing(implicit ctx: ExecuteContext): Unit = {
    val a = IRArray(0, null, 2)
    assertEvalsTo(invoke("indexArray", TInt32, a, I32(0)), 0)
    assertEvalsTo(invoke("indexArray", TInt32, a, I32(1)), null)
    assertEvalsTo(invoke("indexArray", TInt32, a, I32(2)), 2)
    assertEvalsTo(invoke("indexArray", TInt32, a, I32(-1)), 2)
    assertEvalsTo(invoke("indexArray", TInt32, a, I32(-3)), 0)
    assertFatal(invoke("indexArray", TInt32, a, I32(3)), "array index out of bounds")
    assertFatal(invoke("indexArray", TInt32, a, I32(-4)), "array index out of bounds")
    assertEvalsTo(invoke("indexArray", TInt32, naa, I32(2)), null)
    assertEvalsTo(invoke("indexArray", TInt32, a, NA(TInt32)), null)
  }

  @Test def slicing(implicit ctx: ExecuteContext): Unit = {
    val a = IRArray(0, null, 2)
    assertEvalsTo(ArraySlice(a, I32(1), None), FastSeq(null, 2))
    assertEvalsTo(ArraySlice(a, I32(-2), None), FastSeq(null, 2))
    assertEvalsTo(ArraySlice(a, I32(5), None), FastSeq())
    assertEvalsTo(ArraySlice(a, I32(-5), None), FastSeq(0, null, 2))
    assertEvalsTo(ArraySlice(naa, I32(1), None), null)
    assertEvalsTo(ArraySlice(a, NA(TInt32), None), null)

    assertEvalsTo(ArraySlice(a, I32(0), Some(I32(2))), FastSeq(0, null))
    assertEvalsTo(ArraySlice(a, I32(0), Some(I32(-1))), FastSeq(0, null))
    assertEvalsTo(ArraySlice(a, I32(0), Some(I32(5))), FastSeq(0, null, 2))
    assertEvalsTo(ArraySlice(a, I32(0), Some(I32(-5))), FastSeq())
    assertEvalsTo(ArraySlice(naa, I32(0), Some(I32(1))), null)
    assertEvalsTo(ArraySlice(a, I32(0), Some(NA(TInt32))), null)

    assertEvalsTo(ArraySlice(a, I32(1), Some(I32(3))), FastSeq(null, 2))
    assertEvalsTo(ArraySlice(a, I32(1), Some(I32(2))), FastSeq(null))
    assertEvalsTo(ArraySlice(a, I32(0), Some(I32(2))), FastSeq(0, null))
    assertEvalsTo(ArraySlice(a, I32(0), Some(I32(3))), FastSeq(0, null, 2))
    assertEvalsTo(ArraySlice(a, I32(-1), Some(I32(3))), FastSeq(2))
    assertEvalsTo(ArraySlice(a, I32(-4), Some(I32(4))), FastSeq(0, null, 2))
    assertEvalsTo(ArraySlice(naa, I32(1), Some(I32(2))), null)
    assertEvalsTo(ArraySlice(a, I32(1), Some(NA(TInt32))), null)
    assertEvalsTo(ArraySlice(a, NA(TInt32), Some(I32(1))), null)
    assertEvalsTo(ArraySlice(a, I32(3), Some(I32(2))), FastSeq())
  }

  def flatten() = ArraySeq(
    (FastSeq(FastSeq(3, 9, 7), FastSeq(3, 7, 9)), FastSeq(3, 9, 7, 3, 7, 9)),
    (FastSeq(null, FastSeq(1)), FastSeq(1)),
    (FastSeq(null, null), FastSeq()),
    (FastSeq(FastSeq(null), FastSeq(), FastSeq(7)), FastSeq(null, 7)),
    (FastSeq(FastSeq(), FastSeq()), FastSeq()),
  )

  @ParameterizedTest
  def flatten(
    in: IndexedSeq[IndexedSeq[Integer]],
    expected: IndexedSeq[Int],
  )(implicit ctx: ExecuteContext
  ): Unit =
    assertEvalsTo(
      invoke("flatten", TArray(TInt32), MakeArray(in.map(toIRArray(_)), TArray(TArray(TInt32)))),
      expected,
    )

  @Test def testContains(implicit ctx: ExecuteContext): Unit = {
    val t = TArray(TString)

    assertEvalsTo(
      invoke("contains", TBoolean, In(0, t), Str("a")),
      args = FastSeq(FastSeq() -> t),
      expected = false,
    )

    assertEvalsTo(
      invoke("contains", TBoolean, In(0, t), Str("a")),
      args = FastSeq(FastSeq(null) -> t),
      expected = false,
    )

    assertEvalsTo(
      invoke("contains", TBoolean, In(0, t), Str("a")),
      args = FastSeq(FastSeq("c", "a", "b") -> t),
      expected = true,
    )

    assertEvalsTo(
      invoke("contains", TBoolean, In(0, t), Str("a")),
      args = FastSeq(FastSeq("c", "a", "b", null) -> t),
      expected = true,
    )

    assertEvalsTo(
      invoke("contains", TBoolean, In(0, t), Str("a")),
      args = FastSeq((null, t)),
      expected = null,
    )

    assertEvalsTo(
      invoke("contains", TBoolean, In(0, t), NA(t.elementType)),
      args = FastSeq((null, t)),
      expected = null,
    )

    assertEvalsTo(
      invoke("contains", TBoolean, In(0, t), NA(t.elementType)),
      args = FastSeq(FastSeq("a", null) -> t),
      expected = true,
    )
  }

  def testScatter() = ArraySeq(
    (FastSeq("a", "b", "c"), FastSeq(1, 3, 4), FastSeq(null, "a", null, "b", "c")),
    (FastSeq("a", "b", "c"), FastSeq(2, 0, 3), FastSeq("b", null, "a", "c", null)),
    (FastSeq(), FastSeq(), FastSeq(null, null, null)),
    (FastSeq(), FastSeq(), FastSeq()),
  )

  @ParameterizedTest
  def testScatter(
    elts: IndexedSeq[String],
    indices: IndexedSeq[Int],
    expected: IndexedSeq[String],
  )(implicit ctx: ExecuteContext
  ): Unit = {
    val t1 = TArray(TInt32)
    val t2 = TArray(TString)

    assertEvalsTo(
      invoke("scatter", t2, FastSeq(TString), In(0, t2), In(1, t1), expected.length),
      args = FastSeq(elts -> t2, indices -> t1),
      expected = expected,
    )
  }

  def testScatterErrors() = ArraySeq(
    (FastSeq("a", "b", "c"), FastSeq(1, 3, 4), 4, "indices array contained index 4"),
    (
      FastSeq("a", "b"),
      FastSeq(1, 3, 4),
      4,
      "values and indices arrays have different lengths",
    ),
    (FastSeq("a", "b", "c"), FastSeq(1, 2, 2), 2, "values array is larger than result length"),
  )

  @ParameterizedTest
  def testScatterErrors(
    elts: IndexedSeq[String],
    indices: IndexedSeq[Int],
    length: Int,
    regex: String,
  )(implicit ctx: ExecuteContext
  ): Unit = {
    val t1 = TArray(TInt32)
    val t2 = TArray(TString)

    assertFatal(
      invoke("scatter", t2, FastSeq(TString), In(0, t2), In(1, t1), length),
      args = FastSeq(elts -> t2, indices -> t1),
      regex = regex,
    )
  }
}
