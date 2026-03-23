package is.hail.expr.ir

import is.hail.{ExecStrategy, HailSuite}
import is.hail.ExecStrategy.ExecStrategy
import is.hail.collection.FastSeq
import is.hail.expr.ir.TestUtils._
import is.hail.expr.ir.defs.{ArraySlice, F32, F64, I32, In, MakeArray, NA, Str}
import is.hail.types.virtual._

class ArrayFunctionsSuite extends HailSuite {
  val naa = NA(TArray(TInt32))

  implicit val execStrats: Set[ExecStrategy] = ExecStrategy.javaOnly

  def lift(f: (Int, Int) => Int)
    : (IndexedSeq[Integer], IndexedSeq[Integer]) => IndexedSeq[Integer] = {
    case (a, b) =>
      Option(a).zip(Option(b)).headOption.map { case (a0, b0) =>
        a0.zip(b0).map { case (i, j) =>
          Option(i).zip(Option(j)).headOption.map[Integer] { case (m, n) => f(m, n) }.orNull
        }
      }.orNull
  }

  // basic DataProvider tests

  val basicData: Array[IndexedSeq[Integer]] =
    Array(FastSeq(3, 7), null, FastSeq(3, null, 7, null), FastSeq())

  object checkIsEmpty extends TestCases {
    def apply(a: IndexedSeq[Integer])(implicit loc: munit.Location): Unit = test("isEmpty") {
      assertEvalsTo(invoke("isEmpty", TBoolean, toIRArray(a)), Option(a).map(_.isEmpty).orNull)
    }
  }

  basicData.foreach(checkIsEmpty(_))

  object checkAppend extends TestCases {
    def apply(a: IndexedSeq[Integer])(implicit loc: munit.Location): Unit = test("append") {
      assertEvalsTo(
        invoke("append", TArray(TInt32), toIRArray(a), I32(1)),
        Option(a).map(_ :+ 1).orNull,
      )
    }
  }

  basicData.foreach(checkAppend(_))

  object checkAppendNull extends TestCases {
    def apply(a: IndexedSeq[Integer])(implicit loc: munit.Location): Unit = test("appendNull") {
      assertEvalsTo(
        invoke("append", TArray(TInt32), toIRArray(a), NA(TInt32)),
        Option(a).map(_ :+ null).orNull,
      )
    }
  }

  basicData.foreach(checkAppendNull(_))

  object checkSum extends TestCases {
    def apply(a: IndexedSeq[Integer])(implicit loc: munit.Location): Unit = test("sum") {
      assertEvalsTo(
        invoke("sum", TInt32, toIRArray(a)),
        Option(a).flatMap(_.foldLeft[Option[Int]](Some(0))((comb, x) =>
          comb.flatMap(c => Option(x).map(_ + c))
        )).orNull,
      )
    }
  }

  basicData.foreach(checkSum(_))

  object checkProduct extends TestCases {
    def apply(a: IndexedSeq[Integer])(implicit loc: munit.Location): Unit = test("product") {
      assertEvalsTo(
        invoke("product", TInt32, toIRArray(a)),
        Option(a).flatMap(_.foldLeft[Option[Int]](Some(1))((comb, x) =>
          comb.flatMap(c => Option(x).map(_ * c))
        )).orNull,
      )
    }
  }

  basicData.foreach(checkProduct(_))

  test("mean") {
    assertEvalsTo(invoke("mean", TFloat64, IRArray(3, 7)), 5.0)
    assertEvalsTo(invoke("mean", TFloat64, IRArray(3, null, 7)), null)
    assertEvalsTo(invoke("mean", TFloat64, IRArray(3, 7, 11)), 7.0)
    assertEvalsTo(invoke("mean", TFloat64, IRArray()), Double.NaN)
    assertEvalsTo(invoke("mean", TFloat64, IRArray(null)), null)
    assertEvalsTo(invoke("mean", TFloat64, naa), null)
  }

  test("median") {
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

  // basicPairs DataProvider test

  object checkExtend extends TestCases {
    def apply(a: IndexedSeq[Integer], b: IndexedSeq[Integer])(implicit loc: munit.Location): Unit =
      test("extend") {
        assertEvalsTo(
          invoke("extend", TArray(TInt32), toIRArray(a), toIRArray(b)),
          Option(a).zip(Option(b)).headOption.map { case (x, y) => x ++ y }.orNull,
        )
      }
  }

  basicData.combinations(2).foreach { case Array(a, b) => checkExtend(a, b) }

  // sort DataProvider tests

  object checkMin extends TestCases {
    def apply(
      a: IndexedSeq[Integer],
      asc: IndexedSeq[Integer],
      desc: IndexedSeq[Integer],
    )(implicit
      loc: munit.Location
    ): Unit = test("min") {
      assertEvalsTo(
        invoke("min", TInt32, toIRArray(a)),
        Option(asc).filter(!_.contains(null)).flatMap(_.headOption).orNull,
      )
    }
  }

  object checkMax extends TestCases {
    def apply(
      a: IndexedSeq[Integer],
      asc: IndexedSeq[Integer],
      desc: IndexedSeq[Integer],
    )(implicit
      loc: munit.Location
    ): Unit = test("max") {
      assertEvalsTo(
        invoke("max", TInt32, toIRArray(a)),
        Option(desc).filter(!_.contains(null)).flatMap(_.headOption).orNull,
      )
    }
  }

  {
    val sortData: Array[(IndexedSeq[Integer], IndexedSeq[Integer], IndexedSeq[Integer])] = Array(
      (FastSeq(3, 9, 7), FastSeq(3, 7, 9), FastSeq(9, 7, 3)),
      (null, null, null),
      (FastSeq(3, null, 1, null, 3), FastSeq(1, 3, 3, null, null), FastSeq(3, 3, 1, null, null)),
      (FastSeq(1, null, 3, null, 1), FastSeq(1, 1, 3, null, null), FastSeq(3, 1, 1, null, null)),
      (FastSeq(), FastSeq(), FastSeq()),
    )
    for ((a, asc, desc) <- sortData) {
      checkMin(a, asc, desc)
      checkMax(a, asc, desc)
    }
  }

  test("MinMaxNans") {
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

  // argminmax DataProvider tests

  object checkArgmin extends TestCases {
    def apply(
      a: IndexedSeq[Integer],
      argmin: Integer,
      argmax: Integer,
    )(implicit
      loc: munit.Location
    ): Unit = test("argmin") {
      assertEvalsTo(invoke("argmin", TInt32, toIRArray(a)), argmin)
    }
  }

  object checkArgmax extends TestCases {
    def apply(
      a: IndexedSeq[Integer],
      argmin: Integer,
      argmax: Integer,
    )(implicit
      loc: munit.Location
    ): Unit = test("argmax") {
      assertEvalsTo(invoke("argmax", TInt32, toIRArray(a)), argmax)
    }
  }

  {
    val argMinMaxData: Array[(IndexedSeq[Integer], Integer, Integer)] = Array(
      (FastSeq(3, 9, 7), 0, 1),
      (null, null, null),
      (FastSeq(3, null, 1, null, 3), 2, 0),
      (FastSeq(1, null, 3, null, 1), 0, 2),
      (FastSeq(), null, null),
    )
    for ((a, amin, amax) <- argMinMaxData) {
      checkArgmin(a, amin, amax)
      checkArgmax(a, amin, amax)
    }
  }

  // uniqueMinMaxIndex DataProvider tests

  object checkUniqueMinIndex extends TestCases {
    def apply(
      a: IndexedSeq[Integer],
      argmin: Integer,
      argmax: Integer,
    )(implicit
      loc: munit.Location
    ): Unit = test("uniqueMinIndex") {
      assertEvalsTo(invoke("uniqueMinIndex", TInt32, toIRArray(a)), argmin)
    }
  }

  object checkUniqueMaxIndex extends TestCases {
    def apply(
      a: IndexedSeq[Integer],
      argmin: Integer,
      argmax: Integer,
    )(implicit
      loc: munit.Location
    ): Unit = test("uniqueMaxIndex") {
      assertEvalsTo(invoke("uniqueMaxIndex", TInt32, toIRArray(a)), argmax)
    }
  }

  {
    val uniqueMinMaxData: Array[(IndexedSeq[Integer], Integer, Integer)] = Array(
      (FastSeq(3, 9, 7), 0, 1),
      (null, null, null),
      (FastSeq(3, null, 1, null, 3), 2, null),
      (FastSeq(1, null, 3, null, 1), null, 2),
      (FastSeq(), null, null),
    )
    for ((a, amin, amax) <- uniqueMinMaxData) {
      checkUniqueMinIndex(a, amin, amax)
      checkUniqueMaxIndex(a, amin, amax)
    }
  }

  // arrayOps DataProvider tests

  object checkArrayOps extends TestCases {
    def apply(
      a: IndexedSeq[Integer],
      b: IndexedSeq[Integer],
      s: String,
      f: (Int, Int) => Int,
    )(implicit loc: munit.Location
    ): Unit = test("arrayOps") {
      assertEvalsTo(invoke(s, TArray(TInt32), toIRArray(a), toIRArray(b)), lift(f)(a, b))
    }
  }

  object checkArrayOpsFPDiv extends TestCases {
    def apply(a: IndexedSeq[Integer], b: IndexedSeq[Integer])(implicit loc: munit.Location): Unit =
      test("arrayOpsFPDiv") {
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
  }

  object checkArrayOpsPow extends TestCases {
    def apply(a: IndexedSeq[Integer], b: IndexedSeq[Integer])(implicit loc: munit.Location): Unit =
      test("arrayOpsPow") {
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
  }

  object checkArrayOpsDifferentLength extends TestCases {
    def apply(s: String, f: (Int, Int) => Int)(implicit loc: munit.Location): Unit =
      test("arrayOpsDifferentLength") {
        assertFatal(invoke(s, TArray(TInt32), IRArray(1, 2, 3), IRArray(1, 2)), "length mismatch")
        assertFatal(invoke(s, TArray(TInt32), IRArray(1, 2), IRArray(1, 2, 3)), "length mismatch")
      }
  }

  {
    val arrayOpsDataValues: Array[Array[IndexedSeq[Integer]]] = Array[IndexedSeq[Integer]](
      FastSeq(3, 9, 7, 1),
      FastSeq(null, 2, null, 8),
      FastSeq(5, 3, null, null),
      null,
    ).combinations(2).toArray.map(_.toArray)

    val arrayOpsOps: Array[(String, (Int, Int) => Int)] = Array(
      ("add", _ + _),
      ("sub", _ - _),
      ("mul", _ * _),
      ("floordiv", _ / _),
      ("mod", _ % _),
    )

    for {
      Array(a, b) <- arrayOpsDataValues
      (s, f) <- arrayOpsOps
    } checkArrayOps(a, b, s, f)

    for (Array(a, b) <- arrayOpsDataValues) {
      checkArrayOpsFPDiv(a, b)
      checkArrayOpsPow(a, b)
    }

    for ((s, f) <- arrayOpsOps)
      checkArrayOpsDifferentLength(s, f)
  }

  test("indexing") {
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

  test("slicing") {
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

  // flatten DataProvider test

  object checkFlatten extends TestCases {
    def apply(
      in: IndexedSeq[IndexedSeq[Integer]],
      expected: IndexedSeq[Any],
    )(implicit
      loc: munit.Location
    ): Unit = test("flatten") {
      assertEvalsTo(
        invoke("flatten", TArray(TInt32), MakeArray(in.map(toIRArray(_)), TArray(TArray(TInt32)))),
        expected,
      )
    }
  }

  checkFlatten(FastSeq(FastSeq(3, 9, 7), FastSeq(3, 7, 9)), FastSeq(3, 9, 7, 3, 7, 9))
  checkFlatten(FastSeq(null, FastSeq(1)), FastSeq(1))
  checkFlatten(FastSeq(null, null), FastSeq())
  checkFlatten(FastSeq(FastSeq(null), FastSeq(), FastSeq(7)), FastSeq[Any](null, 7))
  checkFlatten(FastSeq(FastSeq(), FastSeq()), FastSeq())

  test("Contains") {
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

  // scatter DataProvider test

  object checkScatter extends TestCases {
    def apply(
      elts: IndexedSeq[String],
      indices: IndexedSeq[Int],
      expected: IndexedSeq[String],
    )(implicit loc: munit.Location
    ): Unit = test("Scatter") {
      val t1 = TArray(TInt32)
      val t2 = TArray(TString)

      assertEvalsTo(
        invoke("scatter", t2, FastSeq(TString), In(0, t2), In(1, t1), expected.length),
        args = FastSeq(elts -> t2, indices -> t1),
        expected = expected,
      )
    }
  }

  checkScatter(FastSeq("a", "b", "c"), FastSeq(1, 3, 4), FastSeq(null, "a", null, "b", "c"))
  checkScatter(FastSeq("a", "b", "c"), FastSeq(2, 0, 3), FastSeq("b", null, "a", "c", null))
  checkScatter(FastSeq(), FastSeq(), FastSeq(null, null, null))
  checkScatter(FastSeq(), FastSeq(), FastSeq())

  // scatter_errors DataProvider test

  object checkScatterErrors extends TestCases {
    def apply(
      elts: IndexedSeq[String],
      indices: IndexedSeq[Int],
      length: Int,
      regex: String,
    )(implicit loc: munit.Location
    ): Unit = test("ScatterErrors") {
      val t1 = TArray(TInt32)
      val t2 = TArray(TString)

      assertFatal(
        invoke("scatter", t2, FastSeq(TString), In(0, t2), In(1, t1), length),
        args = FastSeq(elts -> t2, indices -> t1),
        regex = regex,
      )
    }
  }

  checkScatterErrors(FastSeq("a", "b", "c"), FastSeq(1, 3, 4), 4, "indices array contained index 4")

  checkScatterErrors(
    FastSeq("a", "b"),
    FastSeq(1, 3, 4),
    4,
    "values and indices arrays have different lengths",
  )

  checkScatterErrors(
    FastSeq("a", "b", "c"),
    FastSeq(1, 2, 2),
    2,
    "values array is larger than result length",
  )
}
