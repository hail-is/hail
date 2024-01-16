package is.hail.expr.ir

import is.hail.{stats, ExecStrategy, HailSuite}
import is.hail.TestUtils._
import is.hail.types.virtual._
import is.hail.utils._

import org.apache.spark.sql.Row
import org.testng.annotations.{DataProvider, Test}

class MathFunctionsSuite extends HailSuite {
  hc
  implicit val execStrats = ExecStrategy.values

  val tfloat = TFloat64

  @Test def log2(): Unit = {
    assertEvalsTo(invoke("log2", TInt32, I32(2)), 1)
    assertEvalsTo(invoke("log2", TInt32, I32(32)), 5)
    assertEvalsTo(invoke("log2", TInt32, I32(33)), 5)
    assertEvalsTo(invoke("log2", TInt32, I32(63)), 5)
    assertEvalsTo(invoke("log2", TInt32, I32(64)), 6)
  }

  @Test def roundToNextPowerOf2(): Unit = {
    assertEvalsTo(invoke("roundToNextPowerOf2", TInt32, I32(2)), 2)
    assertEvalsTo(invoke("roundToNextPowerOf2", TInt32, I32(32)), 32)
    assertEvalsTo(invoke("roundToNextPowerOf2", TInt32, I32(33)), 64)
    assertEvalsTo(invoke("roundToNextPowerOf2", TInt32, I32(63)), 64)
    assertEvalsTo(invoke("roundToNextPowerOf2", TInt32, I32(64)), 64)
  }

  @Test def isnan(): Unit = {
    implicit val execStrats = ExecStrategy.javaOnly

    assertEvalsTo(invoke("isnan", TBoolean, F32(0)), false)
    assertEvalsTo(invoke("isnan", TBoolean, F32(Float.NaN)), true)

    assertEvalsTo(invoke("isnan", TBoolean, F64(0)), false)
    assertEvalsTo(invoke("isnan", TBoolean, F64(Double.NaN)), true)
  }

  @Test def is_finite(): Unit = {
    implicit val execStrats = ExecStrategy.javaOnly

    assertEvalsTo(invoke("is_finite", TBoolean, F32(0)), expected = true)
    assertEvalsTo(invoke("is_finite", TBoolean, F32(Float.MaxValue)), expected = true)
    assertEvalsTo(invoke("is_finite", TBoolean, F32(Float.NaN)), expected = false)
    assertEvalsTo(invoke("is_finite", TBoolean, F32(Float.PositiveInfinity)), expected = false)
    assertEvalsTo(invoke("is_finite", TBoolean, F32(Float.NegativeInfinity)), expected = false)

    assertEvalsTo(invoke("is_finite", TBoolean, F64(0)), expected = true)
    assertEvalsTo(invoke("is_finite", TBoolean, F64(Double.MaxValue)), expected = true)
    assertEvalsTo(invoke("is_finite", TBoolean, F64(Double.NaN)), expected = false)
    assertEvalsTo(invoke("is_finite", TBoolean, F64(Double.PositiveInfinity)), expected = false)
    assertEvalsTo(invoke("is_finite", TBoolean, F64(Double.NegativeInfinity)), expected = false)
  }

  @Test def is_infinite(): Unit = {
    implicit val execStrats = ExecStrategy.javaOnly

    assertEvalsTo(invoke("is_infinite", TBoolean, F32(0)), expected = false)
    assertEvalsTo(invoke("is_infinite", TBoolean, F32(Float.MaxValue)), expected = false)
    assertEvalsTo(invoke("is_infinite", TBoolean, F32(Float.NaN)), expected = false)
    assertEvalsTo(invoke("is_infinite", TBoolean, F32(Float.PositiveInfinity)), expected = true)
    assertEvalsTo(invoke("is_infinite", TBoolean, F32(Float.NegativeInfinity)), expected = true)

    assertEvalsTo(invoke("is_infinite", TBoolean, F64(0)), expected = false)
    assertEvalsTo(invoke("is_infinite", TBoolean, F64(Double.MaxValue)), expected = false)
    assertEvalsTo(invoke("is_infinite", TBoolean, F64(Double.NaN)), expected = false)
    assertEvalsTo(invoke("is_infinite", TBoolean, F64(Double.PositiveInfinity)), expected = true)
    assertEvalsTo(invoke("is_infinite", TBoolean, F64(Double.NegativeInfinity)), expected = true)
  }

  @Test def sign(): Unit = {
    implicit val execStrats = ExecStrategy.javaOnly

    assertEvalsTo(invoke("sign", TInt32, I32(2)), 1)
    assertEvalsTo(invoke("sign", TInt32, I32(0)), 0)
    assertEvalsTo(invoke("sign", TInt32, I32(-2)), -1)

    assertEvalsTo(invoke("sign", TInt64, I64(2)), 1L)
    assertEvalsTo(invoke("sign", TInt64, I64(0)), 0L)
    assertEvalsTo(invoke("sign", TInt64, I64(-2)), -1L)

    assertEvalsTo(invoke("sign", TFloat32, F32(2)), 1.0f)
    assertEvalsTo(invoke("sign", TFloat32, F32(0)), 0.0f)
    assertEvalsTo(invoke("sign", TFloat32, F32(-2)), -1.0f)

    assertEvalsTo(invoke("sign", TFloat64, F64(2)), 1.0)
    assertEvalsTo(invoke("sign", TFloat64, F64(0)), 0.0)
    assertEvalsTo(invoke("sign", TFloat64, F64(-2)), -1.0)

    assert(eval(invoke("sign", TFloat64, F64(Double.NaN))).asInstanceOf[Double].isNaN)
    assertEvalsTo(invoke("sign", TFloat64, F64(Double.PositiveInfinity)), 1.0)
    assertEvalsTo(invoke("sign", TFloat64, F64(Double.NegativeInfinity)), -1.0)
  }

  @Test def approxEqual(): Unit = {
    implicit val execStrats = ExecStrategy.javaOnly

    assertEvalsTo(
      invoke("approxEqual", TBoolean, F64(0.025), F64(0.0250000001), F64(1e-4), False(), False()),
      true,
    )
    assertEvalsTo(
      invoke("approxEqual", TBoolean, F64(0.0154), F64(0.0156), F64(1e-4), True(), False()),
      false,
    )
    assertEvalsTo(
      invoke("approxEqual", TBoolean, F64(0.0154), F64(0.0156), F64(1e-3), True(), False()),
      true,
    )
    assertEvalsTo(
      invoke("approxEqual", TBoolean, F64(Double.NaN), F64(Double.NaN), F64(1e-3), True(), False()),
      false,
    )
    assertEvalsTo(
      invoke("approxEqual", TBoolean, F64(Double.NaN), F64(Double.NaN), F64(1e-3), True(), True()),
      true,
    )
    assertEvalsTo(
      invoke(
        "approxEqual",
        TBoolean,
        F64(Double.PositiveInfinity),
        F64(Double.PositiveInfinity),
        F64(1e-3),
        True(),
        False(),
      ),
      true,
    )
    assertEvalsTo(
      invoke(
        "approxEqual",
        TBoolean,
        F64(Double.NegativeInfinity),
        F64(Double.NegativeInfinity),
        F64(1e-3),
        True(),
        False(),
      ),
      true,
    )
    assertEvalsTo(
      invoke(
        "approxEqual",
        TBoolean,
        F64(Double.PositiveInfinity),
        F64(Double.NegativeInfinity),
        F64(1e-3),
        True(),
        False(),
      ),
      false,
    )
  }

  @Test def entropy(): Unit = {
    implicit val execStrats = ExecStrategy.javaOnly

    assertEvalsTo(invoke("entropy", TFloat64, Str("")), 0.0)
    assertEvalsTo(invoke("entropy", TFloat64, Str("a")), 0.0)
    assertEvalsTo(invoke("entropy", TFloat64, Str("aa")), 0.0)
    assertEvalsTo(invoke("entropy", TFloat64, Str("ac")), 1.0)
    assertEvalsTo(invoke("entropy", TFloat64, Str("accctg")), 1.7924812503605778)
  }

  @DataProvider(name = "chi_squared_test")
  def chiSquaredData(): Array[Array[Any]] = Array(
    Array(0, 0, 0, 0, Double.NaN, Double.NaN),
    Array(0, 1, 1, 1, 0.38647623077123266, 0.0),
    Array(1, 0, 1, 1, 0.38647623077123266, Double.PositiveInfinity),
    Array(1, 1, 0, 1, 0.38647623077123266, Double.PositiveInfinity),
    Array(1, 1, 1, 0, 0.38647623077123266, 0.0),
    Array(10, 10, 10, 10, 1.0, 1.0),
    Array(51, 43, 22, 92, 1.462626e-7, (51.0 * 92) / (22 * 43)),
  )

  @Test(dataProvider = "chi_squared_test")
  def chiSquaredTest(a: Int, b: Int, c: Int, d: Int, pValue: Double, oddsRatio: Double): Unit = {
    val r = eval(invoke(
      "chi_squared_test",
      stats.chisqStruct.virtualType,
      ErrorIDs.NO_ERROR,
      a,
      b,
      c,
      d,
    )).asInstanceOf[Row]
    assert(D0_==(pValue, r.getDouble(0)))
    assert(D0_==(oddsRatio, r.getDouble(1)))
  }

  @DataProvider(name = "fisher_exact_test")
  def fisherExactData(): Array[Array[Any]] = Array(
    Array(0, 0, 0, 0, Double.NaN, Double.NaN, Double.NaN, Double.NaN),
    Array(10, 10, 10, 10, 1.0, 1.0, 0.243858, 4.100748),
    Array(51, 43, 22, 92, 2.1565e-7, 4.918058, 2.565937, 9.677930),
  )

  @Test(dataProvider = "fisher_exact_test")
  def fisherExactTest(
    a: Int,
    b: Int,
    c: Int,
    d: Int,
    pValue: Double,
    oddsRatio: Double,
    confLower: Double,
    confUpper: Double,
  ): Unit = {
    val r = eval(invoke(
      "fisher_exact_test",
      stats.fetStruct.virtualType,
      ErrorIDs.NO_ERROR,
      a,
      b,
      c,
      d,
    )).asInstanceOf[Row]
    assert(D0_==(pValue, r.getDouble(0)))
    assert(D0_==(oddsRatio, r.getDouble(1)))
    assert(D0_==(confLower, r.getDouble(2)))
    assert(D0_==(confUpper, r.getDouble(3)))
  }

  @DataProvider(name = "contingency_table_test")
  def contingencyTableData(): Array[Array[Any]] = Array(
    Array(51, 43, 22, 92, 22, 1.462626e-7, 4.95983087),
    Array(51, 43, 22, 92, 23, 2.1565e-7, 4.91805817),
  )

  @Test(dataProvider = "contingency_table_test")
  def contingencyTableTest(
    a: Int,
    b: Int,
    c: Int,
    d: Int,
    minCellCount: Int,
    pValue: Double,
    oddsRatio: Double,
  ): Unit = {
    val r = eval(invoke(
      "contingency_table_test",
      stats.chisqStruct.virtualType,
      ErrorIDs.NO_ERROR,
      a,
      b,
      c,
      d,
      minCellCount,
    )).asInstanceOf[Row]
    assert(D0_==(pValue, r.getDouble(0)))
    assert(D0_==(oddsRatio, r.getDouble(1)))
  }

  @DataProvider(name = "hardy_weinberg_test")
  def hardyWeinbergData(): Array[Array[Any]] = Array(
    Array(0, 0, 0, Double.NaN, 0.5),
    Array(1, 2, 1, 0.57142857, 0.65714285),
    Array(0, 1, 0, 1.0, 0.5),
    Array(100, 200, 100, 0.50062578, 0.96016808),
  )

  @Test(dataProvider = "hardy_weinberg_test")
  def hardyWeinbergTest(nHomRef: Int, nHet: Int, nHomVar: Int, pValue: Double, hetFreq: Double): Unit = {
    val r = eval(invoke(
      "hardy_weinberg_test",
      stats.hweStruct.virtualType,
      ErrorIDs.NO_ERROR,
      nHomRef,
      nHet,
      nHomVar,
      false,
    )).asInstanceOf[Row]
    assert(D0_==(pValue, r.getDouble(0)))
    assert(D0_==(hetFreq, r.getDouble(1)))
  }

  @Test def modulusTest(): Unit = {
    assertFatal(
      invoke("mod", TInt32, I32(1), I32(0)),
      "(modulo by zero)|(error while calling 'mod')",
    )
    assertFatal(
      invoke("mod", TInt64, I64(1), I64(0)),
      "(modulo by zero)|(error while calling 'mod')",
    )
    assertFatal(
      invoke("mod", TFloat32, F32(1), F32(0)),
      "(modulo by zero)|(error while calling 'mod')",
    )
    assertFatal(
      invoke("mod", TFloat64, F64(1), F64(0)),
      "(modulo by zero)|(error while calling 'mod')",
    )
  }

  @Test def testMinMax(): Unit = {
    implicit val execStrats = ExecStrategy.javaOnly
    assertAllEvalTo(
      (invoke("min", TFloat32, F32(1.0f), F32(2.0f)), 1.0f),
      (invoke("max", TFloat32, F32(1.0f), F32(2.0f)), 2.0f),
      (invoke("min", TFloat64, F64(1.0), F64(2.0)), 1.0),
      (invoke("max", TFloat64, F64(1.0), F64(2.0)), 2.0),
      (invoke("min", TInt32, I32(1), I32(2)), 1),
      (invoke("max", TInt32, I32(1), I32(2)), 2),
      (invoke("min", TInt64, I64(1L), I64(2L)), 1L),
      (invoke("max", TInt64, I64(1L), I64(2L)), 2L),
      (invoke("min", TFloat32, F32(Float.NaN), F32(1.0f)), Float.NaN),
      (invoke("min", TFloat64, F64(Double.NaN), F64(1.0)), Double.NaN),
      (invoke("max", TFloat32, F32(Float.NaN), F32(1.0f)), Float.NaN),
      (invoke("max", TFloat64, F64(Double.NaN), F64(1.0)), Double.NaN),
      (invoke("min", TFloat32, F32(1.0f), F32(Float.NaN)), Float.NaN),
      (invoke("min", TFloat64, F64(1.0), F64(Double.NaN)), Double.NaN),
      (invoke("max", TFloat32, F32(1.0f), F32(Float.NaN)), Float.NaN),
      (invoke("max", TFloat64, F64(1.0), F64(Double.NaN)), Double.NaN),
    )
  }
}
