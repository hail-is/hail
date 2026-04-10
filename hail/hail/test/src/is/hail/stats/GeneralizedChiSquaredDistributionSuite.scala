package is.hail.stats

import is.hail.HailSuite

class GeneralizedChiSquaredDistributionSuite extends HailSuite {
  private[this] def pgenchisq(
    c: Double,
    n: Array[Int],
    lb: Array[Double],
    nc: Array[Double],
    sigma: Double,
    lim: Int,
    acc: Double,
  ) =
    new DaviesAlgorithm(c, n, lb, nc, lim, sigma).cdf(acc)

  private[this] def nearEqual(a: Double, b: Double): Boolean =
    /* Davies only reports 6 significant figures */
    Math.abs(a - b) < 0.0000005

  private[this] def nearEqualDAT(x: DaviesAlgorithmTrace, y: DaviesAlgorithmTrace): Boolean = {
    val DaviesAlgorithmTrace(a, b, c, d, e, f, g) = x
    val DaviesAlgorithmTrace(a2, b2, c2, d2, e2, f2, g2) = x
    (nearEqual(a, a2) &&
    b == b2 &&
    c == c2 &&
    nearEqual(d, d2) &&
    nearEqual(e, e2) &&
    nearEqual(f, f2) &&
    g == g2)
  }

  test("0") {
    val (actualValue, actualTrace, actualFault) = pgenchisq(
      1.0,
      Array(1, 1, 1),
      Array(6.0, 3.0, 1.0),
      Array(0.0, 0.0, 0.0),
      0.0,
      1000,
      0.0001,
    )

    assert(nearEqual(actualValue, 0.054213))
    assert(nearEqualDAT(
      actualTrace,
      DaviesAlgorithmTrace(0.76235, 744, 2, 0.03819, 53.37969, 0.0, 51),
    ))
    assertEquals(actualFault, 0)
  }

  test("1") {
    val (actualValue, actualTrace, actualFault) = pgenchisq(
      7.0,
      Array(1, 1, 1),
      Array(6.0, 3.0, 1.0),
      Array(0.0, 0.0, 0.0),
      0.0,
      1000,
      0.0001,
    )
    assert(nearEqual(actualValue, 0.493555))
    assert(nearEqualDAT(
      actualTrace,
      DaviesAlgorithmTrace(1.57018, 625, 2, 0.03964, 34.66214, 0.04784, 51),
    ))
    assertEquals(actualFault, 0)
  }

  test("2") {
    val (actualValue, actualTrace, actualFault) = pgenchisq(
      20.0,
      Array(1, 1, 1),
      Array(6.0, 3.0, 1.0),
      Array(0.0, 0.0, 0.0),
      0.0,
      1000,
      0.0001,
    )
    assert(nearEqual(actualValue, 0.876027))
    assert(nearEqualDAT(
      actualTrace,
      DaviesAlgorithmTrace(3.16244, 346, 1, 0.04602, 15.88681, 0.14159, 32),
    ))
    assertEquals(actualFault, 0)
  }

  test("3") {
    val (actualValue, actualTrace, actualFault) = pgenchisq(
      2.0,
      Array(2, 2, 2),
      Array(6.0, 3.0, 1.0),
      Array(0.0, 0.0, 0.0),
      0.0,
      1000,
      0.0001,
    )
    assert(nearEqual(actualValue, 0.006435))
    assert(nearEqualDAT(
      actualTrace,
      DaviesAlgorithmTrace(0.84764, 74, 1, 0.03514, 2.55311, 0.0, 22),
    ))
    assertEquals(actualFault, 0)
  }

  test("4") {
    val (actualValue, actualTrace, actualFault) = pgenchisq(
      20.0,
      Array(2, 2, 2),
      Array(6.0, 3.0, 1.0),
      Array(0.0, 0.0, 0.0),
      0.0,
      1000,
      0.0001,
    )
    assert(nearEqual(actualValue, 0.600208))
    assert(nearEqualDAT(
      actualTrace,
      DaviesAlgorithmTrace(1.74138, 66, 1, 0.03907, 2.55311, 0.0, 22),
    ))
    assertEquals(actualFault, 0)
  }

  test("5") {
    val (actualValue, actualTrace, actualFault) = pgenchisq(
      60.0,
      Array(2, 2, 2),
      Array(6.0, 3.0, 1.0),
      Array(0.0, 0.0, 0.0),
      0.0,
      1000,
      0.0001,
    )
    assert(nearEqual(actualValue, 0.983897))
    assert(nearEqualDAT(actualTrace, DaviesAlgorithmTrace(3.72757, 50, 1, 0.052, 2.55311, 0.0, 22)))
    assertEquals(actualFault, 0)
  }

  test("6") {
    val (actualValue, actualTrace, actualFault) = pgenchisq(
      10.0,
      Array(6, 4, 2),
      Array(6.0, 3.0, 1.0),
      Array(0.0, 0.0, 0.0),
      0.0,
      1000,
      0.0001,
    )
    assert(nearEqual(actualValue, 0.002697))
    assert(nearEqualDAT(
      actualTrace,
      DaviesAlgorithmTrace(1.20122, 18, 1, 0.02706, 0.46096, 0.0, 20),
    ))
    assertEquals(actualFault, 0)
  }

  test("7") {
    val (actualValue, actualTrace, actualFault) = pgenchisq(
      50.0,
      Array(6, 4, 2),
      Array(6.0, 3.0, 1.0),
      Array(0.0, 0.0, 0.0),
      0.0,
      1000,
      0.0001,
    )
    assert(nearEqual(actualValue, 0.564753))
    assert(nearEqualDAT(
      actualTrace,
      DaviesAlgorithmTrace(2.06868, 15, 1, 0.03269, 0.46096, 0.0, 20),
    ))
    assertEquals(actualFault, 0)
  }

  test("8") {
    val (actualValue, actualTrace, actualFault) = pgenchisq(
      120.0,
      Array(6, 4, 2),
      Array(6.0, 3.0, 1.0),
      Array(0.0, 0.0, 0.0),
      0.0,
      1000,
      0.0001,
    )
    assert(nearEqual(actualValue, 0.991229))
    assert(nearEqualDAT(
      actualTrace,
      DaviesAlgorithmTrace(3.58496, 10, 1, 0.05141, 0.46096, 0.0, 20),
    ))
    assertEquals(actualFault, 0)
  }

  test("9") {
    val (actualValue, actualTrace, actualFault) = pgenchisq(
      10.0,
      Array(2, 4, 6),
      Array(6.0, 3.0, 1.0),
      Array(0.0, 0.0, 0.0),
      0.0,
      1000,
      0.0001,
    )
    assert(nearEqual(actualValue, 0.033357))
    assert(nearEqualDAT(
      actualTrace,
      DaviesAlgorithmTrace(1.29976, 27, 1, 0.03459, 0.88302, 0.0, 19),
    ))
    assertEquals(actualFault, 0)
  }

  test("10") {
    val (actualValue, actualTrace, actualFault) = pgenchisq(
      30.0,
      Array(2, 4, 6),
      Array(6.0, 3.0, 1.0),
      Array(0.0, 0.0, 0.0),
      0.0,
      1000,
      0.0001,
    )
    assert(nearEqual(actualValue, 0.580446))
    assert(nearEqualDAT(
      actualTrace,
      DaviesAlgorithmTrace(2.01747, 24, 1, 0.03887, 0.88302, 0.0, 19),
    ))
    assertEquals(actualFault, 0)
  }

  test("11") {
    val (actualValue, actualTrace, actualFault) = pgenchisq(
      80.0,
      Array(2, 4, 6),
      Array(6.0, 3.0, 1.0),
      Array(0.0, 0.0, 0.0),
      0.0,
      1000,
      0.0001,
    )
    assert(nearEqual(actualValue, 0.991283))
    assert(nearEqualDAT(
      actualTrace,
      DaviesAlgorithmTrace(3.81157, 17, 1, 0.05628, 0.88302, 0.0, 19),
    ))
    assertEquals(actualFault, 0)
  }

  test("12") {
    val (actualValue, actualTrace, actualFault) = pgenchisq(
      20.0,
      Array(6, 2),
      Array(7.0, 3.0),
      Array(6.0, 2.0),
      0.0,
      1000,
      0.0001,
    )
    assert(nearEqual(actualValue, 0.006125))
    assert(nearEqualDAT(
      actualTrace,
      DaviesAlgorithmTrace(1.16271, 16, 1, 0.01561, 0.24013, 0.0, 19),
    ))
    assertEquals(actualFault, 0)
  }

  test("13") {
    val (actualValue, actualTrace, actualFault) = pgenchisq(
      100.0,
      Array(6, 2),
      Array(7.0, 3.0),
      Array(6.0, 2.0),
      0.0,
      1000,
      0.0001,
    )
    assert(nearEqual(actualValue, 0.591339))
    assert(nearEqualDAT(
      actualTrace,
      DaviesAlgorithmTrace(2.02277, 13, 1, 0.01949, 0.24013, 0.0, 19),
    ))
    assertEquals(actualFault, 0)
  }

  test("14") {
    val (actualValue, actualTrace, actualFault) = pgenchisq(
      200.0,
      Array(6, 2),
      Array(7.0, 3.0),
      Array(6.0, 2.0),
      0.0,
      1000,
      0.0001,
    )
    assert(nearEqual(actualValue, 0.977914))
    assert(nearEqualDAT(
      actualTrace,
      DaviesAlgorithmTrace(3.09687, 10, 1, 0.02825, 0.24013, 0.0, 19),
    ))
    assertEquals(actualFault, 0)
  }

  test("15") {
    val (actualValue, actualTrace, actualFault) = pgenchisq(
      10.0,
      Array(1, 1),
      Array(7.0, 3.0),
      Array(6.0, 2.0),
      0.0,
      1000,
      0.0001,
    )
    assert(nearEqual(actualValue, 0.045126))
    assert(nearEqualDAT(
      actualTrace,
      DaviesAlgorithmTrace(0.8712, 603, 2, 0.01628, 13.86318, 0.0, 49),
    ))
    assertEquals(actualFault, 0)
  }

  test("16") {
    val (actualValue, actualTrace, actualFault) = pgenchisq(
      60.0,
      Array(1, 1),
      Array(7.0, 3.0),
      Array(6.0, 2.0),
      0.0,
      1000,
      0.0001,
    )
    assert(nearEqual(actualValue, 0.592431))
    assert(nearEqualDAT(
      actualTrace,
      DaviesAlgorithmTrace(1.69157, 340, 1, 0.02043, 6.93159, 0.24644, 31),
    ))
    assertEquals(actualFault, 0)
  }

  test("17") {
    val (actualValue, actualTrace, actualFault) = pgenchisq(
      150.0,
      Array(1, 1),
      Array(7.0, 3.0),
      Array(6.0, 2.0),
      0.0,
      1000,
      0.0001,
    )
    assert(nearEqual(actualValue, 0.977648))
    assert(nearEqualDAT(
      actualTrace,
      DaviesAlgorithmTrace(3.06625, 87, 1, 0.02888, 2.47557, 0.81533, 29),
    ))
    assertEquals(actualFault, 0)
  }

  test("18") {
    val (actualValue, actualTrace, actualFault) = pgenchisq(
      45.0,
      Array(6, 4, 2, 2, 4, 6),
      Array(6.0, 3.0, 1.0, 12.0, 6.0, 2.0),
      Array(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
      0.0,
      1000,
      0.0001,
    )
    assert(nearEqual(actualValue, 0.01095))
    assert(nearEqualDAT(actualTrace, DaviesAlgorithmTrace(1.82147, 13, 1, 0.01582, 0.193, 0.0, 18)))
    assertEquals(actualFault, 0)
  }

  test("19") {
    val (actualValue, actualTrace, actualFault) = pgenchisq(
      120.0,
      Array(6, 4, 2, 2, 4, 6),
      Array(6.0, 3.0, 1.0, 12.0, 6.0, 2.0),
      Array(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
      0.0,
      1000,
      0.0001,
    )
    assert(nearEqual(actualValue, 0.654735))
    assert(nearEqualDAT(actualTrace, DaviesAlgorithmTrace(2.73768, 11, 1, 0.0195, 0.193, 0.0, 18)))
    assertEquals(actualFault, 0)
  }

  test("20") {
    val (actualValue, actualTrace, actualFault) = pgenchisq(
      210.0,
      Array(6, 4, 2, 2, 4, 6),
      Array(6.0, 3.0, 1.0, 12.0, 6.0, 2.0),
      Array(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
      0.0,
      1000,
      0.0001,
    )
    assert(nearEqual(actualValue, 0.984606))
    assert(nearEqualDAT(actualTrace, DaviesAlgorithmTrace(3.83651, 8, 1, 0.02707, 0.193, 0.0, 18)))
    assertEquals(actualFault, 0)
  }

  test("21") {
    val (actualValue, actualTrace, actualFault) = pgenchisq(
      70.0,
      Array(6, 2, 1, 1),
      Array(7.0, 3.0, 7.0, 3.0),
      Array(6.0, 2.0, 6.0, 2.0),
      0.0,
      1000,
      0.0001,
    )
    assert(nearEqual(actualValue, 0.043679))
    assert(nearEqualDAT(
      actualTrace,
      DaviesAlgorithmTrace(1.65876, 10, 1, 0.01346, 0.12785, 0.0, 18),
    ))
    assertEquals(actualFault, 0)
  }

  test("22") {
    val (actualValue, actualTrace, actualFault) = pgenchisq(
      160.0,
      Array(6, 2, 1, 1),
      Array(7.0, 3.0, 7.0, 3.0),
      Array(6.0, 2.0, 6.0, 2.0),
      0.0,
      1000,
      0.0001,
    )
    assert(nearEqual(actualValue, 0.584765))
    assert(nearEqualDAT(
      actualTrace,
      DaviesAlgorithmTrace(2.34799, 9, 1, 0.01668, 0.12785, 0.0, 18),
    ))
    assertEquals(actualFault, 0)
  }

  test("23") {
    val (actualValue, actualTrace, actualFault) = pgenchisq(
      260.0,
      Array(6, 2, 1, 1),
      Array(7.0, 3.0, 7.0, 3.0),
      Array(6.0, 2.0, 6.0, 2.0),
      0.0,
      1000,
      0.0001,
    )
    assert(nearEqual(actualValue, 0.953774))
    assert(nearEqualDAT(
      actualTrace,
      DaviesAlgorithmTrace(3.11236, 7, 1, 0.02271, 0.12785, 0.0, 18),
    ))
    assertEquals(actualFault, 0)
  }

  test("24") {
    val (actualValue, actualTrace, actualFault) = pgenchisq(
      -40.0,
      Array(6, 2, 1, 1),
      Array(7.0, 3.0, -7.0, -3.0),
      Array(6.0, 2.0, 6.0, 2.0),
      0.0,
      1000,
      0.0001,
    )
    assert(nearEqual(actualValue, 0.078208))
    assert(nearEqualDAT(
      actualTrace,
      DaviesAlgorithmTrace(1.42913, 10, 1, 0.01483, 0.12785, 0.0, 19),
    ))
    assertEquals(actualFault, 0)
  }

  test("25") {
    val (actualValue, actualTrace, actualFault) = pgenchisq(
      40.0,
      Array(6, 2, 1, 1),
      Array(7.0, 3.0, -7.0, -3.0),
      Array(6.0, 2.0, 6.0, 2.0),
      0.0,
      1000,
      0.0001,
    )
    assert(nearEqual(actualValue, 0.522108))
    assert(nearEqualDAT(
      actualTrace,
      DaviesAlgorithmTrace(1.42909, 8, 1, 0.01771, 0.12785, 0.0, 19),
    ))
    assertEquals(actualFault, 0)
  }

  test("26") {
    val (actualValue, actualTrace, actualFault) = pgenchisq(
      140.0,
      Array(6, 2, 1, 1),
      Array(7.0, 3.0, -7.0, -3.0),
      Array(6.0, 2.0, 6.0, 2.0),
      0.0,
      1000,
      0.0001,
    )
    assert(nearEqual(actualValue, 0.96037))
    assert(nearEqualDAT(
      actualTrace,
      DaviesAlgorithmTrace(2.19476, 10, 1, 0.01381, 0.12785, 0.0, 19),
    ))
    assertEquals(actualFault, 0)
  }

  test("27") {
    val (actualValue, actualTrace, actualFault) = pgenchisq(
      120.0,
      Array(6, 4, 2, 2, 4, 6, 6, 2, 1, 1),
      Array(6.0, 3.0, 1.0, 6.0, 3.0, 1.0, 7.0, 3.0, 7.0, 3.0),
      Array(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 6.0, 2.0, 6.0, 2.0),
      0.0,
      1000,
      0.0001,
    )
    assert(nearEqual(actualValue, 0.015844))
    assert(nearEqualDAT(
      actualTrace,
      DaviesAlgorithmTrace(2.33438, 9, 1, 0.01202, 0.09616, 0.0, 18),
    ))
    assertEquals(actualFault, 0)
  }

  test("28") {
    val (actualValue, actualTrace, actualFault) = pgenchisq(
      240.0,
      Array(6, 4, 2, 2, 4, 6, 6, 2, 1, 1),
      Array(6.0, 3.0, 1.0, 6.0, 3.0, 1.0, 7.0, 3.0, 7.0, 3.0),
      Array(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 6.0, 2.0, 6.0, 2.0),
      0.0,
      1000,
      0.0001,
    )
    assert(nearEqual(actualValue, 0.573625))
    assert(nearEqualDAT(actualTrace, DaviesAlgorithmTrace(3.1401, 7, 1, 0.01561, 0.09616, 0.0, 18)))
    assertEquals(actualFault, 0)
  }

  test("29") {
    val (actualValue, actualTrace, actualFault) = pgenchisq(
      400.0,
      Array(6, 4, 2, 2, 4, 6, 6, 2, 1, 1),
      Array(6.0, 3.0, 1.0, 6.0, 3.0, 1.0, 7.0, 3.0, 7.0, 3.0),
      Array(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 6.0, 2.0, 6.0, 2.0),
      0.0,
      1000,
      0.0001,
    )
    assert(nearEqual(actualValue, 0.988332))
    assert(nearEqualDAT(actualTrace, DaviesAlgorithmTrace(4.2142, 6, 1, 0.01812, 0.09616, 0.0, 18)))
    assertEquals(actualFault, 0)
  }

  test("30") {
    val (actualValue, actualTrace, actualFault) = pgenchisq(
      5.0,
      Array(1, 10),
      Array(30.0, 1.0),
      Array(0.0, 0.0),
      0.0,
      1000,
      0.0001,
    )
    assert(nearEqual(actualValue, 0.015392))
    assert(nearEqualDAT(
      actualTrace,
      DaviesAlgorithmTrace(0.95892, 163, 1, 0.00841, 1.3638, 0.0, 22),
    ))
    assertEquals(actualFault, 0)
  }

  test("31") {
    val (actualValue, actualTrace, actualFault) = pgenchisq(
      25.0,
      Array(1, 10),
      Array(30.0, 1.0),
      Array(0.0, 0.0),
      0.0,
      1000,
      0.0001,
    )
    assert(nearEqual(actualValue, 0.510819))
    assert(nearEqualDAT(
      actualTrace,
      DaviesAlgorithmTrace(1.72922, 159, 1, 0.00864, 1.3638, 0.0, 22),
    ))
    assertEquals(actualFault, 0)
  }

  test("32") {
    val (actualValue, actualTrace, actualFault) = pgenchisq(
      100.0,
      Array(1, 10),
      Array(30.0, 1.0),
      Array(0.0, 0.0),
      0.0,
      1000,
      0.0001,
    )
    assert(nearEqual(actualValue, 0.91634))
    assert(nearEqualDAT(
      actualTrace,
      DaviesAlgorithmTrace(4.61788, 143, 1, 0.00963, 1.3638, 0.0, 22),
    ))
    assertEquals(actualFault, 0)
  }

  test("33") {
    val (actualValue, actualTrace, actualFault) = pgenchisq(
      10.0,
      Array(1, 20),
      Array(30.0, 1.0),
      Array(0.0, 0.0),
      0.0,
      1000,
      0.0001,
    )
    assert(nearEqual(actualValue, 0.004925))
    assert(nearEqualDAT(
      actualTrace,
      DaviesAlgorithmTrace(1.26245, 97, 1, 0.00839, 0.80736, 0.0, 21),
    ))
    assertEquals(actualFault, 0)
  }

  test("34") {
    val (actualValue, actualTrace, actualFault) = pgenchisq(
      40.0,
      Array(1, 20),
      Array(30.0, 1.0),
      Array(0.0, 0.0),
      0.0,
      1000,
      0.0001,
    )
    assert(nearEqual(actualValue, 0.573251))
    assert(nearEqualDAT(
      actualTrace,
      DaviesAlgorithmTrace(2.16513, 93, 1, 0.00874, 0.80736, 0.0, 21),
    ))
    assertEquals(actualFault, 0)
  }

  test("35") {
    val (actualValue, actualTrace, actualFault) = pgenchisq(
      100.0,
      Array(1, 20),
      Array(30.0, 1.0),
      Array(0.0, 0.0),
      0.0,
      1000,
      0.0001,
    )
    assert(nearEqual(actualValue, 0.896501))
    assert(nearEqualDAT(
      actualTrace,
      DaviesAlgorithmTrace(3.97055, 86, 1, 0.00954, 0.80736, 0.0, 21),
    ))
    assertEquals(actualFault, 0)
  }

  test("36") {
    val (actualValue, actualTrace, actualFault) = pgenchisq(
      20.0,
      Array(1, 30),
      Array(30.0, 1.0),
      Array(0.0, 0.0),
      0.0,
      1000,
      0.0001,
    )
    assert(nearEqual(actualValue, 0.017101))
    assert(nearEqualDAT(
      actualTrace,
      DaviesAlgorithmTrace(1.65684, 81, 1, 0.00843, 0.67453, 0.0, 20),
    ))
    assertEquals(actualFault, 0)
  }

  test("37") {
    val (actualValue, actualTrace, actualFault) = pgenchisq(
      50.0,
      Array(1, 30),
      Array(30.0, 1.0),
      Array(0.0, 0.0),
      0.0,
      1000,
      0.0001,
    )
    assert(nearEqual(actualValue, 0.566488))
    assert(nearEqualDAT(
      actualTrace,
      DaviesAlgorithmTrace(2.44382, 78, 1, 0.00878, 0.67453, 0.0, 20),
    ))
    assertEquals(actualFault, 0)
  }

  test("38") {
    val (actualValue, actualTrace, actualFault) = pgenchisq(
      100.0,
      Array(1, 30),
      Array(30.0, 1.0),
      Array(0.0, 0.0),
      0.0,
      1000,
      0.0001,
    )
    assert(nearEqual(actualValue, 0.871323))
    assert(nearEqualDAT(
      actualTrace,
      DaviesAlgorithmTrace(3.75545, 72, 1, 0.00944, 0.67453, 0.0, 20),
    ))
    assertEquals(actualFault, 0)
  }
}
