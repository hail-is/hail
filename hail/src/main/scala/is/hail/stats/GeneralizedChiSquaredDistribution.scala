package is.hail.stats

import is.hail.types.physical._
import is.hail.utils._

case class DaviesAlgorithmTrace(
  var absoluteSum: Double,
  var totalNumberOfIntegrationTerms: Int,
  var numberOfIntegrations: Int,
  var integrationIntervalInFinalIntegration: Double,
  var truncationPointInInitialIntegration: Double,
  var standardDeviationOfInitialConvergenceFactor: Double,
  var cyclesToLocateIntegrationParameters: Int,
)

class DaviesResultForPython(
  val value: Double,
  val nIterations: Int,
  val converged: Boolean,
  val fault: Int,
)

object DaviesAlgorithm {
  private val pi = 3.14159265358979
  private val log28 = 0.0866
  private val divisForFindu = Array[Double](2.0, 1.4, 1.2, 1.1)
  private val rats = Array[Int](1, 2, 4, 8);

  val pType = PCanonicalStruct(
    "value" -> PFloat64(required = true),
    "n_iterations" -> PInt32(required = true),
    "converged" -> PBoolean(required = true),
    "fault" -> PInt32(required = true),
  )
}

class DaviesAlgorithm(
  private[this] val c: Double,
  private[this] val n: Array[Int],
  private[this] val lb: Array[Double],
  private[this] val nc: Array[Double],
  private[this] val lim: Int,
  private[this] val sigma: Double,
) {

  /** This algorithm is a direct port of Robert Davies' algorithm described in
    *
    * Davies, Robert. "The distribution of a linear combination of chi-squared random variables."
    * Applied Statistics 29 323-333. 1980.
    *
    * The Fortran code was published with the aforementioned paper. A port to C is available on
    * Davies' website http://www.robertnz.net/download.html . At the time of retrieval (2023-01-15),
    * the code lacks a description of its license. On 2023-01-18 0304 ET I received personal e-mail
    * correspondence from Robert Davies indicating:
    *
    * Assume it has the MIT license. That is on my todo list to say the MIT license applies to all
    * the software on the website unless specified otherwise.
    */
  import DaviesAlgorithm._
  import GeneralizedChiSquaredDistribution._

  private[this] val r: Int = lb.length
  private[this] var count: Int = 0
  private[this] var ndtsrt: Boolean = true // "need to sort"
  private[this] var fail: Boolean = true
  private[this] var th: Array[Int] = new Array[Int](r)
  private[this] var intl: Double = 0.0
  private[this] var ersm: Double = 0.0
  private[this] var sigsq: Double = square(sigma)
  private[this] var lmax: Double = 0.0
  private[this] var lmin: Double = 0.0
  private[this] var mean: Double = 0.0

  private class DaviesException() extends RuntimeException {}

  private[this] def counter(): Unit = {
    count += 1
    if (count > lim) {
      throw new DaviesException()
    }
  }

  def order(): Unit = { // "sort the th array", th appears to be a list of indices into lb
    var j = 0
    while (j < r) {
      val lj = Math.abs(lb(j))
      var k = j - 1
      var break: Boolean = false
      while (k >= 0 && !break) {
        if (lj > Math.abs(lb(th(k)))) {
          th(k + 1) = th(k)
          k -= 1
        } else {
          break = true
        }
      }
      if (!break) {
        assert(k == -1)
      }
      th(k + 1) = j
      j += 1
    }
    ndtsrt = false
  }

  private def errbd(_u: Double): (Double, Double) = {
    var u = _u
    counter()
    var xconst = u * sigsq
    var sum1 = u * xconst
    u = 2.0 * u
    var j = r - 1
    while (j >= 0) {
      val nj = n(j)
      val lj = lb(j)
      val ncj = nc(j)
      val x = u * lj
      val y = 1.0 - x
      xconst = xconst + lj * (ncj / y + nj) / y
      sum1 = (sum1 +
        ncj * square(x / y) +
        nj * (square(x) / y + log1(-x, false)))

      j -= 1
    }

    (exp1(-0.5 * sum1), xconst)
  }

  private def ctff(accx: Double, _u2: Double): (Double, Double) = {
    var u2 = _u2
    var u1 = 0.0
    var c1 = mean
    val rb = 2.0 * (if (u2 > 0.0) { lmax }
                    else { lmin })

    var u = u2 / (1.0 + u2 * rb)

    val errc2 = errbd(u)
    var err = errc2._1
    var c2 = errc2._2
    while (err > accx) {
      u1 = u2
      c1 = c2
      u2 = 2.0 * u2

      u = u2 / (1.0 + u2 * rb)

      val errc2 = errbd(u)
      err = errc2._1
      c2 = errc2._2
    }

    u = (c1 - mean) / (c2 - mean)
    while (u < 0.9) {
      u = (u1 + u2) / 2.0

      val errxconst = errbd(u / (1.0 + u * rb))
      err = errxconst._1
      val xconst = errxconst._2
      if (err > accx) {
        u1 = u
        c1 = xconst
      } else {
        u2 = u
        c2 = xconst
      }

      u = (c1 - mean) / (c2 - mean)
    }

    (c2, u2)
  }

  private def truncation(_u: Double, tausq: Double): Double = {
    counter()
    var u = _u
    var sum1 = 0.0
    var prod2 = 0.0
    var prod3 = 0.0
    var s = 0
    val sum2 = (sigsq + tausq) * square(u)
    var prod1 = 2.0 * sum2
    u = 2.0 * u

    var j = 0
    while (j < r) {
      val lj = lb(j)
      val ncj = nc(j)
      val nj = n(j)

      val x = square(u * lj)
      sum1 = sum1 + ncj * x / (1.0 + x)
      if (x > 1.0) {
        prod2 = prod2 + nj * Math.log(x)
        prod3 = prod3 + nj * log1(x, first = true)
        s = s + nj
      } else {
        prod1 = prod1 + nj * log1(x, first = true)
      }

      j += 1
    }

    sum1 = 0.5 * sum1
    prod2 = prod1 + prod2
    prod3 = prod1 + prod3
    var x = exp1(-sum1 - 0.25 * prod2) / pi
    val y = exp1(-sum1 - 0.25 * prod3) / pi

    var err1 = if (s == 0) {
      1.0
    } else {
      x * 2.0 / s
    }

    var err2 = if (prod3 > 1.0) {
      2.5 * y
    } else {
      1.0
    }

    if (err2 < err1) {
      err1 = err2
    }

    x = 0.5 * sum2

    if (x <= y) {
      err2 = 1.0
    } else {
      err2 = y / x
    }

    if (err1 < err2) {
      err1
    } else {
      err2
    }
  }

  private def findu(_ut: Double, accx: Double): Double = {
    var ut = _ut
    var u = ut / 4.0
    if (truncation(u, 0.0) > accx) {
      u = ut
      while (truncation(u, 0.0) > accx) {
        ut = ut * 4.0
        u = ut
      }
    } else {
      ut = u
      u = u / 4.0
      while (truncation(u, 0.0) <= accx) {
        ut = u
        u = u / 4.0
      }
    }
    var i = 0
    while (i < 4) {
      u = ut / divisForFindu(i)

      if (truncation(u, 0.0) <= accx) {
        ut = u
      }

      i += 1
    }
    ut
  }

  private def integrate(nterm: Int, interv: Double, tausq: Double, mainx: Boolean): Unit = {
    val inpi = interv / pi

    var k = nterm
    while (k >= 0) {
      val u = (k + 0.5) * interv
      var sum1 = -2.0 * u * c
      var sum2 = Math.abs(sum1)
      var sum3 = -0.5 * sigsq * square(u)

      var j = r - 1
      while (j >= 0) {
        val nj = n(j)
        val x = 2.0 * lb(j) * u
        var y = square(x)

        sum3 = sum3 - 0.25 * nj * log1(y, true)
        y = nc(j) * x / (1.0 + y)
        val z = nj * Math.atan(x) + y
        sum1 = sum1 + z
        sum2 = sum2 + Math.abs(z)
        sum3 = sum3 - 0.5 * x * y

        j -= 1
      }

      var x = inpi * exp1(sum3) / u
      if (!mainx) {
        x = x * (1.0 - exp1(-0.5 * tausq * square(u)))
      }
      sum1 = Math.sin(0.5 * sum1) * x
      sum2 = 0.5 * sum2 * x
      intl = intl + sum1
      ersm = ersm + sum2

      k -= 1
    }
  }

  private def cfe(x: Double): Double = {
    counter();
    if (ndtsrt) {
      order()
    }
    var axl = Math.abs(x)
    val sxl = if (x > 0.0) { 1.0 }
    else { -1.0 }
    var sum1 = 0.0;
    var j = r - 1
    var break = false
    while (j >= 0 && !break) {
      val t = th(j);
      if (lb(t) * sxl > 0.0) {
        val lj = Math.abs(lb(t));
        val axl1 = axl - lj * (n(t) + nc(t))
        val axl2 = lj / log28
        if (axl1 > axl2) {
          axl = axl1
        } else {
          if (axl > axl2) {
            axl = axl2
          }
          sum1 = (axl - axl1) / lj

          var k = j - 1
          while (k >= 0) {
            sum1 = sum1 + (n(th(k)) + nc(th(k)))
            k -= 1
          }
          break = true
        }
      }

      j -= 1
    }

    if (sum1 > 100.0) {
      fail = true // FIXME: we can return the fail parameter instead
      1.0
    } else {
      Math.pow(2.0, (sum1 / 4.0)) / (pi * square(axl))
    }
  }

  def cdf(acc: Double): (Double, DaviesAlgorithmTrace, Int) = {
    var acc1 = acc
    val trace = DaviesAlgorithmTrace(0.0, 0, 0, 0.0, 0.0, 0.0, 0)
    var ifault = 0
    var qfval = -1.0
    try {
      ndtsrt = true
      fail = false
      var xlim = lim.toDouble

      /* find mean, sd, max and min of lb, check that parameter values are valid */
      var sd = sigsq

      var j = 0
      while (j < r) {
        val nj = n(j)
        val lj = lb(j)
        val ncj = nc(j)
        if (nj < 0) {
          throw new HailException(
            s"Degrees of freedom parameters must all be positive, $j'th parameter is $nj."
          )
        }
        if (ncj < 0.0) {
          throw new HailException(
            s"Non-centrality parameters must all be positive, $j'th parameter is $ncj."
          )
        }
        sd = sd + square(lj) * (2 * nj + 4.0 * ncj)
        mean = mean + lj * (nj + ncj)
        if (lmax < lj) {
          lmax = lj
        } else if (lmin > lj) {
          lmin = lj
        }

        j += 1
      }

      if (sd == 0.0) {
        if (c > 0.0) {
          qfval = 1.0
        } else {
          qfval = 0.0
        }
        throw new DaviesException()
      }

      if (lmin == 0.0 && lmax == 0.0 && sigma == 0.0) {
        val lbStr = lb.mkString("(", ",", ")")
        throw new HailException(
          s"Either weights vector must be non-zero or sigma must be non-zero, found: $lbStr and $sigma."
        )
      }

      sd = Math.sqrt(sd)

      val almx = if (lmax < -lmin) {
        -lmin
      } else {
        lmax
      }

      /* starting values for findu, ctff */
      var utx = 16.0 / sd
      var up = 4.5 / sd
      var un = -up
      /* truncation point with no convergence factor */
      utx = findu(utx, .5 * acc1)
      /* does convergence factor help */
      if (c != 0.0 && (almx > 0.07 * sd)) {
        // FIXME: return the fail parameter
        val tausq = .25 * acc1 / cfe(c)
        if (fail) {
          fail = false
        } else if (truncation(utx, tausq) < .2 * acc1) {
          sigsq = sigsq + tausq
          utx = findu(utx, .25 * acc1)
          trace.standardDeviationOfInitialConvergenceFactor = Math.sqrt(tausq)
        }
      }
      trace.truncationPointInInitialIntegration = utx
      acc1 = 0.5 * acc1

      /* find RANGE of distribution, quit if outside this */
      var intv = 0.0
      var xnt = 0.0
      var stopL1 = false
      while (!stopL1) {
        val (c2, u2) = ctff(acc1, up)
        up = u2
        val d1 = c2 - c

        if (d1 < 0.0) {
          qfval = 1.0
          throw new DaviesException()
        }
        val (_c2, _u2) = ctff(acc1, un)
        un = _u2
        val d2 = c - _c2
        if (d2 < 0.0) {
          qfval = 0.0
          throw new DaviesException()
        }
        /* find integration interval */
        val divisor = if (d1 > d2) { d1 }
        else { d2 }
        intv = 2.0 * pi / divisor
        /* calculate number of terms required for main and auxillary integrations */
        xnt = utx / intv
        val xntm = 3.0 / Math.sqrt(acc1)
        if (xnt > xntm * 1.5) {
          /* parameters for auxillary integration */
          if (xntm > xlim) {
            ifault = 1
            throw new DaviesException()
          }
          val ntm = Math.floor(xntm + 0.5).toInt
          val intv1 = utx / ntm
          val x = 2.0 * pi / intv1
          if (x <= Math.abs(c)) {
            stopL1 = true
          } else {
            /* calculate convergence factor */
            val tausq = .33 * acc1 / (1.1 * (cfe(c - x) + cfe(c + x)))
            if (fail) {
              stopL1 = true
            } else {
              acc1 = .67 * acc1
              /* auxillary integration */
              integrate(ntm, intv1, tausq, false)
              xlim = xlim - xntm
              sigsq = sigsq + tausq
              trace.numberOfIntegrations += 1
              trace.totalNumberOfIntegrationTerms += ntm + 1
              /* find truncation point with new convergence factor */
              utx = findu(utx, .25 * acc1)
              acc1 = 0.75 * acc1
            }
          }
        } else {
          stopL1 = true
        }
      }

      /* main integration */
      trace.integrationIntervalInFinalIntegration = intv
      if (xnt > xlim) {
        ifault = 1
        throw new DaviesException()
      }
      val nt = Math.floor(xnt + 0.5).toInt;
      integrate(nt, intv, 0.0, true);
      trace.numberOfIntegrations += 1
      trace.totalNumberOfIntegrationTerms += nt + 1
      qfval = 0.5 - intl
      trace.absoluteSum = ersm

      /* test whether round-off error could be significant allow for radix 8 or 16 machines */
      up = ersm
      val x = up + acc / 10.0
      j = 0
      while (j < 4) {
        if (rats(j) * x == rats(j) * up) {
          ifault = 2
        }

        j += 1
      }
    } catch {
      case _: DaviesException =>
    }

    trace.cyclesToLocateIntegrationParameters = count
    (qfval, trace, ifault)
  }
}

object GeneralizedChiSquaredDistribution {
  def exp1(x: Double): Double =
    if (x < -50.0) {
      0.0
    } else {
      Math.exp(x)
    }

  def square(x: Double): Double = x * x

  def cube(x: Double): Double = x * x * x

  def log1(x: Double, first: Boolean): Double = {
    if (Math.abs(x) > 0.1) {
      if (first) {
        Math.log(1.0 + x)
      } else {
        Math.log(1.0 + x) - x
      }
    } else {
      val y = x / (2.0 + x)
      var term = 2.0 * cube(y)
      var k = 3.0
      var s = if (first) {
        2.0 * y
      } else {
        -x * y
      }
      val yy = square(y)
      var s1 = s + term / k
      while (s1 != s) {
        k = k + 2.0
        term = term * yy
        s = s1
        s1 = s + term / k
      }
      s
    }
  }

  def cdf(
    c: Double,
    n: Array[Int],
    lb: Array[Double],
    nc: Array[Double],
    sigma: Double,
    lim: Int,
    acc: Double,
  ): Double = {
    assert(n.length == lb.length)
    assert(lb.length == nc.length)
    assert(lim >= 0)
    assert(acc >= 0)

    val (value, _, fault) = new DaviesAlgorithm(c, n, lb, nc, lim, sigma).cdf(acc)

    assert(fault >= 0 && fault <= 2, fault)

    if (fault == 1) {
      throw new RuntimeException(
        s"Required accuracy ($acc) not achieved. Best value found was: $value."
      )
    }

    if (fault == 2) {
      throw new RuntimeException(
        s"Round-off error is possibly significant. Best value found was: $value."
      )
    }

    value
  }

  def cdfReturnExceptions(
    c: Double,
    n: Array[Int],
    lb: Array[Double],
    nc: Array[Double],
    sigma: Double,
    lim: Int,
    acc: Double,
  ): DaviesResultForPython = {
    assert(n.length == lb.length)
    assert(lb.length == nc.length)
    assert(lim >= 0)
    assert(acc >= 0)

    val (value, trace, fault) = new DaviesAlgorithm(c, n, lb, nc, lim, sigma).cdf(acc)

    assert(fault >= 0 && fault <= 2, fault)

    new DaviesResultForPython(value, trace.numberOfIntegrations, fault == 0, fault)
  }
}
