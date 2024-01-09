package is.hail.expr.ir

import is.hail.asm4s._
import is.hail.expr.ir.orderings.CodeOrdering
import is.hail.types.physical.stypes._
import is.hail.types.physical.stypes.interfaces._
import is.hail.utils.FastSeq

import scala.language.existentials

object BinarySearch {
  object Comparator {
    def fromLtGt(
      ltNeedle: IEmitCode => Code[Boolean],
      gtNeedle: IEmitCode => Code[Boolean],
    ): Comparator = new Comparator {
      def apply(
        cb: EmitCodeBuilder,
        elt: IEmitCode,
        ifLtNeedle: => Unit,
        ifGtNeedle: => Unit,
        ifNeither: => Unit,
      ): Unit = {
        val eltVal = cb.memoize(elt)
        cb.if_(
          ltNeedle(eltVal.loadI(cb)),
          ifLtNeedle,
          cb.if_(gtNeedle(eltVal.loadI(cb)), ifGtNeedle, ifNeither),
        )
      }
    }

    def fromCompare(compare: IEmitCode => Value[Int]): Comparator = new Comparator {
      def apply(
        cb: EmitCodeBuilder,
        elt: IEmitCode,
        ifLtNeedle: => Unit,
        ifGtNeedle: => Unit,
        ifNeither: => Unit,
      ): Unit = {
        val c = cb.memoize(compare(elt))
        cb.if_(c < 0, ifLtNeedle, cb.if_(c > 0, ifGtNeedle, ifNeither))
      }
    }

    def fromPred(pred: IEmitCode => Code[Boolean]): Comparator = new Comparator {
      def apply(
        cb: EmitCodeBuilder,
        elt: IEmitCode,
        ifLtNeedle: => Unit,
        ifGtNeedle: => Unit,
        ifNeither: => Unit,
      ): Unit =
        cb.if_(pred(elt), ifGtNeedle, ifLtNeedle)
    }
  }

  /** Represents a discriminator of values of some type into one of three mutually exclusive
    * categories:
    *   - less than the needle (whatever is being searched for)
    *   - greater than the needle
    *   - neither (interpretation depends on the application, e.g. "equals needle", "contains
    *     needle", "contained in needle")
    */
  abstract class Comparator {
    def apply(
      cb: EmitCodeBuilder,
      elt: IEmitCode,
      ltNeedle: => Unit,
      gtNeedle: => Unit,
      neither: => Unit,
    ): Unit
  }

  /** Returns true if haystack contains an element x such that !ltNeedle(x) and !gtNeedle(x), false
    * otherwise.
    */
  def containsOrdered(
    cb: EmitCodeBuilder,
    haystack: SIndexableValue,
    ltNeedle: IEmitCode => Code[Boolean],
    gtNeedle: IEmitCode => Code[Boolean],
  ): Value[Boolean] =
    containsOrdered(cb, haystack, Comparator.fromLtGt(ltNeedle, gtNeedle))

  /** Returns true if haystack contains an element x such that !lt(x, needle) and !lt(needle, x),
    * false otherwise.
    */
  def containsOrdered(
    cb: EmitCodeBuilder,
    haystack: SIndexableValue,
    needle: EmitValue,
    lt: (IEmitCode, IEmitCode) => Code[Boolean],
    key: IEmitCode => IEmitCode,
  ): Value[Boolean] =
    containsOrdered(
      cb,
      haystack,
      x => lt(key(x), needle.loadI(cb)),
      x => lt(needle.loadI(cb), key(x)),
    )

  def containsOrdered(
    cb: EmitCodeBuilder,
    haystack: SIndexableValue,
    compare: Comparator,
  ): Value[Boolean] =
    runSearch[Boolean](cb, haystack, compare, (_, _, _) => true, (_) => false)

  /** Returns (l, u) such that
    *   - range [0, l) is < needle
    *   - range [l, u) is incomparable ("equal") to needle
    *   - range [u, size) is > needle
    *
    * Assumes comparator separates haystack into < needle, followed by incomparable to needle,
    * followed by > needle.
    */
  def equalRange(
    cb: EmitCodeBuilder,
    haystack: SIndexableValue,
    compare: Comparator,
    ltNeedle: IEmitCode => Code[Boolean],
    gtNeedle: IEmitCode => Code[Boolean],
    start: Value[Int],
    end: Value[Int],
  ): (Value[Int], Value[Int]) = {
    val l = cb.newLocal[Int]("equalRange_l")
    val u = cb.newLocal[Int]("equalRange_u")
    runSearchBoundedUnit(
      cb,
      haystack,
      compare,
      start,
      end,
      (curL, m, curU) => {
        // [start, curL) is < needle
        // [start, m] is <= needle
        // [m, end) is >= needle
        // [curR, end) is > needle
        cb.assign(l, lowerBound(cb, haystack, ltNeedle, curL, m))
        // [curL, l) is < needle
        // [l, m) is >= needle
        cb.assign(u, upperBound(cb, haystack, gtNeedle, cb.memoize(m + 1), curU))
        // [m+1, u) is <= needle
        // [u, curU) is > needle
      },
      m => {
        // [start, m) is < needle
        // [m, end) is > needle
        cb.assign(l, m)
        cb.assign(u, m)
      },
    )
    (l, u)
  }

  /** Returns i in ['start', 'end'] such that
    *   - range [start, i) is < needle
    *   - range [i, end) is >= needle
    *
    * Assumes ltNeedle is down-closed, i.e. all trues precede all falses
    */
  def lowerBound(
    cb: EmitCodeBuilder,
    haystack: SIndexableValue,
    ltNeedle: IEmitCode => Code[Boolean],
    start: Value[Int],
    end: Value[Int],
  ): Value[Int] =
    partitionPoint(cb, haystack, x => !ltNeedle(x), start, end)

  def lowerBound(
    cb: EmitCodeBuilder,
    haystack: SIndexableValue,
    ltNeedle: IEmitCode => Code[Boolean],
  ): Value[Int] =
    lowerBound(cb, haystack, ltNeedle, 0, haystack.loadLength())

  /** Returns i in ['start', 'end'] such that
    *   - range [start, i) is <= needle
    *   - range [i, end) is > needle
    *
    * Assumes gtNeedle is up-closed, i.e. all falses precede all trues
    */
  def upperBound(
    cb: EmitCodeBuilder,
    haystack: SIndexableValue,
    gtNeedle: IEmitCode => Code[Boolean],
    start: Value[Int],
    end: Value[Int],
  ): Value[Int] =
    partitionPoint(cb, haystack, gtNeedle, start, end)

  def upperBound(
    cb: EmitCodeBuilder,
    haystack: SIndexableValue,
    gtNeedle: IEmitCode => Code[Boolean],
  ): Value[Int] =
    lowerBound(cb, haystack, gtNeedle, 0, haystack.loadLength())

  /** Returns 'start' <= i <= 'end' such that
    *   - pred is false on range [start, i), and
    *   - pred is true on range [i, end).
    *
    * Assumes pred partitions a, i.e. for all 0 <= i <= j < haystack.size, if pred(i) then pred(j),
    * i.e. all falses precede all trues.
    */
  def partitionPoint(
    cb: EmitCodeBuilder,
    haystack: SIndexableValue,
    pred: IEmitCode => Code[Boolean],
    start: Value[Int],
    end: Value[Int],
  ): Value[Int] = {
    var i: Value[Int] = null
    runSearchBoundedUnit(
      cb,
      haystack,
      Comparator.fromPred(pred),
      start,
      end,
      (_, _, _) => {}, // unreachable
      _i => i = _i,
    )
    i
  }

  def partitionPoint(
    cb: EmitCodeBuilder,
    haystack: SIndexableValue,
    pred: IEmitCode => Code[Boolean],
  ): Value[Int] =
    partitionPoint(cb, haystack, pred, const(0), haystack.loadLength())

  /** Perform binary search until either
    *   - an index m is found for which haystack(i) is incomparable with the needle, i.e. neither
    *     ltNeedle(m) nor gtNeedle(m). In this case, call found(l, m, u), where
    *     - haystack(m) is incomparable to needle
    *     - range [start, l) is < needle
    *     - range [r, end) is > needle
    *   - it is certain that no such m exists. In this case, call notFound(j), where
    *     - range [start, j) is < needle
    *     - range [j, end) is > needle
    *
    * Assumes comparator separates haystack into < needle, followed by incomparable to needle,
    * followed by > needle.
    */
  private def runSearchBoundedUnit(
    cb: EmitCodeBuilder,
    haystack: SIndexableValue,
    compare: Comparator,
    start: Value[Int],
    end: Value[Int],
    found: (Value[Int], Value[Int], Value[Int]) => Unit,
    notFound: Value[Int] => Unit,
  ): Unit = {
    val left = cb.newLocal[Int]("left", start)
    val right = cb.newLocal[Int]("right", end)
    // loop invariants:
    // - range [start, left) is < needle
    // - range [right, end) is > needle
    // - left <= right
    // terminates b/c (right - left) strictly decreases each iteration
    cb.loop { recur =>
      cb.if_(
        left < right, {
          val mid = cb.memoize((left + right) >>> 1) // works even when sum overflows
          compare(
            cb,
            haystack.loadElement(cb, mid), {
              // range [start, mid] is < needle
              cb.assign(left, mid + 1)
              cb.goto(recur)
            }, {
              // range [mid, end) is > needle
              cb.assign(right, mid)
              cb.goto(recur)
            },
            // haystack(mid) is incomparable to needle
            found(left, mid, right),
          )
        },
        // now loop invariants hold, with left = right, so
        // - range [start, left) is < needle
        // - range [left, end) is > needle
        notFound(left),
      )
    }
  }

  private def runSearchUnit(
    cb: EmitCodeBuilder,
    haystack: SIndexableValue,
    compare: Comparator,
    found: (Value[Int], Value[Int], Value[Int]) => Unit,
    notFound: Value[Int] => Unit,
  ): Unit =
    runSearchBoundedUnit(cb, haystack, compare, 0, haystack.loadLength(), found, notFound)

  private def runSearchBounded[T: TypeInfo](
    cb: EmitCodeBuilder,
    haystack: SIndexableValue,
    compare: Comparator,
    start: Value[Int],
    end: Value[Int],
    found: (Value[Int], Value[Int], Value[Int]) => Code[T],
    notFound: Value[Int] => Code[T],
  ): Value[T] = {
    val ret = cb.newLocal[T]("runSearch_ret")
    runSearchBoundedUnit(
      cb,
      haystack,
      compare,
      start,
      end,
      (l, m, r) => cb.assign(ret, found(l, m, r)),
      i => cb.assign(ret, notFound(i)),
    )
    ret
  }

  private def runSearch[T: TypeInfo](
    cb: EmitCodeBuilder,
    haystack: SIndexableValue,
    compare: Comparator,
    found: (Value[Int], Value[Int], Value[Int]) => Code[T],
    notFound: Value[Int] => Code[T],
  ): Value[T] =
    runSearchBounded[T](cb, haystack, compare, 0, haystack.loadLength(), found, notFound)
}

class BinarySearch[C](
  mb: EmitMethodBuilder[C],
  containerType: SContainer,
  eltType: EmitType,
  getKey: (EmitCodeBuilder, EmitValue) => EmitValue,
  bound: String = "lower",
) {
  val containerElementType: EmitType = containerType.elementEmitType

  val findElt = mb.genEmitMethod(
    "findElt",
    FastSeq[ParamType](containerType.paramType, eltType.paramType),
    typeInfo[Int],
  )

  // Returns i in [0, n] such that a(j) < key for j in [0, i), and a(j) >= key
  // for j in [i, n)
  findElt.emitWithBuilder[Int] { cb =>
    val haystack = findElt.getSCodeParam(1).asIndexable
    val needle = findElt.getEmitParam(cb, 2)

    val f: (
      EmitCodeBuilder,
      SIndexableValue,
      IEmitCode => Code[Boolean],
    ) => Value[Int] = bound match {
      case "upper" => BinarySearch.upperBound
      case "lower" => BinarySearch.lowerBound
    }

    f(
      cb,
      haystack,
      { containerElement =>
        val elementVal = cb.memoize(containerElement, "binary_search_elt")
        val compareVal = getKey(cb, elementVal)
        bound match {
          case "upper" =>
            val gt = mb.ecb.getOrderingFunction(compareVal.st, eltType.st, CodeOrdering.Gt())
            gt(cb, compareVal, needle)
          case "lower" =>
            val lt = mb.ecb.getOrderingFunction(compareVal.st, eltType.st, CodeOrdering.Lt())
            lt(cb, compareVal, needle)
        }
      },
    )
  }

  // check missingness of v before calling
  def search(cb: EmitCodeBuilder, array: SValue, v: EmitCode): Value[Int] =
    cb.memoize(cb.invokeCode[Int](findElt, cb.this_, array, v))
}
