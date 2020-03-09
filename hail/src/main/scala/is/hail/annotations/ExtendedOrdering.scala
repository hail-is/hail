package is.hail.annotations

import is.hail.utils._
import org.apache.spark.sql.Row

object ExtendedOrdering {
  def extendToNull[S](ord: Ordering[S]): ExtendedOrdering = {
    new ExtendedOrdering {
      def compareNonnull(x: T, y: T): Int = ord.compare(x.asInstanceOf[S], y.asInstanceOf[S])

      override def ltNonnull(x: T, y: T): Boolean = ord.lt(x.asInstanceOf[S], y.asInstanceOf[S])

      override def lteqNonnull(x: T, y: T): Boolean = ord.lteq(x.asInstanceOf[S], y.asInstanceOf[S])

      override def equivNonnull(x: T, y: T): Boolean = ord.equiv(x.asInstanceOf[S], y.asInstanceOf[S])
    }
  }

  def iterableOrdering[T](ord: ExtendedOrdering): ExtendedOrdering =
    new ExtendedOrdering {
      def compareNonnull(x: T, y: T): Int = {
        val xit = x.asInstanceOf[Iterable[T]].iterator
        val yit = y.asInstanceOf[Iterable[T]].iterator

        while (xit.hasNext && yit.hasNext) {
          val c = ord.compare(xit.next(), yit.next())
          if (c != 0)
            return c
        }

        java.lang.Boolean.compare(xit.hasNext, yit.hasNext)
      }

      override def ltNonnull(x: T, y: T): Boolean = {
        val xit = x.asInstanceOf[Iterable[T]].iterator
        val yit = y.asInstanceOf[Iterable[T]].iterator

        while (xit.hasNext && yit.hasNext) {
          val xi = xit.next()
          val yi = yit.next()
          if (ord.lt(xi, yi))
            return true
          else if (!ord.equiv(xi, yi))
            return false
        }

        xit.hasNext < yit.hasNext
      }

      override def lteqNonnull(x: T, y: T): Boolean = {
        val xit = x.asInstanceOf[Iterable[T]].iterator
        val yit = y.asInstanceOf[Iterable[T]].iterator

        while (xit.hasNext && yit.hasNext) {
          val xi = xit.next()
          val yi = yit.next()
          if (ord.lt(xi, yi))
            return true
          else if (!ord.equiv(xi, yi))
            return false
        }

        xit.hasNext <= yit.hasNext
      }

      override def equivNonnull(x: T, y: T): Boolean = {
        val xit = x.asInstanceOf[Iterable[T]].iterator
        val yit = y.asInstanceOf[Iterable[T]].iterator

        while (xit.hasNext && yit.hasNext) {
          val xi = xit.next()
          val yi = yit.next()
          if (!ord.equiv(xi, yi))
            return false
        }

        xit.hasNext == yit.hasNext
      }
    }

  def sortArrayOrdering(ord: ExtendedOrdering): ExtendedOrdering =
    new ExtendedOrdering {
      private val itOrd = iterableOrdering(ord)

      // ord can be null if the element type is a TVariable
      private val elemOrd = if (ord != null) ord.toOrdering else null

      def compareNonnull(x: T, y: T): Int = {
        itOrd.compareNonnull(
          x.asInstanceOf[Array[T]].sorted(elemOrd).toFastIndexedSeq,
          y.asInstanceOf[Array[T]].sorted(elemOrd).toFastIndexedSeq)
      }

      override def ltNonnull(x: T, y: T): Boolean = {
        itOrd.ltNonnull(
          x.asInstanceOf[Array[T]].sorted(elemOrd).toFastIndexedSeq,
          y.asInstanceOf[Array[T]].sorted(elemOrd).toFastIndexedSeq)
      }

      override def lteqNonnull(x: T, y: T): Boolean = {
        itOrd.lteqNonnull(
          x.asInstanceOf[Array[T]].sorted(elemOrd).toFastIndexedSeq,
          y.asInstanceOf[Array[T]].sorted(elemOrd).toFastIndexedSeq)
      }

      override def equivNonnull(x: T, y: T): Boolean = {
        itOrd.equivNonnull(
          x.asInstanceOf[Array[T]].sorted(elemOrd).toFastIndexedSeq,
          y.asInstanceOf[Array[T]].sorted(elemOrd).toFastIndexedSeq)
      }
    }

  def setOrdering(ord: ExtendedOrdering): ExtendedOrdering =
    new ExtendedOrdering {
      private val saOrd = sortArrayOrdering(ord)

      def compareNonnull(x: T, y: T): Int = {
        saOrd.compareNonnull(
          x.asInstanceOf[Iterable[T]].toArray,
          y.asInstanceOf[Iterable[T]].toArray)
      }

      override def ltNonnull(x: T, y: T): Boolean = {
        saOrd.ltNonnull(
          x.asInstanceOf[Iterable[T]].toArray,
          y.asInstanceOf[Iterable[T]].toArray)
      }

      override def lteqNonnull(x: T, y: T): Boolean = {
        saOrd.lteqNonnull(
          x.asInstanceOf[Iterable[T]].toArray,
          y.asInstanceOf[Iterable[T]].toArray)
      }

      override def equivNonnull(x: T, y: T): Boolean = {
        saOrd.equivNonnull(
          x.asInstanceOf[Iterable[T]].toArray,
          y.asInstanceOf[Iterable[T]].toArray)
      }
    }

  def mapOrdering(ord: ExtendedOrdering): ExtendedOrdering =
    new ExtendedOrdering {
      private val saOrd = sortArrayOrdering(ord)

      private def toArrayOfT(x: T): Array[T] =
        x.asInstanceOf[Map[_, _]].iterator.map { case (k, v) => Row(k, v): T }.toArray

      def compareNonnull(x: T, y: T): Int = {
        saOrd.compareNonnull(
          toArrayOfT(x), toArrayOfT(y))
      }

      override def ltNonnull(x: T, y: T): Boolean = {
        saOrd.ltNonnull(
          toArrayOfT(x), toArrayOfT(y))
      }

      override def lteqNonnull(x: T, y: T): Boolean = {
        saOrd.lteqNonnull(
          toArrayOfT(x), toArrayOfT(y))
      }

      override def equivNonnull(x: T, y: T): Boolean = {
        saOrd.equivNonnull(
          toArrayOfT(x), toArrayOfT(y))
      }
    }

  def rowOrdering(fieldOrd: Array[ExtendedOrdering]): ExtendedOrdering =
    new ExtendedOrdering { outer =>
      override def compareNonnull(x: T, y: T): Int = {
        val rx = x.asInstanceOf[Row]
        val ry = y.asInstanceOf[Row]

        val commonPrefix = math.min(fieldOrd.length, math.min(rx.length, ry.length))
        var i = 0
        while (i < commonPrefix) {
          val c = fieldOrd(i).compare(rx.get(i), ry.get(i))
          if (c != 0)
            return c
          i += 1
        }

        // equal
        0
      }

      override def ltNonnull(x: T, y: T): Boolean = {
        val rx = x.asInstanceOf[Row]
        val ry = y.asInstanceOf[Row]

        val commonPrefix = math.min(fieldOrd.length, math.min(rx.length, ry.length))
        var i = 0
        while (i < commonPrefix) {
          val fOrd = fieldOrd(i)
          val rxi = rx.get(i)
          val ryi = ry.get(i)

          if (fOrd.lt(rxi, ryi))
            return true
          if (!fOrd.equiv(rxi, ryi))
            return false
          i += 1
        }

        // equal
        false
      }

      override def lteqNonnull(x: T, y: T): Boolean = {
        val rx = x.asInstanceOf[Row]
        val ry = y.asInstanceOf[Row]

        val commonPrefix = math.min(fieldOrd.length, math.min(rx.length, ry.length))
        var i = 0
        while (i < commonPrefix) {
          val fOrd = fieldOrd(i)
          val rxi = rx.get(i)
          val ryi = ry.get(i)

          if (fOrd.lt(rxi, ryi))
            return true
          if (!fOrd.equiv(rxi, ryi))
            return false
          i += 1
        }

        // equal
        true
      }

      override def equivNonnull(x: T, y: T): Boolean = {
        val rx = x.asInstanceOf[Row]
        val ry = y.asInstanceOf[Row]

        val commonPrefix = math.min(fieldOrd.length, math.min(rx.length, ry.length))
        var i = 0
        while (i < commonPrefix) {
          val fOrd = fieldOrd(i)
          val rxi = rx.get(i)
          val ryi = ry.get(i)
          if (!fOrd.equiv(rxi, ryi))
            return false
          i += 1
        }

        // equal
        true
      }

      override lazy val intervalEndpointOrdering = new IntervalEndpointOrdering {
        override def compareIntervalEndpoints(xp: Any, xs: Int, yp: Any, ys: Int): Int = {
          val xpp = xp.asInstanceOf[Row]
          val ypp = yp.asInstanceOf[Row]
          val l = fieldOrd.length

          val c = outer.compare(xpp, ypp)
          if (c != 0)
            c
          else if (xpp != null && ypp != null) {
            val ll = xpp.length
            val rr = ypp.length
            if (l < ll && l < rr)
              0
            else {
              val cl = Integer.compare(xpp.length, ypp.length)
              if (cl == 0) Integer.compare(xs, ys)
              else if (cl < 0) xs
              else -ys
            }
          } else
            Integer.compare(xs, ys)
        }

        // Returns true if for any rows r1 and r2 with r1 < x and r2 > y,
        // the length of the largest common prefix of r1 and r2 is less than
        // or equal to 'allowedOverlap'
        override def lteqWithOverlap(allowedOverlap: Int)(x: IntervalEndpoint, y: IntervalEndpoint): Boolean = {
          require(allowedOverlap <= fieldOrd.length)
          val xp = x
          val yp = y
          val xpp = xp.point.asInstanceOf[Row]
          val ypp = yp.point.asInstanceOf[Row]
          val l = fieldOrd.length

          val prefix = Seq(l, xpp.length, ypp.length, allowedOverlap + 1).min
          var i = 0
          while (i < prefix) {
            val c = fieldOrd(i).compare(xpp.get(i), ypp.get(i))
            if (c != 0)
              return c < 0
            i += 1
          }
          val cl = xpp.length compare ypp.length
          if (allowedOverlap == l)
            prefix == l ||
              (cl < 0 && xp.sign < 0) ||
              (cl > 0 && yp.sign > 0) ||
              (cl == 0 && xp.sign <= yp.sign)
          else
            (xpp.length <= allowedOverlap + 1 || ypp.length <= allowedOverlap + 1) && (
              (cl < 0 && xp.sign < 0) ||
                (cl > 0 && yp.sign > 0) ||
                (cl == 0 && xp.sign <= yp.sign))
        }
      }
    }
}

abstract class ExtendedOrdering extends Serializable {
  outer =>

  type T = Any

  def compareNonnull(x: T, y: T): Int

  def ltNonnull(x: T, y: T): Boolean = compareNonnull(x, y) < 0

  def lteqNonnull(x: T, y: T): Boolean  = compareNonnull(x, y) <= 0

  def equivNonnull(x: T, y: T): Boolean = compareNonnull(x, y) == 0

  def gtNonnull(x: T, y: T): Boolean = ltNonnull(y, x)

  def gteqNonnull(x: T, y: T): Boolean = lteqNonnull(y, x)

  def compare(x: T, y: T): Int = {
    if (y == null) {
      if (x == null)
        0
      else -1
    } else {
      if (x == null)
        1
      else
        compareNonnull(x, y)
    }
  }

  def lt(x: T, y: T): Boolean = {
    if (y == null) {
      if (x == null)
        false
      else
        true
    } else {
      if (x == null)
        false
      else
        ltNonnull(x, y)
    }
  }

  def lteq(x: T, y: T): Boolean = {
    if (y == null) {
      if (x == null)
        true
      else
        true
    } else {
      if (x == null)
        false
      else
        lteqNonnull(x, y)
    }
  }

  def equiv(x: T, y: T): Boolean = {
    if (y == null) {
      if (x == null)
        true
      else
        false
    } else {
      if (x == null)
        false
      else
        equivNonnull(x, y)
    }
  }

  def gt(x: T, y: T): Boolean = lt(y, x)

  def gteq(x: T, y: T): Boolean = lteq(y, x)

  // reverses the sense of the non-null comparison only
  def reverse: ExtendedOrdering = new ExtendedOrdering {
    override def reverse: ExtendedOrdering = outer

    def compareNonnull(x: T, y: T): Int = outer.compareNonnull(y, x)

    override def ltNonnull(x: T, y: T): Boolean = outer.ltNonnull(y, x)

    override def lteqNonnull(x: T, y: T): Boolean = outer.lteqNonnull(y, x)

    override def equivNonnull(x: T, y: T): Boolean = outer.equivNonnull(y, x)
  }

  def toOrdering: Ordering[T] = new Ordering[T] {
    def compare(x: T, y: T): Int = outer.compare(x, y)

    override def lt(x: T, y: T): Boolean = outer.lt(x, y)

    override def lteq(x: T, y: T): Boolean = outer.lteq(x, y)

    override def equiv(x: T, y: T): Boolean = outer.equiv(x, y)
  }

  lazy val intervalEndpointOrdering: IntervalEndpointOrdering =
    new IntervalEndpointOrdering {
      override def compareIntervalEndpoints(xp: Any, xs: Int, yp: Any, ys: Int): Int = {
        val c = outer.compare(xp, yp)
        if (c != 0)
          c
        else
          Integer.compare(xs, ys)
      }

      override def lteqWithOverlap(allowedOverlap: Int)(x: IntervalEndpoint, y: IntervalEndpoint): Boolean = {
        val xp = x.point
        val xs = x.sign
        val yp = y.point
        val ys = y.sign

        val c = outer.compare(xp, yp)
        if (c != 0)
          c < 0
        else
          allowedOverlap == 1 || xs <= ys
      }
    }
}

abstract class IntervalEndpointOrdering extends ExtendedOrdering {
  def compareIntervalEndpoints(xp: Any, xs: Int, yp: Any, ys: Int): Int

  def lteqWithOverlap(allowedOverlap: Int)(x: IntervalEndpoint, y: IntervalEndpoint): Boolean

  override def compareNonnull(x: Any, y: Any): Int = {
    val xp = if (x.isInstanceOf[IntervalEndpoint]) x.asInstanceOf[IntervalEndpoint].point else x
    val xs = if (x.isInstanceOf[IntervalEndpoint]) x.asInstanceOf[IntervalEndpoint].sign else 0
    val yp = if (y.isInstanceOf[IntervalEndpoint]) y.asInstanceOf[IntervalEndpoint].point else y
    val ys = if (y.isInstanceOf[IntervalEndpoint]) y.asInstanceOf[IntervalEndpoint].sign else 0
    compareIntervalEndpoints(xp, xs, yp, ys)
  }
}
