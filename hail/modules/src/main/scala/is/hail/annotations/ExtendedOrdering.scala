package is.hail.annotations

import is.hail.utils._

import org.apache.spark.sql.Row

object ExtendedOrdering {
  def extendToNull[S](ord: Ordering[S], _missingEqual: Boolean = true): ExtendedOrdering = {
    new ExtendedOrdering {
      val missingEqual = _missingEqual

      def compareNonnull(x: T, y: T): Int = ord.compare(x.asInstanceOf[S], y.asInstanceOf[S])

      override def ltNonnull(x: T, y: T): Boolean = ord.lt(x.asInstanceOf[S], y.asInstanceOf[S])

      override def lteqNonnull(x: T, y: T): Boolean = ord.lteq(x.asInstanceOf[S], y.asInstanceOf[S])

      override def equivNonnull(x: T, y: T): Boolean =
        ord.equiv(x.asInstanceOf[S], y.asInstanceOf[S])
    }
  }

  def iterableOrdering[T](ord: ExtendedOrdering, _missingEqual: Boolean = true): ExtendedOrdering =
    new ExtendedOrdering {
      val missingEqual = _missingEqual

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

  def sortArrayOrdering(ord: ExtendedOrdering, _missingEqual: Boolean = true): ExtendedOrdering =
    new ExtendedOrdering {
      val missingEqual = _missingEqual

      private val itOrd = iterableOrdering(ord)

      // ord can be null if the element type is a TVariable
      val elemOrd = if (ord != null) ord.toOrdering else null

      def compareNonnull(x: T, y: T): Int =
        itOrd.compareNonnull(
          x.asInstanceOf[Array[T]].sorted(elemOrd).toFastSeq,
          y.asInstanceOf[Array[T]].sorted(elemOrd).toFastSeq,
        )

      override def ltNonnull(x: T, y: T): Boolean =
        itOrd.ltNonnull(
          x.asInstanceOf[Array[T]].sorted(elemOrd).toFastSeq,
          y.asInstanceOf[Array[T]].sorted(elemOrd).toFastSeq,
        )

      override def lteqNonnull(x: T, y: T): Boolean =
        itOrd.lteqNonnull(
          x.asInstanceOf[Array[T]].sorted(elemOrd).toFastSeq,
          y.asInstanceOf[Array[T]].sorted(elemOrd).toFastSeq,
        )

      override def equivNonnull(x: T, y: T): Boolean =
        itOrd.equivNonnull(
          x.asInstanceOf[Array[T]].sorted(elemOrd).toFastSeq,
          y.asInstanceOf[Array[T]].sorted(elemOrd).toFastSeq,
        )
    }

  def setOrdering(ord: ExtendedOrdering, _missingEqual: Boolean = true): ExtendedOrdering =
    new ExtendedOrdering {
      private val saOrd = sortArrayOrdering(ord)

      val missingEqual = _missingEqual

      def compareNonnull(x: T, y: T): Int =
        saOrd.compareNonnull(
          x.asInstanceOf[Iterable[T]].toArray,
          y.asInstanceOf[Iterable[T]].toArray,
        )

      override def ltNonnull(x: T, y: T): Boolean =
        saOrd.ltNonnull(
          x.asInstanceOf[Iterable[T]].toArray,
          y.asInstanceOf[Iterable[T]].toArray,
        )

      override def lteqNonnull(x: T, y: T): Boolean =
        saOrd.lteqNonnull(
          x.asInstanceOf[Iterable[T]].toArray,
          y.asInstanceOf[Iterable[T]].toArray,
        )

      override def equivNonnull(x: T, y: T): Boolean =
        saOrd.equivNonnull(
          x.asInstanceOf[Iterable[T]].toArray,
          y.asInstanceOf[Iterable[T]].toArray,
        )
    }

  def mapOrdering(ord: ExtendedOrdering, _missingEqual: Boolean = true): ExtendedOrdering =
    new ExtendedOrdering {
      private val saOrd = sortArrayOrdering(ord)

      val missingEqual = _missingEqual

      private def toArrayOfT(x: T): Array[T] =
        x.asInstanceOf[Map[_, _]].iterator.map { case (k, v) => Row(k, v): T }.toArray

      def compareNonnull(x: T, y: T): Int =
        saOrd.compareNonnull(
          toArrayOfT(x),
          toArrayOfT(y),
        )

      override def ltNonnull(x: T, y: T): Boolean =
        saOrd.ltNonnull(
          toArrayOfT(x),
          toArrayOfT(y),
        )

      override def lteqNonnull(x: T, y: T): Boolean =
        saOrd.lteqNonnull(
          toArrayOfT(x),
          toArrayOfT(y),
        )

      override def equivNonnull(x: T, y: T): Boolean =
        saOrd.equivNonnull(
          toArrayOfT(x),
          toArrayOfT(y),
        )
    }

  def rowOrdering(fieldOrd: Array[ExtendedOrdering], _missingEqual: Boolean = true)
    : ExtendedOrdering =
    new ExtendedOrdering {
      outer =>
      val missingEqual = _missingEqual

      override def compareNonnull(x: T, y: T): Int = {
        val rx = x.asInstanceOf[Row]
        val ry = y.asInstanceOf[Row]
        val rLen = rx.length
        assert(rLen == fieldOrd.length)
        assert(ry.length == fieldOrd.length)

        var i = 0
        while (i < rLen) {
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
        val rLen = rx.length
        assert(rLen == ry.length)

        var i = 0
        while (i < rLen) {
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
        val rLen = rx.length
        assert(rLen == ry.length)

        var i = 0
        while (i < rLen) {
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
        val rLen = rx.length
        assert(rLen == ry.length)

        var i = 0
        while (i < rLen) {
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
    }
}

abstract class ExtendedOrdering extends Serializable {
  outer =>

  type T = Any

  val missingEqual: Boolean

  def compareNonnull(x: T, y: T): Int

  def ltNonnull(x: T, y: T): Boolean = compareNonnull(x, y) < 0

  def lteqNonnull(x: T, y: T): Boolean = compareNonnull(x, y) <= 0

  def equivNonnull(x: T, y: T): Boolean = compareNonnull(x, y) == 0

  def gtNonnull(x: T, y: T): Boolean = ltNonnull(y, x)

  def gteqNonnull(x: T, y: T): Boolean = lteqNonnull(y, x)

  def compare(x: T, y: T): Int = {
    if (y == null) {
      if (x == null)
        if (missingEqual) 0 else -1
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
        !missingEqual
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
        missingEqual
      else
        false
    } else {
      if (x == null)
        false
      else
        equivNonnull(x, y)
    }
  }

  def gt(x: T, y: T): Boolean = {
    if (y == null) {
      false
    } else {
      if (x == null)
        true
      else
        gtNonnull(x, y)
    }
  }

  def gteq(x: T, y: T): Boolean = {
    if (y == null) {
      if (x == null)
        missingEqual
      else
        false
    } else {
      if (x == null)
        true
      else
        gteqNonnull(x, y)
    }
  }

  // reverses the sense of the non-null comparison only
  def reverse: ExtendedOrdering = new ExtendedOrdering {
    val missingEqual = outer.missingEqual

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

      override def lteqWithOverlap(allowedOverlap: Int)(x: IntervalEndpoint, y: IntervalEndpoint)
        : Boolean = {
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

  val missingEqual = true

  override def compareNonnull(x: Any, y: Any): Int = {
    val xp = if (x.isInstanceOf[IntervalEndpoint]) x.asInstanceOf[IntervalEndpoint].point else x
    val xs = if (x.isInstanceOf[IntervalEndpoint]) x.asInstanceOf[IntervalEndpoint].sign else 0
    val yp = if (y.isInstanceOf[IntervalEndpoint]) y.asInstanceOf[IntervalEndpoint].point else y
    val ys = if (y.isInstanceOf[IntervalEndpoint]) y.asInstanceOf[IntervalEndpoint].sign else 0
    compareIntervalEndpoints(xp, xs, yp, ys)
  }
}
