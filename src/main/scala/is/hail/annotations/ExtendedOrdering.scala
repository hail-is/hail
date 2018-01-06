package is.hail.annotations

import is.hail.utils._
import org.apache.spark.sql.Row

object ExtendedOrdering {
  def extendToNull[S](ord: Ordering[S]): ExtendedOrdering = {
    new ExtendedOrdering {
      def compareNonnull(x: T, y: T, missingGreatest: Boolean): Int = ord.compare(x.asInstanceOf[S], y.asInstanceOf[S])

      override def ltNonnull(x: T, y: T, missingGreatest: Boolean): Boolean = ord.lt(x.asInstanceOf[S], y.asInstanceOf[S])

      override def lteqNonnull(x: T, y: T, missingGreatest: Boolean): Boolean = ord.lteq(x.asInstanceOf[S], y.asInstanceOf[S])

      override def gtNonnull(x: T, y: T, missingGreatest: Boolean): Boolean = ord.gt(x.asInstanceOf[S], y.asInstanceOf[S])

      override def gteqNonnull(x: T, y: T, missingGreatest: Boolean): Boolean = ord.gteq(x.asInstanceOf[S], y.asInstanceOf[S])

      override def equivNonnull(x: T, y: T, missingGreatest: Boolean): Boolean = ord.equiv(x.asInstanceOf[S], y.asInstanceOf[S])

      override def minNonnull(x: T, y: T, missingGreatest: Boolean): T = ord.min(x.asInstanceOf[S], y.asInstanceOf[S])

      override def maxNonnull(x: T, y: T, missingGreatest: Boolean): T = ord.max(x.asInstanceOf[S], y.asInstanceOf[S])
    }
  }

  def extendToNull(ord: ExtendedOrdering): ExtendedOrdering = {
    new ExtendedOrdering {
      def compareNonnull(x: T, y: T, missingGreatest: Boolean): Int = ord.compare(x, y, missingGreatest)

      override def ltNonnull(x: T, y: T, missingGreatest: Boolean): Boolean = ord.lt(x, y, missingGreatest)

      override def lteqNonnull(x: T, y: T, missingGreatest: Boolean): Boolean = ord.lteq(x, y, missingGreatest)

      override def gtNonnull(x: T, y: T, missingGreatest: Boolean): Boolean = ord.gt(x, y, missingGreatest)

      override def gteqNonnull(x: T, y: T, missingGreatest: Boolean): Boolean = ord.gteq(x, y, missingGreatest)

      override def equivNonnull(x: T, y: T, missingGreatest: Boolean): Boolean = ord.equiv(x, y, missingGreatest)

      override def minNonnull(x: T, y: T, missingGreatest: Boolean): T = ord.min(x, y, missingGreatest)

      override def maxNonnull(x: T, y: T, missingGreatest: Boolean): T = ord.max(x, y, missingGreatest)
    }
  }

  def iterableOrdering[T](ord: ExtendedOrdering): ExtendedOrdering =
    new ExtendedOrdering {
      def compareNonnull(x: T, y: T, missingGreatest: Boolean): Int = {
        val xit = x.asInstanceOf[Iterable[T]].iterator
        val yit = y.asInstanceOf[Iterable[T]].iterator

        while (xit.hasNext && yit.hasNext) {
          val c = ord.compare(xit.next(), yit.next(), missingGreatest)
          if (c != 0)
            return c
        }

        java.lang.Boolean.compare(xit.hasNext, yit.hasNext)
      }
    }

  def sortArrayOrdering(ord: ExtendedOrdering): ExtendedOrdering =
    new ExtendedOrdering {
      private val itOrd = iterableOrdering(ord)
      
      def compareNonnull(x: T, y: T, missingGreatest: Boolean): Int = {
        val ax = x.asInstanceOf[Array[T]]
        val ay = y.asInstanceOf[Array[T]]
        val scalaOrd = ord.toOrdering
        itOrd.compareNonnull(ax.sorted(scalaOrd).toFastIndexedSeq, ay.sorted(scalaOrd).toFastIndexedSeq, missingGreatest)
      }
    }

  def setOrdering(ord: ExtendedOrdering): ExtendedOrdering =
    new ExtendedOrdering {
      private val saOrd = sortArrayOrdering(ord)

      def compareNonnull(x: T, y: T, missingGreatest: Boolean): Int = {
        val ix = x.asInstanceOf[Iterable[T]]
        val iy = y.asInstanceOf[Iterable[T]]
        saOrd.compareNonnull(ix.toArray, iy.toArray, missingGreatest)
      }
    }

  def mapOrdering(ord: ExtendedOrdering): ExtendedOrdering =
    new ExtendedOrdering {
      private val saOrd = sortArrayOrdering(ord)

      def compareNonnull(x: T, y: T, missingGreatest: Boolean): Int = {
        val mx = x.asInstanceOf[Map[T, T]]
        val my = y.asInstanceOf[Map[T, T]]
        
        saOrd.compareNonnull(
          mx.toArray.map { case (k, v) => Row(k, v): T },
          my.toArray.map { case (k, v) => Row(k, v): T },
          missingGreatest)
      }
    }

  def rowOrdering(fieldOrd: Array[ExtendedOrdering]): ExtendedOrdering =
    new ExtendedOrdering {
      def compareNonnull(x: T, y: T, missingGreatest: Boolean): Int = {
        val rx = x.asInstanceOf[Row]
        val ry = y.asInstanceOf[Row]
        
        var i = 0
        while (i < fieldOrd.length) {
          val c = fieldOrd(i).compare(rx.get(i), ry.get(i), missingGreatest)
          if (c != 0)
            return c
          i += 1
        }

        // equal
        0
      }
    }
}

abstract class ExtendedOrdering extends Serializable {
  outer =>
  
  type T = Any
  
  def compareNonnull(x: T, y: T, missingGreatest: Boolean): Int

  def ltNonnull(x: T, y: T, missingGreatest: Boolean): Boolean = compareNonnull(x, y, missingGreatest) < 0

  def lteqNonnull(x: T, y: T, missingGreatest: Boolean): Boolean = compareNonnull(x, y, missingGreatest) <= 0

  def gtNonnull(x: T, y: T, missingGreatest: Boolean): Boolean = compareNonnull(x, y, missingGreatest) > 0

  def gteqNonnull(x: T, y: T, missingGreatest: Boolean): Boolean = compareNonnull(x, y, missingGreatest) >= 0

  def equivNonnull(x: T, y: T, missingGreatest: Boolean): Boolean = compareNonnull(x, y, missingGreatest) == 0

  def minNonnull(x: T, y: T, missingGreatest: Boolean): T = {
    if (ltNonnull(x, y, missingGreatest))
      x
    else
      y
  }

  def maxNonnull(x: T, y: T, missingGreatest: Boolean): T = {
    if (gtNonnull(x, y, missingGreatest))
      x
    else
      y
  }

  def compare(x: T, y: T, missingGreatest: Boolean): Int = {
    if (y == null) {
      if (x == null)
        0
      else if (missingGreatest) -1 else 1
    } else {
      if (x == null)
        if (missingGreatest) 1 else -1
      else
        compareNonnull(x, y, missingGreatest)
    }
  }

  def lt(x: T, y: T, missingGreatest: Boolean): Boolean = {
    if (y == null) {
      if (x == null)
        false
      else
        missingGreatest
    } else {
      if (x == null)
        !missingGreatest
      else
        ltNonnull(x, y, missingGreatest)
    }
  }

  def lteq(x: T, y: T, missingGreatest: Boolean): Boolean = {
    if (y == null) {
      if (x == null)
        true
      else
        missingGreatest
    } else {
      if (x == null)
        !missingGreatest
      else
        lteqNonnull(x, y, missingGreatest)
    }
  }

  def gt(x: T, y: T, missingGreatest: Boolean): Boolean = {
    if (y == null) {
      if (x == null)
        false
      else
        !missingGreatest
    } else {
      if (x == null)
        missingGreatest
      else
        gtNonnull(x, y, missingGreatest)
    }
  }

  def gteq(x: T, y: T, missingGreatest: Boolean): Boolean = {
    if (y == null) {
      if (x == null)
        true
      else
        !missingGreatest
    } else {
      if (x == null)
        missingGreatest
      else
        gteqNonnull(x, y, missingGreatest)
    }
  }
  
  def equiv(x: T, y: T, missingGreatest: Boolean): Boolean = {
    if (y == null) {
      if (x == null)
        true
      else
        false
    } else {
      if (x == null)
        false
      else
        equivNonnull(x, y, missingGreatest)
    }
  }

  def min(x: T, y: T, missingGreatest: Boolean): T = {
    if (y == null) {
      if (missingGreatest) x else y
    } else {
      if (x == null)
        if (missingGreatest) y else x
      else
        minNonnull(x, y, missingGreatest)
    }
  }

  def max(x: T, y: T, missingGreatest: Boolean): T = {
    if (y == null) {
      if (missingGreatest) y else x
    } else {
      if (x == null)
        if (missingGreatest) x else y
      else
        maxNonnull(x, y, missingGreatest)
    }
  }

  def compare(x: T, y: T): Int = compare(x, y, missingGreatest = true)

  def lt(x: T, y: T): Boolean = lt(x, y, missingGreatest = true)

  def lteq(x: T, y: T): Boolean = lteq(x, y, missingGreatest = true)

  def gt(x: T, y: T): Boolean = gt(x, y, missingGreatest = true)

  def gteq(x: T, y: T): Boolean = gteq(x, y, missingGreatest = true)

  def equiv(x: T, y: T): Boolean = equiv(x, y, missingGreatest = true)

  def min(x: T, y: T): T = min(x, y, missingGreatest = true)

  def max(x: T, y: T): T = max(x, y, missingGreatest = true)

  // reverses the sense of the non-null comparison only
  def reverse: ExtendedOrdering = new ExtendedOrdering {
    override def reverse: ExtendedOrdering = outer

    def compareNonnull(x: T, y: T, missingGreatest: Boolean): Int = outer.compareNonnull(y, x, missingGreatest)

    override def ltNonnull(x: T, y: T, missingGreatest: Boolean): Boolean = outer.ltNonnull(y, x, missingGreatest)

    override def lteqNonnull(x: T, y: T, missingGreatest: Boolean): Boolean = outer.lteqNonnull(y, x, missingGreatest)

    override def gtNonnull(x: T, y: T, missingGreatest: Boolean): Boolean = outer.gtNonnull(y, x, missingGreatest)

    override def gteqNonnull(x: T, y: T, missingGreatest: Boolean): Boolean = outer.gteqNonnull(y, x, missingGreatest)

    override def equivNonnull(x: T, y: T, missingGreatest: Boolean): Boolean = outer.equivNonnull(y, x, missingGreatest)

    override def minNonnull(x: T, y: T, missingGreatest: Boolean): T = outer.maxNonnull(x, y, missingGreatest)

    override def maxNonnull(x: T, y: T, missingGreatest: Boolean): T = outer.minNonnull(x, y, missingGreatest)
  }

  def toOrdering: Ordering[T] = new Ordering[T] {
    def compare(x: T, y: T): Int = outer.compare(x, y)

    override def lt(x: T, y: T): Boolean = outer.lt(x, y)

    override def lteq(x: T, y: T): Boolean = outer.lteq(x, y)

    override def gt(x: T, y: T): Boolean = outer.gt(x, y)

    override def gteq(x: T, y: T): Boolean = outer.gteq(x, y)

    override def equiv(x: T, y: T): Boolean = outer.equiv(x, y)

    override def min(x: T, y: T): T = outer.min(x, y)

    override def max(x: T, y: T): T = outer.max(x, y)
  }

  def toOrdering(missingGreatest: Boolean): Ordering[T] = new Ordering[T] {
    def compare(x: T, y: T): Int = outer.compare(x, y, missingGreatest)

    override def lt(x: T, y: T): Boolean = outer.lt(x, y, missingGreatest)

    override def lteq(x: T, y: T): Boolean = outer.lteq(x, y, missingGreatest)

    override def gt(x: T, y: T): Boolean = outer.gt(x, y, missingGreatest)

    override def gteq(x: T, y: T): Boolean = outer.gteq(x, y, missingGreatest)

    override def equiv(x: T, y: T): Boolean = outer.equiv(x, y, missingGreatest)

    override def min(x: T, y: T): T = outer.min(x, y, missingGreatest)

    override def max(x: T, y: T): T = outer.max(x, y, missingGreatest)    
  }
}
