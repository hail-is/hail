package is.hail.rvd

import is.hail.annotations.{ExtendedOrdering, IntervalEndpointOrdering, SafeRow}
import is.hail.backend.{ExecuteContext, HailStateManager}
import is.hail.types.physical.{PStruct, PType}
import is.hail.types.virtual._
import is.hail.utils._

import org.apache.spark.sql.Row

object PartitionBoundOrdering {
  def apply(ctx: ExecuteContext, _kType: Type): ExtendedOrdering =
    apply(ctx.stateManager, _kType)

  def apply(sm: HailStateManager, _kType: Type): ExtendedOrdering = {
    val kType = _kType.asInstanceOf[TBaseStruct]
    val fieldOrd = kType.types.map(_.ordering(sm))

    new ExtendedOrdering {
      outer =>
      val missingEqual = true

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
        override def lteqWithOverlap(allowedOverlap: Int)(x: IntervalEndpoint, y: IntervalEndpoint)
          : Boolean = {
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
                (cl == 0 && xp.sign <= yp.sign)
            )
        }
      }
    }
  }
}
