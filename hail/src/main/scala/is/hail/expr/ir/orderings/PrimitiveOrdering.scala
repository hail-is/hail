package is.hail.expr.ir.orderings

import is.hail.asm4s.Code
import is.hail.expr.ir.{EmitClassBuilder, EmitCodeBuilder}
import is.hail.types.physical.PCode
import is.hail.types.physical.stypes.primitives._

object Int32Ordering {
  def make(t1: SInt32, t2: SInt32, ecb: EmitClassBuilder[_]): CodeOrdering = {
    new CodeOrdering {

      val type1: SInt32 = t1
      val type2: SInt32 = t2

      def _compareNonnull(cb: EmitCodeBuilder, x: PCode, y: PCode): Code[Int] =
        Code.invokeStatic2[java.lang.Integer, Int, Int, Int]("compare", x.tcode[Int], y.tcode[Int])

      def _ltNonnull(cb: EmitCodeBuilder, x: PCode, y: PCode): Code[Boolean] = x.tcode[Int] < y.tcode[Int]

      def _lteqNonnull(cb: EmitCodeBuilder, x: PCode, y: PCode): Code[Boolean] = x.tcode[Int] <= y.tcode[Int]

      def _gtNonnull(cb: EmitCodeBuilder, x: PCode, y: PCode): Code[Boolean] = x.tcode[Int] > y.tcode[Int]

      def _gteqNonnull(cb: EmitCodeBuilder, x: PCode, y: PCode): Code[Boolean] = x.tcode[Int] >= y.tcode[Int]

      def _equivNonnull(cb: EmitCodeBuilder, x: PCode, y: PCode): Code[Boolean] = x.tcode[Int].ceq(y.tcode[Int])
    }
  }
}


object Int64Ordering {
  def make(t1: SInt64, t2: SInt64, ecb: EmitClassBuilder[_]): CodeOrdering = {
    new CodeOrdering {

      val type1: SInt64 = t1
      val type2: SInt64 = t2

      def _compareNonnull(cb: EmitCodeBuilder, x: PCode, y: PCode): Code[Int] =
        Code.invokeStatic2[java.lang.Long, Long, Long, Int]("compare", x.tcode[Long], y.tcode[Long])

      def _ltNonnull(cb: EmitCodeBuilder, x: PCode, y: PCode): Code[Boolean] = x.tcode[Long] < y.tcode[Long]

      def _lteqNonnull(cb: EmitCodeBuilder, x: PCode, y: PCode): Code[Boolean] = x.tcode[Long] <= y.tcode[Long]

      def _gtNonnull(cb: EmitCodeBuilder, x: PCode, y: PCode): Code[Boolean] = x.tcode[Long] > y.tcode[Long]

      def _gteqNonnull(cb: EmitCodeBuilder, x: PCode, y: PCode): Code[Boolean] = x.tcode[Long] >= y.tcode[Long]

      def _equivNonnull(cb: EmitCodeBuilder, x: PCode, y: PCode): Code[Boolean] = x.tcode[Long].ceq(y.tcode[Long])
    }
  }
}

object Float32Ordering {
  def make(t1: SFloat32, t2: SFloat32, ecb: EmitClassBuilder[_]): CodeOrdering = {
    new CodeOrdering {

      val type1: SFloat32 = t1
      val type2: SFloat32 = t2

      def _compareNonnull(cb: EmitCodeBuilder, x: PCode, y: PCode): Code[Int] =
        Code.invokeStatic2[java.lang.Float, Float, Float, Int]("compare", x.tcode[Float], y.tcode[Float])

      def _ltNonnull(cb: EmitCodeBuilder, x: PCode, y: PCode): Code[Boolean] = x.tcode[Float] < y.tcode[Float]

      def _lteqNonnull(cb: EmitCodeBuilder, x: PCode, y: PCode): Code[Boolean] = x.tcode[Float] <= y.tcode[Float]

      def _gtNonnull(cb: EmitCodeBuilder, x: PCode, y: PCode): Code[Boolean] = x.tcode[Float] > y.tcode[Float]

      def _gteqNonnull(cb: EmitCodeBuilder, x: PCode, y: PCode): Code[Boolean] = x.tcode[Float] >= y.tcode[Float]

      def _equivNonnull(cb: EmitCodeBuilder, x: PCode, y: PCode): Code[Boolean] = x.tcode[Float].ceq(y.tcode[Float])
    }
  }
}

object Float64Ordering {
  def make(t1: SFloat64, t2: SFloat64, ecb: EmitClassBuilder[_]): CodeOrdering = {
    new CodeOrdering {

      val type1: SFloat64 = t1
      val type2: SFloat64 = t2

      def _compareNonnull(cb: EmitCodeBuilder, x: PCode, y: PCode): Code[Int] =
        Code.invokeStatic2[java.lang.Double, Double, Double, Int]("compare", x.tcode[Double], y.tcode[Double])

      def _ltNonnull(cb: EmitCodeBuilder, x: PCode, y: PCode): Code[Boolean] = x.tcode[Double] < y.tcode[Double]

      def _lteqNonnull(cb: EmitCodeBuilder, x: PCode, y: PCode): Code[Boolean] = x.tcode[Double] <= y.tcode[Double]

      def _gtNonnull(cb: EmitCodeBuilder, x: PCode, y: PCode): Code[Boolean] = x.tcode[Double] > y.tcode[Double]

      def _gteqNonnull(cb: EmitCodeBuilder, x: PCode, y: PCode): Code[Boolean] = x.tcode[Double] >= y.tcode[Double]

      def _equivNonnull(cb: EmitCodeBuilder, x: PCode, y: PCode): Code[Boolean] = x.tcode[Double].ceq(y.tcode[Double])
    }
  }
}

object BooleanOrdering {
  def make(t1: SBoolean, t2: SBoolean, ecb: EmitClassBuilder[_]): CodeOrdering = {

    new CodeOrderingCompareConsistentWithOthers {
      val type1: SBoolean = t1
      val type2: SBoolean = t2

      def _compareNonnull(cb: EmitCodeBuilder, x: PCode, y: PCode): Code[Int] =
        Code.invokeStatic2[java.lang.Boolean, Boolean, Boolean, Int]("compare", x.tcode[Boolean], y.tcode[Boolean])
    }
  }
}
