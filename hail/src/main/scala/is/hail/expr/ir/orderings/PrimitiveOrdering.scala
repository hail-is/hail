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
        Code.invokeStatic2[java.lang.Integer, Int, Int, Int]("compare", x.asInt.intCode(cb), y.asInt.intCode(cb))

      def _ltNonnull(cb: EmitCodeBuilder, x: PCode, y: PCode): Code[Boolean] = x.asInt.intCode(cb) < y.asInt.intCode(cb)

      def _lteqNonnull(cb: EmitCodeBuilder, x: PCode, y: PCode): Code[Boolean] = x.asInt.intCode(cb) <= y.asInt.intCode(cb)

      def _gtNonnull(cb: EmitCodeBuilder, x: PCode, y: PCode): Code[Boolean] = x.asInt.intCode(cb) > y.asInt.intCode(cb)

      def _gteqNonnull(cb: EmitCodeBuilder, x: PCode, y: PCode): Code[Boolean] = x.asInt.intCode(cb) >= y.asInt.intCode(cb)

      def _equivNonnull(cb: EmitCodeBuilder, x: PCode, y: PCode): Code[Boolean] = x.asInt.intCode(cb).ceq(y.asInt.intCode(cb))
    }
  }
}


object Int64Ordering {
  def make(t1: SInt64, t2: SInt64, ecb: EmitClassBuilder[_]): CodeOrdering = {
    new CodeOrdering {

      val type1: SInt64 = t1
      val type2: SInt64 = t2

      def _compareNonnull(cb: EmitCodeBuilder, x: PCode, y: PCode): Code[Int] =
        Code.invokeStatic2[java.lang.Long, Long, Long, Int]("compare", x.asLong.longCode(cb), y.asLong.longCode(cb))

      def _ltNonnull(cb: EmitCodeBuilder, x: PCode, y: PCode): Code[Boolean] = x.asLong.longCode(cb) < y.asLong.longCode(cb)

      def _lteqNonnull(cb: EmitCodeBuilder, x: PCode, y: PCode): Code[Boolean] = x.asLong.longCode(cb) <= y.asLong.longCode(cb)

      def _gtNonnull(cb: EmitCodeBuilder, x: PCode, y: PCode): Code[Boolean] = x.asLong.longCode(cb) > y.asLong.longCode(cb)

      def _gteqNonnull(cb: EmitCodeBuilder, x: PCode, y: PCode): Code[Boolean] = x.asLong.longCode(cb) >= y.asLong.longCode(cb)

      def _equivNonnull(cb: EmitCodeBuilder, x: PCode, y: PCode): Code[Boolean] = x.asLong.longCode(cb).ceq(y.asLong.longCode(cb))
    }
  }
}

object Float32Ordering {
  def make(t1: SFloat32, t2: SFloat32, ecb: EmitClassBuilder[_]): CodeOrdering = {
    new CodeOrdering {

      val type1: SFloat32 = t1
      val type2: SFloat32 = t2

      def _compareNonnull(cb: EmitCodeBuilder, x: PCode, y: PCode): Code[Int] =
        Code.invokeStatic2[java.lang.Float, Float, Float, Int]("compare", x.asFloat.floatCode(cb), y.asFloat.floatCode(cb))

      def _ltNonnull(cb: EmitCodeBuilder, x: PCode, y: PCode): Code[Boolean] = x.asFloat.floatCode(cb) < y.asFloat.floatCode(cb)

      def _lteqNonnull(cb: EmitCodeBuilder, x: PCode, y: PCode): Code[Boolean] = x.asFloat.floatCode(cb) <= y.asFloat.floatCode(cb)

      def _gtNonnull(cb: EmitCodeBuilder, x: PCode, y: PCode): Code[Boolean] = x.asFloat.floatCode(cb) > y.asFloat.floatCode(cb)

      def _gteqNonnull(cb: EmitCodeBuilder, x: PCode, y: PCode): Code[Boolean] = x.asFloat.floatCode(cb) >= y.asFloat.floatCode(cb)

      def _equivNonnull(cb: EmitCodeBuilder, x: PCode, y: PCode): Code[Boolean] = x.asFloat.floatCode(cb).ceq(y.asFloat.floatCode(cb))
    }
  }
}

object Float64Ordering {
  def make(t1: SFloat64, t2: SFloat64, ecb: EmitClassBuilder[_]): CodeOrdering = {
    new CodeOrdering {

      val type1: SFloat64 = t1
      val type2: SFloat64 = t2

      def _compareNonnull(cb: EmitCodeBuilder, x: PCode, y: PCode): Code[Int] =
        Code.invokeStatic2[java.lang.Double, Double, Double, Int]("compare", x.asDouble.doubleCode(cb), y.asDouble.doubleCode(cb))

      def _ltNonnull(cb: EmitCodeBuilder, x: PCode, y: PCode): Code[Boolean] = x.asDouble.doubleCode(cb) < y.asDouble.doubleCode(cb)

      def _lteqNonnull(cb: EmitCodeBuilder, x: PCode, y: PCode): Code[Boolean] = x.asDouble.doubleCode(cb) <= y.asDouble.doubleCode(cb)

      def _gtNonnull(cb: EmitCodeBuilder, x: PCode, y: PCode): Code[Boolean] = x.asDouble.doubleCode(cb) > y.asDouble.doubleCode(cb)

      def _gteqNonnull(cb: EmitCodeBuilder, x: PCode, y: PCode): Code[Boolean] = x.asDouble.doubleCode(cb) >= y.asDouble.doubleCode(cb)

      def _equivNonnull(cb: EmitCodeBuilder, x: PCode, y: PCode): Code[Boolean] = x.asDouble.doubleCode(cb).ceq(y.asDouble.doubleCode(cb))
    }
  }
}

object BooleanOrdering {
  def make(t1: SBoolean, t2: SBoolean, ecb: EmitClassBuilder[_]): CodeOrdering = {

    new CodeOrderingCompareConsistentWithOthers {
      val type1: SBoolean = t1
      val type2: SBoolean = t2

      def _compareNonnull(cb: EmitCodeBuilder, x: PCode, y: PCode): Code[Int] =
        Code.invokeStatic2[java.lang.Boolean, Boolean, Boolean, Int]("compare", x.asBoolean.boolCode(cb), y.asBoolean.boolCode(cb))
    }
  }
}
