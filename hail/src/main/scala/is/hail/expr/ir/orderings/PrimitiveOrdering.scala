package is.hail.expr.ir.orderings

import is.hail.asm4s.Code
import is.hail.expr.ir.{EmitClassBuilder, EmitCodeBuilder}
import is.hail.types.physical.stypes.SCode
import is.hail.types.physical.stypes.primitives._

object Int32Ordering {
  def make(t1: SInt32.type, t2: SInt32.type, ecb: EmitClassBuilder[_]): CodeOrdering = {
    new CodeOrdering {

      val type1: SInt32.type = t1
      val type2: SInt32.type = t2

      def _compareNonnull(cb: EmitCodeBuilder, x: SCode, y: SCode): Code[Int] =
        Code.invokeStatic2[java.lang.Integer, Int, Int, Int]("compare", x.asInt.intCode(cb), y.asInt.intCode(cb))

      def _ltNonnull(cb: EmitCodeBuilder, x: SCode, y: SCode): Code[Boolean] = x.asInt.intCode(cb) < y.asInt.intCode(cb)

      def _lteqNonnull(cb: EmitCodeBuilder, x: SCode, y: SCode): Code[Boolean] = x.asInt.intCode(cb) <= y.asInt.intCode(cb)

      def _gtNonnull(cb: EmitCodeBuilder, x: SCode, y: SCode): Code[Boolean] = x.asInt.intCode(cb) > y.asInt.intCode(cb)

      def _gteqNonnull(cb: EmitCodeBuilder, x: SCode, y: SCode): Code[Boolean] = x.asInt.intCode(cb) >= y.asInt.intCode(cb)

      def _equivNonnull(cb: EmitCodeBuilder, x: SCode, y: SCode): Code[Boolean] = x.asInt.intCode(cb).ceq(y.asInt.intCode(cb))
    }
  }
}


object Int64Ordering {
  def make(t1: SInt64.type, t2: SInt64.type, ecb: EmitClassBuilder[_]): CodeOrdering = {
    new CodeOrdering {

      val type1: SInt64.type = t1
      val type2: SInt64.type = t2

      def _compareNonnull(cb: EmitCodeBuilder, x: SCode, y: SCode): Code[Int] =
        Code.invokeStatic2[java.lang.Long, Long, Long, Int]("compare", x.asLong.longCode(cb), y.asLong.longCode(cb))

      def _ltNonnull(cb: EmitCodeBuilder, x: SCode, y: SCode): Code[Boolean] = x.asLong.longCode(cb) < y.asLong.longCode(cb)

      def _lteqNonnull(cb: EmitCodeBuilder, x: SCode, y: SCode): Code[Boolean] = x.asLong.longCode(cb) <= y.asLong.longCode(cb)

      def _gtNonnull(cb: EmitCodeBuilder, x: SCode, y: SCode): Code[Boolean] = x.asLong.longCode(cb) > y.asLong.longCode(cb)

      def _gteqNonnull(cb: EmitCodeBuilder, x: SCode, y: SCode): Code[Boolean] = x.asLong.longCode(cb) >= y.asLong.longCode(cb)

      def _equivNonnull(cb: EmitCodeBuilder, x: SCode, y: SCode): Code[Boolean] = x.asLong.longCode(cb).ceq(y.asLong.longCode(cb))
    }
  }
}

object Float32Ordering {
  def make(t1: SFloat32.type, t2: SFloat32.type, ecb: EmitClassBuilder[_]): CodeOrdering = {
    new CodeOrdering {

      val type1: SFloat32.type = t1
      val type2: SFloat32.type = t2

      def _compareNonnull(cb: EmitCodeBuilder, x: SCode, y: SCode): Code[Int] =
        Code.invokeStatic2[java.lang.Float, Float, Float, Int]("compare", x.asFloat.floatCode(cb), y.asFloat.floatCode(cb))

      def _ltNonnull(cb: EmitCodeBuilder, x: SCode, y: SCode): Code[Boolean] = x.asFloat.floatCode(cb) < y.asFloat.floatCode(cb)

      def _lteqNonnull(cb: EmitCodeBuilder, x: SCode, y: SCode): Code[Boolean] = x.asFloat.floatCode(cb) <= y.asFloat.floatCode(cb)

      def _gtNonnull(cb: EmitCodeBuilder, x: SCode, y: SCode): Code[Boolean] = x.asFloat.floatCode(cb) > y.asFloat.floatCode(cb)

      def _gteqNonnull(cb: EmitCodeBuilder, x: SCode, y: SCode): Code[Boolean] = x.asFloat.floatCode(cb) >= y.asFloat.floatCode(cb)

      def _equivNonnull(cb: EmitCodeBuilder, x: SCode, y: SCode): Code[Boolean] = x.asFloat.floatCode(cb).ceq(y.asFloat.floatCode(cb))
    }
  }
}

object Float64Ordering {
  def make(t1: SFloat64.type, t2: SFloat64.type, ecb: EmitClassBuilder[_]): CodeOrdering = {
    new CodeOrdering {

      val type1: SFloat64.type = t1
      val type2: SFloat64.type = t2

      def _compareNonnull(cb: EmitCodeBuilder, x: SCode, y: SCode): Code[Int] =
        Code.invokeStatic2[java.lang.Double, Double, Double, Int]("compare", x.asDouble.doubleCode(cb), y.asDouble.doubleCode(cb))

      def _ltNonnull(cb: EmitCodeBuilder, x: SCode, y: SCode): Code[Boolean] = x.asDouble.doubleCode(cb) < y.asDouble.doubleCode(cb)

      def _lteqNonnull(cb: EmitCodeBuilder, x: SCode, y: SCode): Code[Boolean] = x.asDouble.doubleCode(cb) <= y.asDouble.doubleCode(cb)

      def _gtNonnull(cb: EmitCodeBuilder, x: SCode, y: SCode): Code[Boolean] = x.asDouble.doubleCode(cb) > y.asDouble.doubleCode(cb)

      def _gteqNonnull(cb: EmitCodeBuilder, x: SCode, y: SCode): Code[Boolean] = x.asDouble.doubleCode(cb) >= y.asDouble.doubleCode(cb)

      def _equivNonnull(cb: EmitCodeBuilder, x: SCode, y: SCode): Code[Boolean] = x.asDouble.doubleCode(cb).ceq(y.asDouble.doubleCode(cb))
    }
  }
}

object BooleanOrdering {
  def make(t1: SBoolean.type, t2: SBoolean.type, ecb: EmitClassBuilder[_]): CodeOrdering = {

    new CodeOrderingCompareConsistentWithOthers {
      val type1: SBoolean.type = t1
      val type2: SBoolean.type = t2

      def _compareNonnull(cb: EmitCodeBuilder, x: SCode, y: SCode): Code[Int] =
        Code.invokeStatic2[java.lang.Boolean, Boolean, Boolean, Int]("compare", x.asBoolean.boolCode(cb), y.asBoolean.boolCode(cb))
    }
  }
}
