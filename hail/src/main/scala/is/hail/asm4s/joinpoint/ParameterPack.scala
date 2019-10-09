package is.hail.asm4s.joinpoint

import is.hail.asm4s._
import is.hail.expr.ir
import is.hail.expr.ir.EmitTriplet
import is.hail.expr.types.physical.PType

object ParameterPack {
  implicit val unit: ParameterPack[Unit] = new ParameterPack[Unit] {
    def push(u: Unit): Code[Unit] = Code._empty
    def newLocals(mb: MethodBuilder): ParameterStore[Unit] = ParameterStore.unit
    def newFields(fb: FunctionBuilder[_], name: String): (Unit => Code[Unit], Unit) =
      (_ => Code._empty, ())
  }

  implicit def code[T](implicit tti: TypeInfo[T]): ParameterPack[Code[T]] =
    new ParameterPack[Code[T]] {
      def push(v: Code[T]): Code[Unit] = coerce[Unit](v)
      def newLocals(mb: MethodBuilder): ParameterStore[Code[T]] = {
        val x = mb.newLocal(tti)
        ParameterStore(x.storeInsn, x.load)
      }
      def newFields(fb: FunctionBuilder[_], name: String): (Code[T] => Code[Unit], Code[T]) = {
        val x = fb.newField(name)(tti)
        (x := _, x.load)
      }
    }

  implicit def tuple2[A, B](
    implicit ap: ParameterPack[A],
    bp: ParameterPack[B]
  ): ParameterPack[(A, B)] = new ParameterPack[(A, B)] {
    def push(v: (A, B)): Code[Unit] = Code(ap.push(v._1), bp.push(v._2))
    def newLocals(mb: MethodBuilder): ParameterStore[(A, B)] = {
      val as = ap.newLocals(mb)
      val bs = bp.newLocals(mb)
      ParameterStore(Code(bs.store, as.store), (as.load, bs.load))
    }
    def newFields(fb: FunctionBuilder[_], name: String): (((A, B)) => Code[Unit], (A, B)) = {
      val (setA, a) = ap.newFields(fb, name + "_1")
      val (setB, b) = bp.newFields(fb, name + "_2")
      ({ case (a, b) => Code(setA(a), setB(b)) }, (a, b))
    }
  }

  implicit def tuple3[A, B, C](
    implicit ap: ParameterPack[A],
    bp: ParameterPack[B],
    cp: ParameterPack[C]
  ): ParameterPack[(A, B, C)] = new ParameterPack[(A, B, C)] {
    def push(v: (A, B, C)): Code[Unit] = Code(ap.push(v._1), bp.push(v._2), cp.push(v._3))
    def newLocals(mb: MethodBuilder): ParameterStore[(A, B, C)] = {
      val as = ap.newLocals(mb)
      val bs = bp.newLocals(mb)
      val cs = cp.newLocals(mb)
      ParameterStore(Code(cs.store, bs.store, as.store), (as.load, bs.load, cs.load))
    }
    def newFields(fb: FunctionBuilder[_], name: String): (((A, B, C)) => Code[Unit], (A, B, C)) = {
      val (setA, a) = ap.newFields(fb, name + "_1")
      val (setB, b) = bp.newFields(fb, name + "_2")
      val (setC, c) = cp.newFields(fb, name + "_3")
      ({ case (a, b, c) => Code(setA(a), setB(b), setC(c)) }, (a, b, c))
    }
  }

  def let[A: ParameterPack, X](mb: MethodBuilder, a0: A)(k: A => Code[X]): Code[X] = {
    val ap = implicitly[ParameterPack[A]]
    val as = ap.newLocals(mb)
    Code(ap.push(a0), as.store, k(as.load))
  }
}

object ParameterStore {
  def unit: ParameterStore[Unit] = ParameterStore(Code._empty, ())
}

trait ParameterPack[A] {
  def push(a: A): Code[Unit]
  def newLocals(mb: MethodBuilder): ParameterStore[A]
  def newFields(fb: FunctionBuilder[_], name: String): (A => Code[Unit], A)
}

case class ParameterStore[A](
  store: Code[Unit],
  load: A
) {
  def :=(v: A)(implicit p: ParameterPack[A]): Code[Unit] =
    Code(p.push(v), store)

  def :=(cc: JoinPoint.CallCC[A]): Code[Unit] =
    Code(cc.code, store)
}

object TypedTriplet {
  def apply(t: PType, et: EmitTriplet): TypedTriplet[t.type] =
    TypedTriplet(et.setup, et.m, et.v)

  def missing(t: PType): TypedTriplet[t.type] =
    TypedTriplet(t, EmitTriplet(Code._empty, true, ir.defaultValue(t)))

  class Pack[P] private[joinpoint](t: PType) extends ParameterPack[TypedTriplet[P]] {
    def push(trip: TypedTriplet[P]): Code[Unit] = Code(
      trip.setup,
      trip.m.mux(
        Code(coerce[Unit](ir.defaultValue(t)), coerce[Unit](const(true))),
        Code(coerce[Unit](trip.v), coerce[Unit](const(false)))))

    def newLocals(mb: MethodBuilder): ParameterStore[TypedTriplet[P]] = {
      val m = mb.newLocal[Boolean]("m")
      val v = mb.newLocal("v")(ir.typeToTypeInfo(t))
      ParameterStore(Code(m.storeInsn, v.storeInsn), TypedTriplet(Code._empty, m, v))
    }

    def newFields(fb: FunctionBuilder[_], name: String): ((TypedTriplet[P] => Code[Unit]), TypedTriplet[P]) = {
      val m = fb.newField[Boolean](name + "_missing")
      val v = fb.newField(name)(ir.typeToTypeInfo(t))
      def set(trip: TypedTriplet[P]): Code[Unit] =
        Code(trip.setup,
          trip.m.mux(
            m := true,
            Code(m := false, v.storeAny(trip.v))))
      (set, TypedTriplet(Code._empty, m, v))
    }
  }

  def pack(t: PType): Pack[t.type] = new Pack(t)
}

case class TypedTriplet[P] private(setup: Code[Unit], m: Code[Boolean], v: Code[_]) {
  def untyped: EmitTriplet = EmitTriplet(setup, m, v)
}
