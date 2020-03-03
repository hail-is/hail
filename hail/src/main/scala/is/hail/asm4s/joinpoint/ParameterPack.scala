package is.hail.asm4s.joinpoint

import is.hail.asm4s._
import is.hail.expr.ir
import is.hail.expr.ir.{EmitTriplet, PValue}
import is.hail.expr.types.physical.PType

trait ParameterPack[A] {
  def push(a: A): Code[Unit]

  def pushAny(a: Any): Code[Unit] = push(a.asInstanceOf[A])

  def newLocals(mb: MethodBuilder, name: String = null): ParameterStore[A]

  def newFields(fb: FunctionBuilder[_], name: String): ParameterStore[A]
}

case object ParameterPackUnit extends ParameterPack[Unit] {
  def push(u: Unit): Code[Unit] = Code._empty

  def newLocals(mb: MethodBuilder, name: String = null): ParameterStore[Unit] = ParameterStoreUnit

  def newFields(fb: FunctionBuilder[_], name: String): ParameterStore[Unit] = ParameterStoreUnit
}

case class ParameterPackCode[T](tti: TypeInfo[T]) extends ParameterPack[Code[T]] {
  def push(v: Code[T]): Code[Unit] = coerce[Unit](v)

  def newLocals(mb: MethodBuilder, name: String = null): ParameterStore[Code[T]] = {
    val x = mb.newLocal(name)(tti)
    new ParameterStore[Code[T]] {
      def storeInsn: Code[Unit] = x.storeInsn

      def store(a: Code[T]): Code[Unit] = x.store(a)

      def load: Code[T] = x.load()

      def init: Code[Unit] = x.storeAny(ir.defaultValue(tti))
    }
  }

  def newFields(fb: FunctionBuilder[_], name: String): ParameterStore[Code[T]] = {
    val x = fb.newField(name)(tti)
    new ParameterStore[Code[T]] {
      def storeInsn: Code[Unit] = throw new UnsupportedOperationException("storeInsn not supported for fields")

      def store(a: Code[T]): Code[Unit] = x.store(a)

      def load: Code[T] = x.load()

      def init: Code[Unit] = x.storeAny(ir.defaultValue(tti))
    }
  }
}

case class ParamPackTuple2[A, B](ap: ParameterPack[A], bp: ParameterPack[B]) extends ParameterPack[(A, B)] {
  def push(v: (A, B)): Code[Unit] = Code(ap.push(v._1), bp.push(v._2))

  def newLocals(mb: MethodBuilder, name: String = null): ParameterStore[(A, B)] =
    if (name == null)
      ParameterStoreTuple2(ap.newLocals(mb), bp.newLocals(mb))
    else
      ParameterStoreTuple2(ap.newLocals(mb, name + "_1"),
        bp.newLocals(mb, name + "_2"))

  def newFields(fb: FunctionBuilder[_], name: String): ParameterStore[(A, B)] =
    ParameterStoreTuple2(ap.newFields(fb, name + "_1"),
      bp.newFields(fb, name + "_2"))
}

case class ParamPackTuple3[A, B, C](ap: ParameterPack[A], bp: ParameterPack[B], cp: ParameterPack[C]) extends ParameterPack[(A, B, C)] {
  def push(v: (A, B, C)): Code[Unit] =
    Code(ap.push(v._1), bp.push(v._2), cp.push(v._3))

  def newLocals(mb: MethodBuilder, name: String = null): ParameterStore[(A, B, C)] =
    if (name == null)
      ParameterStoreTuple3(ap.newLocals(mb), bp.newLocals(mb), cp.newLocals(mb))
    else
      ParameterStoreTuple3(ap.newLocals(mb, name + "_1"),
        bp.newLocals(mb, name + "_2"),
        cp.newLocals(mb, name + "_3"))

  def newFields(fb: FunctionBuilder[_], name: String): ParameterStore[(A, B, C)] =
    ParameterStoreTuple3(ap.newFields(fb, name + "_1"),
      bp.newFields(fb, name + "_2"),
      cp.newFields(fb, name + "_3"))
}

case class ParamPackArray(pps: IndexedSeq[ParameterPack[_]]) extends ParameterPack[IndexedSeq[_]] {
  override def push(a: IndexedSeq[_]): Code[Unit] =
    pps.zip(a).foldLeft(Code._empty) { case (acc, (pp, v)) =>
      Code(acc, pp.pushAny(v))
    }

  override def newLocals(mb: MethodBuilder, name: String = null): ParameterStore[IndexedSeq[_]] =
    if (name == null) {
      ParameterStoreArray(pps.map(_.newLocals(mb)))
    } else {

      val locals = pps.zipWithIndex.map { case (pp, i) =>
        pp.newLocals(mb, name + s"_$i")
      }
      ParameterStoreArray(locals)
    }

  def newFields(fb: FunctionBuilder[_], name: String): ParameterStore[IndexedSeq[_]] = {
    val fields = pps.zipWithIndex.map { case (pp, i) =>
      pp.newFields(fb, name + s"_$i")
    }
    ParameterStoreArray(fields)
  }

  def newFields(fb: FunctionBuilder[_], names: IndexedSeq[String]): ParameterStoreArray = {
    val fields = pps.zip(names).map { case (pp, name) =>
      pp.newFields(fb, name)
    }
    ParameterStoreArray(fields)
  }
}

object ParameterPack {
  implicit val unit: ParameterPack[Unit] = ParameterPackUnit

  implicit def code[T](implicit tti: TypeInfo[T]): ParameterPack[Code[T]] = ParameterPackCode(tti)

  implicit def tuple2[A, B](
    implicit ap: ParameterPack[A],
    bp: ParameterPack[B]
  ): ParameterPack[(A, B)] = ParamPackTuple2(ap, bp)

  implicit def tuple3[A, B, C](
    implicit ap: ParameterPack[A],
    bp: ParameterPack[B],
    cp: ParameterPack[C]
  ): ParameterPack[(A, B, C)] = ParamPackTuple3(ap, bp, cp)


  def array(pps: IndexedSeq[ParameterPack[_]]): ParamPackArray = ParamPackArray(pps)

  def let[A: ParameterPack, X](mb: MethodBuilder, a0: A)(k: A => Code[X]): Code[X] = {
    val ap = implicitly[ParameterPack[A]]
    val as = ap.newLocals(mb)
    Code(ap.push(a0), as.storeInsn, k(as.load))
  }
}


abstract class ParameterStore[A] {
  private[joinpoint] def storeInsn: Code[Unit]

  def load: A

  def store(v: A): Code[Unit]

  def :=(v: A): Code[Unit] = {
    store(v)
  }

  def :=(cc: JoinPoint.CallCC[A]): Code[Unit] = Code(cc.code, storeInsn)

  def storeAny(v: Any): Code[Unit] = store(v.asInstanceOf[A])

  def init: Code[Unit]
}

case object ParameterStoreUnit extends ParameterStore[Unit] {
  def storeInsn: Code[Unit] = Code._empty

  def store(v: Unit): Code[Unit] = Code._empty

  def load: Unit = ()

  def init: Code[Unit] = Code._empty
}

case class ParameterStoreTuple2[A, B](pa: ParameterStore[A], pb: ParameterStore[B]) extends ParameterStore[(A, B)] {
  def load: (A, B) = (pa.load, pb.load)

  def store(v: (A, B)): Code[Unit] = v match {
    case (a, b) => Code(pa.store(a), pb.store(b))
  }

  def storeInsn: Code[Unit] = Code(pb.storeInsn, pa.storeInsn)

  def init: Code[Unit] = Code(pa.init, pb.init)
}

case class ParameterStoreTuple3[A, B, C](
  pa: ParameterStore[A],
  pb: ParameterStore[B],
  pc: ParameterStore[C]
) extends ParameterStore[(A, B, C)] {
  def load: (A, B, C) = (pa.load, pb.load, pc.load)

  def store(v: (A, B, C)): Code[Unit] = v match {
    case (a, b, c) => Code(pa.store(a), pb.store(b), pc.store(c))
  }

  def storeInsn: Code[Unit] = Code(pc.storeInsn, pb.storeInsn, pa.storeInsn)

  def init: Code[Unit] = Code(pa.init, pb.init, pc.init)

}

case class ParameterStoreArray(pss: IndexedSeq[ParameterStore[_]]) extends ParameterStore[IndexedSeq[_]] {
  def load: IndexedSeq[_] = pss.map(_.load)

  def store(vs: IndexedSeq[_]): Code[Unit] = {
    assert(pss.length == vs.length)
    pss.zip(vs).foldLeft(Code._empty) { case (acc, (ps, v)) => Code(acc, ps.storeAny(v)) }
  }

  def storeInsn: Code[Unit] = pss.map(_.storeInsn).fold(Code._empty) { case (acc, c) => Code(c, acc) } // order of c and acc is important

  def init: Code[Unit] = pss.foldLeft[Code[Unit]](Code._empty) { case (acc, ps) => Code(acc, ps.init) }

}

case class ParameterPackTriplet[P](t: PType) extends ParameterPack[TypedTriplet[P]] {
  val ti: TypeInfo[P] = ir.typeToTypeInfo(t).asInstanceOf[TypeInfo[P]]
  val ppm = implicitly[ParameterPack[Code[Boolean]]]
  val ppv = ParameterPack.code[P](ti).asInstanceOf[ParameterPack[Code[_]]]

  def push(trip: TypedTriplet[P]): Code[Unit] = Code(
    trip.setup,
    trip.m.mux(
      Code(coerce[Unit](ir.defaultValue(ti)), coerce[Unit](const(true))),
      Code(coerce[Unit](trip.v), coerce[Unit](const(false)))))

  def newLocals(mb: MethodBuilder, name: String = null): ParameterStoreTriplet[P] = {
    val psm = ppm.newLocals(mb, if (name == null) "m" else s"${ name }_missing")
    val psv = ppv.newLocals(mb, if (name == null) "v" else name)
    ParameterStoreTriplet[P](t, psm, psv)
  }

  def newFields(fb: FunctionBuilder[_], name: String): ParameterStoreTriplet[P] = {
    val psm = ppm.newFields(fb, if (name == null) "m" else s"${ name }_missing")
    val psv = ppv.newFields(fb, if (name == null) "v" else name)
    ParameterStoreTriplet[P](t, psm, psv)
  }
}

case class ParameterStoreTriplet[A](t: PType, psm: ParameterStore[Code[Boolean]], psv: ParameterStore[Code[_]]) extends ParameterStore[TypedTriplet[A]] {
  val ti: TypeInfo[A] = ir.typeToTypeInfo(t).asInstanceOf[TypeInfo[A]]
  def load: TypedTriplet[A] = TypedTriplet[A](t, Code._empty, psm.load, psv.load)

  def store(trip: TypedTriplet[A]): Code[Unit] = Code(
    trip.setup,
    trip.m.mux(
    Code(psm.store(true), psv.storeAny(ir.defaultValue(ti))),
    Code(psm.store(false), psv.storeAny(trip.v))))

  def storeInsn: Code[Unit] = ParameterStoreTuple2(psv, psm).storeInsn

  def init: Code[Unit] = Code(psm.init, psv.init)
}

object TypedTriplet {
  def apply(t: PType, et: EmitTriplet): TypedTriplet[t.type] =
    TypedTriplet(et.setup, et.m, et.pv)

  def apply[A](t: PType, setup: Code[Unit], m: Code[Boolean], v: Code[_]): TypedTriplet[A] =
    TypedTriplet(setup, m, PValue(t, v))

  def missing(t: PType): TypedTriplet[t.type] =
    TypedTriplet(t, EmitTriplet(Code._empty, true, t.defaultValue))

  def pack(t: PType): ParameterPackTriplet[t.type] = ParameterPackTriplet(t)
}

case class TypedTriplet[P] private(setup: Code[Unit], m: Code[Boolean], pv: PValue) {
  def v: Code[_] = pv.code

  def untyped: EmitTriplet = EmitTriplet(setup, m, pv)
}
