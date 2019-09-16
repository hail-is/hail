package is.hail.asm4s.joinpoint

import is.hail.asm4s._

object ParameterPack {
  implicit val unit: ParameterPack[Unit] = new ParameterPack[Unit] {
    def push(u: Unit): Code[Unit] = Code._empty
    def newLocals(mb: MethodBuilder): ParameterStore[Unit] = ParameterStore.unit
  }

  implicit def code[T](implicit tti: TypeInfo[T]): ParameterPack[Code[T]] =
    new ParameterPack[Code[T]] {
      def push(v: Code[T]): Code[Unit] = coerce[Unit](v)
      def newLocals(mb: MethodBuilder): ParameterStore[Code[T]] = {
        val x = mb.newLocal(tti)
        ParameterStore(x.storeInsn, x.load)
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
  }
}

object ParameterStore {
  def unit: ParameterStore[Unit] = ParameterStore(Code._empty, ())
}

trait ParameterPack[A] {
  def push(a: A): Code[Unit]
  def newLocals(mb: MethodBuilder): ParameterStore[A]
}

case class ParameterStore[A](
  store: Code[Unit],
  load: A
)
