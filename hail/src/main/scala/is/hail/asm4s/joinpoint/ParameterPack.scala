package is.hail.asm4s.joinpoint

import is.hail.asm4s._

object ParameterPack {
  implicit val unit: ParameterPack[Unit] = new ParameterPack[Unit] {
    def push(u: Unit) = Code._empty
    def store(mb: MethodBuilder) = ParameterStore[Unit](Code._empty, ())
  }

  implicit def code[T](implicit tti: TypeInfo[T]): ParameterPack[Code[T]] =
    new ParameterPack[Code[T]] {
      def push(v: Code[T]): Code[Unit] = coerce[Unit](v)
      def store(mb: MethodBuilder): ParameterStore[Code[T]] = {
        val x = mb.newLocal(tti)
        ParameterStore(x.storeInsn, x.load)
      }
    }

  implicit def tuple2[A, B](implicit ap: ParameterPack[A], bp: ParameterPack[B]): ParameterPack[(A, B)] =
    new ParameterPack[(A, B)] {
      def push(v: (A, B)) = Code(ap.push(v._1), bp.push(v._2))
      def store(mb: MethodBuilder) = {
        val as = ap.store(mb)
        val bs = bp.store(mb)
        ParameterStore(Code(bs.pop, as.pop), (as.load, bs.load))
      }
    }

  implicit def tuple3[A, B, C](
    implicit ap: ParameterPack[A],
    bp: ParameterPack[B],
    cp: ParameterPack[C]
  ): ParameterPack[(A, B, C)] = new ParameterPack[(A, B, C)] {
    def push(v: (A, B, C)) = Code(ap.push(v._1), bp.push(v._2), cp.push(v._3))
    def store(mb: MethodBuilder) = {
      val as = ap.store(mb)
      val bs = bp.store(mb)
      val cs = cp.store(mb)
      ParameterStore(Code(cs.pop, bs.pop, as.pop), (as.load, bs.load, cs.load))
    }
  }
}

trait ParameterPack[A] {
  def push(a: A): Code[Unit]
  def store(mb: MethodBuilder): ParameterStore[A]
}

case class ParameterStore[A](
  pop: Code[Unit],
  load: A
)
