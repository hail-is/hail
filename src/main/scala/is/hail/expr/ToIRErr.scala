package is.hail.expr

import scala.reflect.{ClassTag, classTag}

object ToIRErr {
  def all[T](actions: Seq[ToIRErr[T]]): ToIRErr[Seq[T]] =
    actions.foldLeft[ToIRErr[Seq[T]]](success(Seq[T]())) {
      case (ToIRSuccess(xs), ToIRSuccess(x)) => success(x +: xs)
      case (_: ToIRSuccess[_], f: ToIRFailure[_]) => f.as[Seq[T]]
      case (f: ToIRFailure[_], _: ToIRSuccess[_]) => f
      case (ToIRFailure(irs), ToIRFailure(moreIrs)) => fail(irs ++ moreIrs)
    }
  def all[T, U](a: ToIRErr[T], b: ToIRErr[U]): ToIRErr[(T, U)] =
    (a, b) match {
      case (ToIRSuccess(t), ToIRSuccess(u)) => success(t, u)
      case (_: ToIRSuccess[_], f: ToIRFailure[_]) => f.as[(T, U)]
      case (f: ToIRFailure[_], _: ToIRSuccess[_]) => f.as[(T, U)]
      case (ToIRFailure(irs), ToIRFailure(moreIrs)) => fail(irs ++ moreIrs)
    }
  def success[T](t: T): ToIRErr[T] = ToIRSuccess(t)
  def fail[T](ir: AST, message: String = null): ToIRErr[T] =
    ToIRFailure(Seq((ir, message)))
  def fail[T](irs: Seq[(AST, String)]): ToIRErr[T] = ToIRFailure(irs)
  def whenOfType[T: ClassTag](a: AST): ToIRErr[T] =
    a.`type` match {
      case t: T => success(t)
      case _ =>
        fail(a, s"${a.`type`} should be a subtype of ${classTag[T].runtimeClass.getSimpleName}")
    }
  def fromOption[T](blame: AST, message: String, ot: Option[T]): ToIRErr[T] = ot match {
    case Some(t) => success(t)
    case None => fail(blame, message)
  }
}

trait ToIRErr[T] {
  def map[U](f: T => U): ToIRErr[U]
  def flatMap[U](f: T => ToIRErr[U]): ToIRErr[U]
  def filter(p: T => Boolean): ToIRErr[T]
}

case class ToIRSuccess[T](value: T) extends ToIRErr[T] {
  def map[U](f: T => U): ToIRErr[U] =
    ToIRSuccess(f(value))
  def flatMap[U](f: T => ToIRErr[U]): ToIRErr[U] =
    f(value)
  def filter(p: T => Boolean): ToIRErr[T] =
    // heh, no IR to blame here
    //
    // better solution possible if Scala for notation allowed actions that return unit, a la:
    // for {
    //   x <- rhs.toIR
    //   blameUnless(blamedIr, message, x.isInstanceOf[T])
    // } yield x
    //
    // only current option is
    //
    // for {
    //   x <- rhs.toIR
    //   _ <- blameUnless(...)
    // } yield x
    //
    if (p(value)) this
    else ToIRErr.fail(null, Thread.currentThread().getStackTrace()(2).toString())
}

case class ToIRFailure[T](incovertible: Seq[(AST, String)]) extends ToIRErr[T] {
  // as is safe because this class contains no `T`s
  def as[U]: ToIRErr[U] = this.asInstanceOf[ToIRErr[U]]
  def map[U](f: T => U): ToIRErr[U] = as[U]
  def flatMap[U](f: T => ToIRErr[U]): ToIRErr[U] = as[U]
  def filter(p: T => Boolean): ToIRErr[T] = this
}
