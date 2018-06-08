package is.hail.expr

import scala.reflect.{ClassTag, classTag}

object ToIRErr {
  def all[T](actions: Seq[ToIRErr[T]]): ToIRErr[Seq[T]] =
    actions.foldLeft[ToIRErr[Seq[T]]](success(Seq[T]())) {
      case (ToIRSuccess(xs), ToIRSuccess(x)) => success(x +: xs)
      case (_: ToIRSuccess[_], f: ToIRFailure[_]) => f.as[Seq[T]]
      case (f: ToIRFailure[_], _: ToIRSuccess[_]) => f
      case (ToIRFailure(irs), ToIRFailure(moreIrs)) => fail(irs ++ moreIrs)
    }.map(_.reverse)
  def all[T, U](a: ToIRErr[T], b: ToIRErr[U]): ToIRErr[(T, U)] =
    (a, b) match {
      case (ToIRSuccess(t), ToIRSuccess(u)) => success(t, u)
      case (_: ToIRSuccess[_], f: ToIRFailure[_]) => f.as[(T, U)]
      case (f: ToIRFailure[_], _: ToIRSuccess[_]) => f.as[(T, U)]
      case (ToIRFailure(irs), ToIRFailure(moreIrs)) => fail(irs ++ moreIrs)
    }

  def orElse[T](a: ToIRErr[T], b: ToIRErr[T]): ToIRErr[T] =
    (a, b) match {
      case (ToIRSuccess(t), _) => success(t)
      case (ToIRFailure(_), ToIRSuccess(u)) => success(u)
      case (ToIRFailure(t), ToIRFailure(u)) => fail(t ++ u)
    }

  def success[T](t: T): ToIRErr[T] =
    ToIRSuccess(t)
  def fail[T](ir: AST): ToIRErr[T] =
    ToIRFailure(Seq((ir, null, getCaller())))
  def fail[T](ir: AST, message: String): ToIRErr[T] =
    ToIRFailure(Seq((ir, message, getCaller())))
  def fail[T](ir: AST, message: String, blame: StackTraceElement): ToIRErr[T] =
    ToIRFailure(Seq((ir, message, blame)))
  def fail[T](irs: Seq[(AST, String, StackTraceElement)]): ToIRErr[T] =
    ToIRFailure(irs)
  def whenOfType[T: ClassTag](a: AST): ToIRErr[T] =
    a.`type` match {
      case t: T => success(t)
      case _ =>
        fail(a, s"${a.`type`} should be a subtype of ${classTag[T].runtimeClass.getSimpleName}", getCaller())
    }
  def fromOption[T](blame: AST, message: String, ot: Option[T]): ToIRErr[T] = ot match {
    case Some(t) => success(t)
    case None => fail(blame, message, getCaller())
  }
  def blameWhen(blame: AST, message: String, condition: Boolean): ToIRErr[Unit] =
    if (condition) fail(blame, message, getCaller()) else success(())
  private[expr] def getCaller(): StackTraceElement =
    Thread.currentThread().getStackTrace()(3)

  def exactlyOne[T](
    blame: AST,
    a: Option[T],
    b: Option[T],
    c: Option[T]
  ): ToIRErr[T] = (a, b, c) match {
    case (Some(x), None, None) => success(x)
    case (None, Some(x), None) => success(x)
    case (None, None, Some(x)) => success(x)
    case (x, y, z) => fail(blame, s"expected exactly one of $x, $y, $z to be Some.", getCaller())
  }
}

trait ToIRErr[T] {
  def map[U](f: T => U): ToIRErr[U]
  def flatMap[U](f: T => ToIRErr[U]): ToIRErr[U]
  def toOption: Option[T]
}

case class ToIRSuccess[T](value: T) extends ToIRErr[T] {
  def map[U](f: T => U): ToIRErr[U] =
    ToIRSuccess(f(value))
  def flatMap[U](f: T => ToIRErr[U]): ToIRErr[U] =
    f(value)
  def toOption: Option[T] = Some(value)
}

case class ToIRFailure[T](incovertible: Seq[(AST, String, StackTraceElement)]) extends ToIRErr[T] {
  // as is safe because this class contains no `T`s
  def as[U]: ToIRErr[U] = this.asInstanceOf[ToIRErr[U]]
  def map[U](f: T => U): ToIRErr[U] = as[U]
  def flatMap[U](f: T => ToIRErr[U]): ToIRErr[U] = as[U]
  def toOption: Option[T] = None
}
