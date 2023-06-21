package is.hail.expr.ir.lowering

import cats.mtl.Ask
import cats.syntax.all._
import cats.{ApplicativeThrow, Monad, MonadThrow}
import is.hail.annotations.Region
import is.hail.asm4s.HailClassLoader
import is.hail.backend.local.LocalTaskContext
import is.hail.backend.{ExecuteContext, HailTaskContext}
import is.hail.expr.ir.{BaseIR, Pretty}
import is.hail.io.fs.FS
import is.hail.utils.HailException

import scala.language.{higherKinds, implicitConversions}
import scala.util.Try

object utils {
  def scopedExecution[M[_], A](use: (HailClassLoader, FS, HailTaskContext, Region) => M[A])
                              (implicit A: Ask[M, ExecuteContext], M: MonadThrow[M]): M[A] =
    for {
      ctx <- A.ask
      taskContext = new LocalTaskContext(0, 0)
      attempt <- use(ctx.theHailClassLoader, ctx.fs, taskContext, ctx.r).attempt
      _ = taskContext.close()
      result <- attempt.fold(M.raiseError, M.pure)
    } yield result

  def time[M[_], A](name: String)(a: => A)(implicit M: Ask[M, ExecuteContext]): M[A] =
    M.reader(_.timer.time(name)(a))

  def timeM[M[_]: Monad, A](name: String)(fa: M[A])(implicit M: Ask[M, ExecuteContext]): M[A] =
    M.reader(_.timer).flatMap(_.timeM(name)(fa))

  def newTmpPath[M[_]](prefix: String, ext: String = null)(implicit M: Ask[M, ExecuteContext]): M[String] =
    M.reader(_.createTmpPath(prefix, ext))

  def readFlag[M[_]](name: String)(implicit M: Ask[M, ExecuteContext]): M[String] =
    M.reader(_.getFlag(name))

  def assertA[M[_]](condition: Boolean, message: String = "")(implicit M: ApplicativeThrow[M]): M[Unit] =
    if (condition) M.unit else M.raiseError(new AssertionError(message).fillInStackTrace())

  def raisePretty[M[_], A](mkMessage: (BaseIR => String) => Throwable)
                          (implicit A: Ask[M, ExecuteContext], M: MonadThrow[M]): M[A] =
    A.ask.flatMap { ctx => M.raiseError(mkMessage(Pretty(ctx, _))) }

  def prettyFatal[M[_]: MonadThrow, A](mkMessage: (BaseIR => String) => String)
                                      (implicit A: Ask[M, ExecuteContext]): M[A] =
    raisePretty(pretty => new HailException(mkMessage(pretty)))

  @inline def unsafe[M[_], A](fa: => A)(implicit M: MonadThrow[M]): M[A] =
    Try(fa).fold(M.raiseError, M.pure)
}