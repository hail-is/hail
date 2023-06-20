package is.hail.backend

import cats.mtl.{Ask, Local}
import cats.syntax.all._
import cats.{ApplicativeThrow, Monad, MonadThrow}
import is.hail.annotations.Region
import is.hail.asm4s.HailClassLoader
import is.hail.backend.local.LocalTaskContext
import is.hail.expr.ir.lowering.MonadLower
import is.hail.expr.ir.{BaseIR, Pretty}
import is.hail.io.fs.FS
import is.hail.utils.HailException

import scala.language.{higherKinds, implicitConversions}
import scala.util.Try

trait MonadExecute[M[_]]
  extends MonadThrow[M]
  with Local[M, ExecuteContext]
  { }

object MonadExecute {
  @inline def apply[M[_]: MonadLower]: MonadLower[M] =
    implicitly
}

object utils {
  def scopedExecution[M[_], A](use: (HailClassLoader, FS, HailTaskContext, Region) => M[A])
                              (implicit M: MonadExecute[M]): M[A] =
    for {
      ctx <- M.ask
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
                          (implicit M: MonadExecute[M]): M[A] =
    M.flatMap(M.ask) { ctx => M.raiseError(mkMessage(Pretty(ctx, _))) }

  def prettyFatal[M[_]: MonadExecute, A](mkMessage: (BaseIR => String) => String): M[A] =
    raisePretty(pretty => new HailException(mkMessage(pretty)))

  @inline def unsafe[M[_], A](fa: => A)(implicit M: MonadThrow[M]): M[A] =
    Try(fa).fold(M.raiseError, M.pure)
}