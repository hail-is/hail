package is.hail

import is.hail.utils.HailException

import scala.reflect.ClassTag

import org.apache.spark.SparkException
import org.junit.jupiter.api.Assertions.{assertThrows, assertTrue}

object JUnitTestUtils {

  def intercept[E <: Throwable: ClassTag](f: => Any): E =
    assertThrows(
      implicitly[ClassTag[E]].runtimeClass.asInstanceOf[Class[E]],
      () => { f; () },
    )

  def interceptException[E <: Throwable: ClassTag](regex: String)(f: => Any): Unit = {
    val thrown = intercept[E](f)
    val msg = thrown.getMessage
    val matched = msg != null && regex.r.findFirstIn(msg).isDefined
    assertTrue(
      matched,
      s"expected exception with pattern '$regex'\n  Found: $msg",
    )
  }

  def interceptFatal(regex: String)(f: => Any): Unit =
    interceptException[HailException](regex)(f)

  def interceptSpark(regex: String)(f: => Any): Unit =
    interceptException[SparkException](regex)(f)

  def interceptAssertion(regex: String)(f: => Any): Unit =
    interceptException[AssertionError](regex)(f)

  def cartesian[A, B](as: Iterable[A], bs: Iterable[B]): Iterable[(A, B)] =
    for {
      a <- as
      b <- bs
    } yield (a, b)
}
