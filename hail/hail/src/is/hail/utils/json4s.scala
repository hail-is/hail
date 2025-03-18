package is.hail.utils

import is.hail.backend.ExecuteContext
import org.json4s.reflect.ScalaType
import org.json4s._

import scala.language.implicitConversions
import scala.reflect.{ClassTag, Manifest}

abstract class Json4sFormat[A: Manifest, B <: JValue: ClassTag] {
  def A: Manifest[A] = implicitly
  def B: ClassTag[B] = implicitly
  def reader: Json4sReader[A, B]
  def writer: Json4sWriter[A, B]
  def hints: TypeHints = NoTypeHints
}

abstract class Json4sWriter[A: Manifest, B <: JValue] {
  def apply(a: A)(implicit fs: Formats): B
}

abstract class Json4sReader[A: Manifest, B <: JValue] {
  def apply[C: Manifest](ctx: ExecuteContext, jv: B)(implicit fs: Formats): A
}

package object json4s {

  implicit final class WriterFn[A, B <: JValue](fn: A => Formats => B) extends Json4sWriter[A, B] {
    override def apply(a: A)(implicit fs: Formats): B =
      fn(a)(fs)
  }

  implicit final class ReaderFn[A: Manifest, B <: JValue](fn: (ExecuteContext, B) => Formats => A)
      extends Json4sReader[A, B] {
    override def apply[C: Manifest](ctx: ExecuteContext, jv: B)(implicit fs: Formats): A =
      fn(ctx, jv)(fs)
  }

  implicit final class SerializerToJson4sFormat[A: Manifest](s: Serializer[A])
      extends Json4sFormat[A, JValue] {
    private[this] lazy val typeInfo = ScalaType(A).typeInfo

    override def reader: Json4sReader[A, JValue] =
      (_: ExecuteContext, jv: JValue) =>
        (formats: Formats) =>
          s.deserialize(formats)((typeInfo, jv))

    override def writer: Json4sWriter[A, JValue] =
      (a: A) =>
        (formats: Formats) =>
          s.serialize(formats)(a)
  }

  final private[this] class ManyFormats(fs: Vector[Json4sFormat[_, _]])
      extends Json4sFormat[Any, JValue] {

    override object reader extends Json4sReader[Any, JValue] {
      override def apply[C: Manifest](ctx: ExecuteContext, jv: JValue)(implicit formats: Formats)
        : Any = {
        val ma: Seq[Any] =
          for {
            f <- fs.view
            if f.B.runtimeClass == jv.getClass && f.A == manifest[C]
            reader = f.reader.asInstanceOf[Json4sReader[Any, JValue]]
          } yield reader(ctx, jv)

        ma.headOption.getOrElse(throw new MappingException(
          f"no known conversion from JValue to ${manifest[C].runtimeClass.getName}"
        ))
      }
    }

    override object writer extends Json4sWriter[Any, JValue] {
      override def apply(a: Any)(implicit formats: Formats): JValue = {
        val mjv: Seq[JValue] =
          for {
            f <- fs.view
            a <- f.A.unapply(a)
            write = f.writer.asInstanceOf[Json4sWriter[Any, JValue]]
          } yield write(a)

        mjv.headOption.getOrElse(
          throw new MappingException(f"no known conversion from ${a.getClass} to JValue")
        )
      }
    }

  }

  implicit def formatToFormats[A, B <: JValue](f: Json4sFormat[A, B]): Formats =
    f.hints + f.reader

  implicit def writerToSerializer[A: Manifest, B <: JValue: ClassTag](w: Json4sWriter[A, B])
    : Serializer[A] =
    new CustomSerializer[A](f => (PartialFunction.empty, { case a: A => w(a)(f) }))

  implicit def unsafeReaderToSerializer[A: ClassTag, B <: JValue: ClassTag](r: Json4sReader[A, B])
    : ExecuteContext => Serializer[A] =
    ctx =>
      new CustomSerializer[A](f => ({ case v: B => r(ctx, v)(f) }, PartialFunction.empty))
}
