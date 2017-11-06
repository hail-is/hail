package is.hail.expr.ir

import is.hail.utils._
import is.hail.annotations.MemoryBuffer
import is.hail.asm4s._
import is.hail.expr.{TInt32, TInt64, TArray, TContainer, TStruct, TFloat32, TFloat64, TBoolean, Type}
import is.hail.annotations.StagedRegionValueBuilder

import scala.collection.mutable

object Primitives {
  private case class Primitive(name: String, f: (Array[Type]) => (Type, Array[Code[_]] => Code[_]))
  private val primitives: mutable.Map[String, Primitive] = mutable.HashMap()
  private def numeric(
    name: String,
    i: (Code[Int], Code[Int]) => Code[Int],
    l: (Code[Long], Code[Long]) => Code[Long],
    f: (Code[Float], Code[Float]) => Code[Float],
    d: (Code[Double], Code[Double]) => Code[Double]
  ): Primitive = Primitive(name,
    { case Array(TInt32, TInt32)     => TInt32   -> { case Array(x, y) => i(x.asInstanceOf[Code[Int]], y.asInstanceOf[Code[Int]]) }
      case Array(TInt64, TInt64)     => TInt64   -> { case Array(x, y) => l(x.asInstanceOf[Code[Long]], y.asInstanceOf[Code[Long]]) }
      case Array(TFloat32, TFloat32) => TFloat32 -> { case Array(x, y) => f(x.asInstanceOf[Code[Float]], y.asInstanceOf[Code[Float]]) }
      case Array(TFloat64, TFloat64) => TFloat64 -> { case Array(x, y) => d(x.asInstanceOf[Code[Double]], y.asInstanceOf[Code[Double]]) }
      case x => throw new RuntimeException(s"boom ${x.toSeq}") })
  Array[Primitive](
    numeric("+", _ + _, _ + _, _ + _, _ + _),
    numeric("/", _ / _, _ / _, _ / _, _ / _),
    Primitive("||", { case Array(TBoolean, TBoolean) => TBoolean -> { case Array(x, y) => x.asInstanceOf[Code[Boolean]] || y.asInstanceOf[Code[Boolean]] }
      case x => throw new RuntimeException(s"boom ${x.toSeq}")}),
    Primitive("&&", { case Array(TBoolean, TBoolean) => TBoolean -> { case Array(x, y) => x.asInstanceOf[Code[Boolean]] && y.asInstanceOf[Code[Boolean]] }
      case x => throw new RuntimeException(s"boom ${x.toSeq}")}),
    Primitive("!", { case Array(TBoolean) => TBoolean -> { case Array(x) => !x.asInstanceOf[Code[Boolean]] }
      case x => throw new RuntimeException(s"boom ${x.toSeq}")})
  ).foreach(x => primitives += (x.name -> x))

  def lookup(name: String, paramTyps: Array[Type], params: Array[Code[_]]): Code[_] =
    primitives(name).f(paramTyps)._2(params)

  def returnTyp(name: String, paramTyps: Array[Type]): Type =
    primitives(name).f(paramTyps)._1
}
