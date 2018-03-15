package is.hail.expr.ir.functions

import is.hail.asm4s._
import is.hail.expr.ir._
import is.hail.expr.types._
import is.hail.utils._
import is.hail.asm4s.coerce

import scala.collection.mutable
import scala.reflect.ClassTag

object IRFunctionRegistry {

  val registry: mutable.Map[String, Seq[(Seq[Type], Seq[IR] => IR)]] = mutable.Map().withDefaultValue(Seq.empty)

  def addIRFunction(f: IRFunction) {
    val l = registry(f.name)
    registry.put(f.name,
      l :+ (f.argTypes, { args: Seq[IR] =>
        ApplyFunction(f, args)
      }))
  }

  def addIR(name: String, types: Seq[Type], f: Seq[IR] => IR) {
    val l = registry(name)
    registry.put(name, l :+ ((types, f)))
  }

  def lookupFunction(name: String, args: Seq[Type]): Option[Seq[IR] => IR] = {
    assert(args.forall(_ != null))
    val validMethods = registry(name).flatMap { case (ts, f) =>
      if (ts.length == args.length) {
        ts.foreach(_.clear())
        if ((ts, args).zipped.forall(_.unify(_)))
          Some(f)
        else
          None
      } else {
        None
      }
    }
    validMethods match {
      case Seq() => None
      case Seq(x) => Some(x)
      case _ => fatal(s"Multiple methods found that satisfy $name$args.")
    }
  }

  CallFunctions.registerAll()
  GenotypeFunctions.registerAll()
  MathFunctions.registerAll()
  UtilFunctions.registerAll()
}

abstract class RegistryFunctions {

  def registerAll(): Unit

  private val boxes = mutable.Map[String, Box[Type]]()

  private def tvBoxes(name: String) = boxes.getOrElseUpdate(name, Box[Type]())

  def tv(name: String): TVariable =
    TVariable(name, b = tvBoxes(name))

  def tv(name: String, cond: Type => Boolean): TVariable =
    TVariable(name, cond, tvBoxes(name))

  def tnum(name: String): TVariable =
    tv(name, _.isInstanceOf[TNumeric])

  def registerCode(mname: String, aTypes: Array[Type], rType: Type)(impl: (MethodBuilder, Array[Code[_]]) => Code[_]) {
    IRFunctionRegistry.addIRFunction(new IRFunction {
      override val name: String = mname

      override val argTypes: Seq[Type] = aTypes

      override val returnType: Type = rType

      override def apply(mb: MethodBuilder, args: Code[_]*): Code[_] = impl(mb, args.toArray)
    })
  }

  def registerScalaFunction(mname: String, argTypes: Array[Type], rType: Type)(cls: Class[_], method: String) {
    registerCode(mname, argTypes, rType) { (mb, args) =>
      val cts = argTypes.map(TypeToIRIntermediateClassTag(_).runtimeClass)
      Code.invokeScalaObject(cls, method, cts, args)(TypeToIRIntermediateClassTag(rType))
    }
  }

  def registerScalaFunction(mname: String, types: Type*)(cls: Class[_], method: String): Unit =
    registerScalaFunction(mname: String, types.init.toArray, types.last)(cls, method)

  def registerJavaStaticFunction(mname: String, argTypes: Array[Type], rType: Type)(cls: Class[_], method: String) {
    registerCode(mname, argTypes, rType) { (mb, args) =>
      val cts = argTypes.map(TypeToIRIntermediateClassTag(_).runtimeClass)
      Code.invokeStatic(cls, method, cts, args)(TypeToIRIntermediateClassTag(rType))
    }
  }

  def registerJavaStaticFunction(mname: String, types: Type*)(cls: Class[_], method: String): Unit =
    registerJavaStaticFunction(mname, types.init.toArray, types.last)(cls, method)

  def registerIR(mname: String, argTypes: Array[Type])(f: Seq[IR] => IR) {
    IRFunctionRegistry.addIR(mname, argTypes, f)
  }

  def registerCode(mname: String, rt: Type)(impl: MethodBuilder => Code[_]): Unit =
    registerCode(mname, Array[Type](), rt) { case (mb, Array()) => impl(mb) }

  def registerCode[T1: TypeInfo](mname: String, mt1: Type, rt: Type)(impl: (MethodBuilder, Code[T1]) => Code[_]): Unit =
    registerCode(mname, Array(mt1), rt) { case (mb, Array(a1)) => impl(mb, coerce[T1](a1)) }

  def registerCode[T1: TypeInfo, T2: TypeInfo](mname: String, mt1: Type, mt2: Type, rt: Type)(impl: (MethodBuilder, Code[T1], Code[T2]) => Code[_]): Unit =
    registerCode(mname, Array(mt1, mt2), rt) { case (mb, Array(a1, a2)) => impl(mb, coerce[T1](a1), coerce[T2](a2)) }

  def registerCode[T1: TypeInfo, T2: TypeInfo, T3: TypeInfo](mname: String, mt1: Type, mt2: Type, mt3: Type, rt: Type)
    (impl: (MethodBuilder, Code[_], Code[_], Code[_]) => Code[_]): Unit =
    registerCode(mname, Array(mt1, mt2, mt3), rt) { case (mb, Array(a1, a2, a3)) => impl(mb, coerce[T1](a1), coerce[T2](a2), coerce[T3](a3)) }

  def registerIR(mname: String)(f: () => IR): Unit =
    registerIR(mname, Array[Type]()) { case Seq() => f() }

  def registerIR(mname: String, mt1: Type)(f: IR => IR): Unit =
    registerIR(mname, Array(mt1)) { case Seq(a1) => f(a1) }

  def registerIR(mname: String, mt1: Type, mt2: Type)(f: (IR, IR) => IR): Unit =
    registerIR(mname, Array(mt1, mt2)) { case Seq(a1, a2) => f(a1, a2) }

  def registerIR(mname: String, mt1: Type, mt2: Type, mt3: Type)(f: (IR, IR, IR) => IR): Unit =
    registerIR(mname, Array(mt1, mt2, mt3)) { case Seq(a1, a2, a3) => f(a1, a2, a3) }
}

abstract class IRFunction {
  def name: String

  def argTypes: Seq[Type]

  def apply(mb: MethodBuilder, args: Code[_]*): Code[_]

  def returnType: Type

  override def toString: String = s"$name(${ argTypes.mkString(", ") }): $returnType"

}
