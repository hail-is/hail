package is.hail.expr.ir.functions

import is.hail.asm4s._
import is.hail.expr.{AST, FunType, Lambda, TypeTag}
import is.hail.expr.ir._
import is.hail.expr.types._
import is.hail.utils._
import is.hail.asm4s.coerce

import scala.collection.mutable
import scala.reflect.ClassTag

object IRFunctionRegistry {

  val registry: mutable.Map[String, Seq[(TypeTag, Seq[IR] => IR)]] = mutable.Map().withDefaultValue(Seq.empty)

  def addIRFunction(f: IRFunction) {
    val l = registry(f.name)
    registry.put(f.name,
      l :+ (f.types, { args: Seq[IR] =>
        ApplyFunction(f, args)
      }))
  }

  def addIR(name: String, types: Seq[Type], f: Seq[IR] => IR) {
    val l = registry(name)
    registry.put(name, l :+ ((FunType(types: _*), f)))
  }

  def lookupFunction(name: String, args: TypeTag): Option[Seq[IR] => IR] = {
    val validMethods = registry(name).flatMap { case (tt, f) =>
      if (tt.xs.length == args.xs.length) {
        tt.clear()
        if (tt.unify(args))
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

  UtilFunctions.registerAll()
}

abstract class RegistryFunctions {

  def registerAll(): Unit

  private val tvs = mutable.Map[String, TVariable]()
  private val tnums = mutable.Map[String, TVariable]()

  def tv(name: String): TVariable =
    tvs.getOrElseUpdate(name, TVariable(name))

  def tnum(name: String): TVariable =
    tnums.getOrElseUpdate(name, TVariable(name, _.isInstanceOf[TNumeric]))

  def tvar(cond: Type => Boolean) = TVariable("COND", cond)

  def registerCode(mname: String, argTypes: Array[Type], rType: Type)(impl: (MethodBuilder, Array[Code[_]]) => Code[_]) {
    IRFunctionRegistry.addIRFunction(new IRFunction {
      override val name: String = mname

      override val types: TypeTag = FunType(argTypes: _*)

      override val returnType: Type = rType

      override def apply(mb: MethodBuilder, args: Code[_]*): Code[_] = impl(mb, args.toArray)
    })
  }

  def registerScalaFunction[R: ClassTag](mname: String, argTypes: Array[Type], rType: Type)(obj: Any, method: String) {
    registerCode(mname, argTypes, rType) { (mb, args) =>
      val cts = argTypes.map(TypeToIRIntermediateClassTag(_).runtimeClass)
      Code.invokeScalaObject[R](obj, method, cts, args)
    }
  }

  def registerScalaFunction[R: ClassTag](mname: String, types: Type*)(obj: Any, method: String): Unit =
    registerScalaFunction(mname: String, types.init.toArray, types.last)(obj, method)

  def registerJavaStaticFunction[C: ClassTag, R: ClassTag](mname: String, argTypes: Array[Type], rType: Type)(method: String) {
    registerCode(mname, argTypes, rType) { (mb, args) =>
      val cts = argTypes.map(TypeToIRIntermediateClassTag(_).runtimeClass)
      Code.invokeStatic[C, R](method, cts, args)
    }
  }

  def registerJavaStaticFunction[C: ClassTag, R: ClassTag](mname: String, types: Type*)(method: String): Unit =
    registerJavaStaticFunction[C, R](mname, types.init.toArray, types.last)(method)

  def registerIR(mname: String, argTypes: Array[Type])(f: Seq[IR] => IR) {
    IRFunctionRegistry.addIR(mname, argTypes, f)
  }

  def registerCode(mname: String, rt: Type)(impl: MethodBuilder => Code[_]): Unit =
    registerCode(mname, Array[Type](), rt) { case (mb, Array()) => impl(mb) }

  def registerCode[T1: TypeInfo](mname: String, mt1: Type, rt: Type)(impl: (MethodBuilder, Code[T1]) => Code[_]): Unit = {
    assert(typeToTypeInfo(mt1) == typeInfo[T1])
    registerCode(mname, Array(mt1), rt) { case (mb, Array(a1)) => impl(mb, coerce[T1](a1)) }
  }

  def registerCode[T1: TypeInfo, T2: TypeInfo](mname: String, mt1: Type, mt2: Type, rt: Type)(impl: (MethodBuilder, Code[T1], Code[T2]) => Code[_]): Unit = {
    assert(typeToTypeInfo(mt1) == typeInfo[T1])
    assert(typeToTypeInfo(mt2) == typeInfo[T2])
    registerCode(mname, Array(mt1, mt2), rt) { case (mb, Array(a1, a2)) => impl(mb, coerce[T1](a1), coerce[T2](a2)) }
  }

  def registerCode[T1: TypeInfo, T2: TypeInfo, T3: TypeInfo](mname: String, mt1: Type, mt2: Type, mt3: Type, rt: Type)
    (impl: (MethodBuilder, Code[_], Code[_], Code[_]) => Code[_]): Unit = {
    assert(typeToTypeInfo(mt1) == typeInfo[T1])
    assert(typeToTypeInfo(mt2) == typeInfo[T2])
    assert(typeToTypeInfo(mt3) == typeInfo[T3])
    registerCode(mname, Array(mt1, mt2, mt3), rt) { case (mb, Array(a1, a2, a3)) => impl(mb, coerce[T1](a1), coerce[T2](a2), coerce[T3](a3)) }
  }

  def registerIR(mname: String, rt: Type)(f: () => IR): Unit =
    registerIR(mname, Array[Type]()) { case Seq() => f() }

  def registerIR(mname: String, mt1: Type, rt: Type)(f: IR => IR): Unit =
    registerIR(mname, Array(mt1)) { case Seq(a1) => f(a1) }

  def registerIR(mname: String, mt1: Type, mt2: Type, rt: Type)(f: (IR, IR) => IR): Unit =
    registerIR(mname, Array(mt1, mt2)) { case Seq(a1, a2) => f(a1, a2) }

  def registerIR(mname: String, mt1: Type, mt2: Type, mt3: Type, rt: Type)(f: (IR, IR, IR) => IR): Unit =
    registerIR(mname, Array(mt1, mt2, mt3)) { case Seq(a1, a2, a3) => f(a1, a2, a3) }

}

abstract class IRFunction {
  def name: String

  def types: TypeTag

  def apply(mb: MethodBuilder, args: Code[_]*): Code[_]

  def returnType: Type

}