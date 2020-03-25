package is.hail.expr.ir.functions
import is.hail.annotations._
import is.hail.asm4s._
import is.hail.expr.ir._
import is.hail.expr.types._
import is.hail.utils._
import is.hail.asm4s.coerce
import is.hail.experimental.ExperimentalFunctions
import is.hail.expr.types.physical._
import is.hail.expr.types.virtual._
import is.hail.variant.Locus

import scala.collection.JavaConverters._
import scala.collection.mutable
import scala.reflect._

object IRFunctionRegistry {
  private val userAddedFunctions: mutable.Set[(String, (Type, Seq[Type]))] = mutable.HashSet.empty

  def clearUserFunctions() {
    userAddedFunctions.foreach { case (name, (rt, argTypes)) => removeIRFunction(name, rt, argTypes) }
    userAddedFunctions.clear()
  }

  val irRegistry: mutable.Map[String, mutable.Map[(Seq[Type], Type, Boolean), Seq[IR] => IR]] =
    new mutable.HashMap[String, mutable.Map[(Seq[Type], Type, Boolean), Seq[IR] => IR]]()

  val codeRegistry: mutable.MultiMap[String, IRFunction] =
    new mutable.HashMap[String, mutable.Set[IRFunction]] with mutable.MultiMap[String, IRFunction]

  def addIRFunction(f: IRFunction): Unit = {
    if (!isJavaIdentifier(f.name))
      throw new IllegalArgumentException(s"Illegal function name, not Java identifier: ${ f.name }")

    codeRegistry.addBinding(f.name, f)
  }

  def addIR(name: String, argTypes: Seq[Type], retType: Type, alwaysInline: Boolean, f: Seq[IR] => IR): Unit = {
    if (!isJavaIdentifier(name))
      throw new IllegalArgumentException(s"Illegal function name, not Java identifier: $name")

    val m = irRegistry.getOrElseUpdate(name, new mutable.HashMap[(Seq[Type], Type, Boolean), Seq[IR] => IR]())
    m.update((argTypes, retType, alwaysInline), f)
  }

  def pyRegisterIR(mname: String,
    argNames: java.util.ArrayList[String],
    argTypeStrs: java.util.ArrayList[String], retType: String,
    body: IR): Unit = {
    if (!isJavaIdentifier(mname))
      throw new IllegalArgumentException(s"Illegal function name, not Java identifier: $mname")

    val argTypes = argTypeStrs.asScala.map(IRParser.parseType).toFastIndexedSeq
    userAddedFunctions += ((mname, (body.typ, argTypes)))
    addIR(mname,
      argTypes, IRParser.parseType(retType), false, { args =>
        Subst(body,
          BindingEnv(Env[IR](argNames.asScala.zip(args): _*)))
      })
  }

  def removeIRFunction(name: String, rt: Type, args: Seq[Type]): Unit = {
    val functions = codeRegistry(name)
    val toRemove = functions.filter(_.unify(args :+ rt)).toArray
    assert(toRemove.length == 1)
    codeRegistry.removeBinding(name, toRemove.head)
  }

  def removeIRFunction(name: String): Unit =
    codeRegistry.remove(name)

  def lookupFunction(name: String, rt: Type, args: Seq[Type]): Option[IRFunction] =
    codeRegistry.lift(name).map { fs => fs.filter(t => t.unify(args :+ rt)).toSeq }.getOrElse(FastSeq()) match {
      case Seq() => None
      case Seq(f) => Some(f)
      case _ => fatal(s"Multiple functions found that satisfy $name(${ args.mkString(",") }).")
    }

  def lookupIR(name: String, rt: Type, args: Seq[Type]): Option[((Seq[Type], Type, Boolean), Seq[IR] => IR)] = {
    irRegistry.getOrElse(name, Map.empty).filter { case ((argTypes, _, _), _) =>
     argTypes.length == args.length && {
       argTypes.foreach(_.clear())
       (argTypes, args).zipped.forall(_.unify(_))
     }
    }.toSeq match {
      case Seq() => None
      case Seq(kv) => Some(kv)
      case _ => fatal(s"Multiple functions found that satisfy $name(${args.mkString(",")}).")
    }
  }

  def lookupConversion(name: String, rt: Type, args: Seq[Type]): Option[Seq[IR] => IR] = {
    val validIR: Option[Seq[IR] => IR] = lookupIR(name, rt, args).map {
      case ((_, _, inline), conversion) => args =>
        val x = ApplyIR(name, args)
        x.conversion = conversion
        x.inline = inline
        x
    }

    val validMethods = lookupFunction(name, rt, args).map { f => { irArgs: Seq[IR] =>
      f match {
        case _: SeededIRFunction =>
          ApplySeeded(name, irArgs.init, irArgs.last.asInstanceOf[I64].x, f.returnType.subst())
        case _: IRFunctionWithoutMissingness => Apply(name, irArgs, f.returnType.subst())
        case _: IRFunctionWithMissingness => ApplySpecial(name, irArgs, f.returnType.subst())
      }
    }
    }

    (validIR, validMethods) match {
      case (None, None) =>
        None
      case (None, Some(x)) => Some(x)
      case (Some(x), None) => Some(x)
      case _ => fatal(s"Multiple methods found that satisfy $name(${ args.mkString(",") }).")
    }
  }

  Seq(
    ArrayFunctions,
    NDArrayFunctions,
    CallFunctions,
    DictFunctions,
    GenotypeFunctions,
    IntervalFunctions,
    LocusFunctions,
    MathFunctions,
    RandomSeededFunctions,
    SetFunctions,
    StringFunctions,
    UtilFunctions,
    ExperimentalFunctions
  ).foreach(_.registerAll())

  def dumpFunctions(): Unit = {
    def dtype(t: Type): String = s"""dtype("${ StringEscapeUtils.escapeString(t.toString) }\")"""

    irRegistry.foreach { case (name, fns) =>
        fns.foreach { case ((argTypes, retType, _), f) =>
          println(s"""register_function("${ StringEscapeUtils.escapeString(name) }", (${ argTypes.map(dtype).mkString(",") }), ${ dtype(retType) })""")
        }
    }

    codeRegistry.foreach { case (name, fns) =>
        fns.foreach { f =>
          println(s"""${
            if (f.isInstanceOf[SeededIRFunction])
              "register_seeded_function"
            else
              "register_function"
          }("${ StringEscapeUtils.escapeString(name) }", (${ f.argTypes.map(dtype).mkString(",") }), ${ dtype(f.returnType) })""")
        }
    }
  }
}

abstract class RegistryFunctions {

  def registerAll(): Unit

  private val boxes = mutable.Map[String, Box[Type]]()

  def tv(name: String): TVariable =
    TVariable(name)

  def tv(name: String, cond: String): TVariable =
    TVariable(name, cond)

  def tnum(name: String): TVariable =
    tv(name, "numeric")

  def wrapArg(r: EmitRegion, t: PType): Code[_] => Code[_] = t match {
    case _: PBoolean => coerce[Boolean]
    case _: PInt32 => coerce[Int]
    case _: PInt64 => coerce[Long]
    case _: PFloat32 => coerce[Float]
    case _: PFloat64 => coerce[Double]
    case _: PCall => coerce[Int]
    case t: PString => c => t.loadString(coerce[Long](c))
    case t: PLocus => c => LocusFunctions.getLocus(r, coerce[Long](c), t)
    case _ => c =>
      Code.invokeScalaObject[PType, Region, Long, Any](
        UnsafeRow.getClass, "read",
        r.mb.getPType(t),
        r.region, coerce[Long](c))
  }

  def boxedTypeInfo(t: PType): TypeInfo[_ >: Null] = t match {
    case _: PBoolean => classInfo[java.lang.Boolean]
    case _: PInt32 => classInfo[java.lang.Integer]
    case _: PInt64 => classInfo[java.lang.Long]
    case _: PFloat32 => classInfo[java.lang.Float]
    case _: PFloat64 => classInfo[java.lang.Double]
    case _: PCall => classInfo[java.lang.Integer]
    case t: PString => classInfo[java.lang.String]
    case t: PLocus => classInfo[Locus]
    case _ => classInfo[AnyRef]
  }

  def boxArg(r: EmitRegion, t: PType): Code[_] => Code[AnyRef] = t match {
    case _: PBoolean => c => Code.boxBoolean(coerce[Boolean](c))
    case _: PInt32 => c => Code.boxInt(coerce[Int](c))
    case _: PInt64 => c => Code.boxLong(coerce[Long](c))
    case _: PFloat32 => c => Code.boxFloat(coerce[Float](c))
    case _: PFloat64 => c => Code.boxDouble(coerce[Double](c))
    case _: PCall => c => Code.boxInt(coerce[Int](c))
    case t: PString => c => t.loadString(coerce[Long](c))
    case t: PLocus => c => LocusFunctions.getLocus(r, coerce[Long](c), t)
    case _ => c =>
      Code.invokeScalaObject[PType, Region, Long, AnyRef](
        UnsafeRow.getClass, "readAnyRef",
        r.mb.getPType(t),
        r.region, coerce[Long](c))
  }

  def unwrapReturn(r: EmitRegion, pt: PType): Code[_] => Code[_] = pt.virtualType match {
    case TBoolean => identity[Code[_]]
    case TInt32 => identity[Code[_]]
    case TInt64 => identity[Code[_]]
    case TFloat32 => identity[Code[_]]
    case TFloat64 => identity[Code[_]]
    case TString => c =>
      pt.asInstanceOf[PString].allocateAndStoreString(r.mb, r.region, coerce[String](c))
    case TCall => coerce[Int]
    case TArray(TInt32) => c =>
      val srvb = new StagedRegionValueBuilder(r, pt)
      val alocal = r.mb.newLocal[IndexedSeq[Int]]()
      val len = r.mb.newLocal[Int]()
      val v = r.mb.newLocal[java.lang.Integer]()

      Code(
        alocal := coerce[IndexedSeq[Int]](c),
        len := alocal.invoke[Int]("size"),
        Code(
          srvb.start(len),
          Code.whileLoop(srvb.arrayIdx < len,
            v := Code.checkcast[java.lang.Integer](alocal.invoke[Int, java.lang.Object]("apply", srvb.arrayIdx)),
            v.isNull.mux(srvb.setMissing(), srvb.addInt(v.invoke[Int]("intValue"))),
            srvb.advance())),
        srvb.offset)
    case TArray(TFloat64) => c =>
      val srvb = new StagedRegionValueBuilder(r, pt)
      val alocal = r.mb.newLocal[IndexedSeq[Double]]()
      val len = r.mb.newLocal[Int]()
      val v = r.mb.newLocal[java.lang.Double]()

      Code(
        alocal := coerce[IndexedSeq[Double]](c),
        len := alocal.invoke[Int]("size"),
        Code(
          srvb.start(len),
          Code.whileLoop(srvb.arrayIdx < len,
            v := Code.checkcast[java.lang.Double](alocal.invoke[Int, java.lang.Object]("apply", srvb.arrayIdx)),
            v.isNull.mux(srvb.setMissing(), srvb.addDouble(v.invoke[Double]("doubleValue"))),
            srvb.advance())),
        srvb.offset)
    case TArray(TString) => c =>
      val srvb = new StagedRegionValueBuilder(r, pt)
      val alocal = r.mb.newLocal[IndexedSeq[String]]()
      val len = r.mb.newLocal[Int]()
      val v = r.mb.newLocal[java.lang.String]()

      Code(
        alocal := coerce[IndexedSeq[String]](c),
        len := alocal.invoke[Int]("size"),
        Code(
          srvb.start(len),
          Code.whileLoop(srvb.arrayIdx < len,
            v := Code.checkcast[java.lang.String](alocal.invoke[Int, java.lang.Object]("apply", srvb.arrayIdx)),
            v.isNull.mux(srvb.setMissing(), srvb.addString(v)),
            srvb.advance())),
        srvb.offset)
  }

  def registerCode(mname: String, aTypes: Array[Type], rType: Type, pt: Seq[PType] => PType)
    (impl: (EmitRegion, PType, Array[(PType, Code[_])]) => Code[_]) {
    IRFunctionRegistry.addIRFunction(new IRFunctionWithoutMissingness {
      override val name: String = mname

      override val argTypes: Seq[Type] = aTypes

      override val returnType: Type = rType

      override def returnPType(argTypes: Seq[PType], returnType: Type): PType = {
        val p = if (pt == null) PType.canonical(returnType) else pt(argTypes)

        // IRFunctionWithoutMissingness returns missing if any arguments are missing
        p.setRequired(argTypes.forall(_.required))
      }

      override def apply(r: EmitRegion, returnPType: PType, args: (PType, Code[_])*): Code[_] = {
        unify(args.map(_._1.virtualType))
        impl(r, returnPType, args.toArray)
      }
    })
  }

  def registerCodeWithMissingness(mname: String, aTypes: Array[Type], rType: Type, pt: Seq[PType] => PType)
    (impl: (EmitRegion, PType, Array[(PType, EmitCode)]) => EmitCode) {
    IRFunctionRegistry.addIRFunction(new IRFunctionWithMissingness {
      override val name: String = mname

      override val argTypes: Seq[Type] = aTypes

      override val returnType: Type = rType

      override def returnPType(argTypes: Seq[PType], returnType: Type): PType = if (pt == null) PType.canonical(returnType) else pt(argTypes)

      override def apply(r: EmitRegion, rpt: PType, args: (PType, EmitCode)*): EmitCode = {
        unify(args.map(_._1.virtualType))
        impl(r, rpt, args.toArray)
      }
    })
  }

  def registerScalaFunction(mname: String, argTypes: Array[Type], rType: Type, pt: Seq[PType] => PType)(cls: Class[_], method: String) {
    registerCode(mname, argTypes, rType, pt) { case (r, rt, args) =>
      val cts = argTypes.map(TypeToIRIntermediateClassTag(_).runtimeClass)
      Code.invokeScalaObject(cls, method, cts, args.map(_._2))(TypeToIRIntermediateClassTag(rType))
    }
  }

  def registerWrappedScalaFunction(mname: String, argTypes: Array[Type], rType: Type, pt: Seq[PType] => PType)(cls: Class[_], method: String) {
    def ct(typ: Type): ClassTag[_] = typ match {
      case TString => classTag[String]
      case TArray(TInt32) => classTag[IndexedSeq[Int]]
      case TArray(TFloat64) => classTag[IndexedSeq[Double]]
      case TArray(TString) => classTag[IndexedSeq[String]]
      case TSet(TString) => classTag[Set[String]]
      case TDict(TString, TString) => classTag[Map[String, String]]
      case t => TypeToIRIntermediateClassTag(t)
    }

    registerCode(mname, argTypes, rType, pt) { case (r, rt, args) =>
      val cts = argTypes.map(ct(_).runtimeClass)
      val out = Code.invokeScalaObject(cls, method, cts, args.map { case (t, a) => wrapArg(r, t)(a) })(ct(rType))
      unwrapReturn(r, rt)(out)
    }
  }

  def registerWrappedScalaFunction(mname: String, a1: Type, rType: Type, pt: PType => PType)(cls: Class[_], method: String): Unit =
    registerWrappedScalaFunction(mname, Array(a1), rType, unwrappedApply(pt))(cls, method)

  def registerWrappedScalaFunction(mname: String, a1: Type, a2: Type, rType: Type, pt: (PType, PType) => PType)(cls: Class[_], method: String): Unit =
    registerWrappedScalaFunction(mname, Array(a1, a2), rType, unwrappedApply(pt))(cls, method)

  def registerWrappedScalaFunction(mname: String, a1: Type, a2: Type, a3: Type, rType: Type,
    pt: (PType, PType, PType) => PType)(cls: Class[_], method: String): Unit =
    registerWrappedScalaFunction(mname, Array(a1, a2, a3), rType, unwrappedApply(pt))(cls, method)

  def registerJavaStaticFunction(mname: String, argTypes: Array[Type], rType: Type, pt: Seq[PType] => PType)(cls: Class[_], method: String) {
    registerCode(mname, argTypes, rType, pt) { case (r, rt, args) =>
      val cts = argTypes.map(TypeToIRIntermediateClassTag(_).runtimeClass)
      Code.invokeStatic(cls, method, cts, args.map(_._2))(TypeToIRIntermediateClassTag(rType))
    }
  }

  def registerIR(mname: String, argTypes: Array[Type], retType: Type, inline: Boolean = false)(f: Seq[IR] => IR) {
    IRFunctionRegistry.addIR(mname, argTypes, retType, inline, f)
  }

  def registerCode(mname: String, rt: Type, pType: PType)(impl: EmitRegion => Code[_]): Unit =
    registerCode(mname, Array[Type](), rt, (_: Seq[PType]) => pType) { case (r, rt, array) => impl(r) }

  def registerCode[A1](mname: String, mt1: Type, rt: Type, pt: PType => PType)(impl: (EmitRegion, PType, (PType, Code[A1])) => Code[_]): Unit =
    registerCode(mname, Array(mt1), rt, unwrappedApply(pt)) {
      case (r, rt, Array(a1: (PType, Code[A1]) @unchecked)) => impl(r, rt, a1)
    }

  def registerCode[A1, A2](mname: String, mt1: Type, mt2: Type, rt: Type, pt: (PType, PType) => PType)
    (impl: (EmitRegion, PType, (PType, Code[A1]), (PType, Code[A2])) => Code[_]): Unit =
    registerCode(mname, Array(mt1, mt2), rt, unwrappedApply(pt)) {
      case (r, rt, Array(
      a1: (PType, Code[A1]) @unchecked,
      a2: (PType, Code[A2]) @unchecked)) => impl(r, rt, a1, a2)
    }

  def registerCode[A1, A2, A3](mname: String, mt1: Type, mt2: Type, mt3: Type, rt: Type, pt: (PType, PType, PType) => PType)
    (impl: (EmitRegion, PType, (PType, Code[A1]), (PType, Code[A2]), (PType, Code[A3])) => Code[_]): Unit =
    registerCode(mname, Array(mt1, mt2, mt3), rt, unwrappedApply(pt)) {
      case (r, rt, Array(
      a1: (PType, Code[A1]) @unchecked,
      a2: (PType, Code[A2]) @unchecked,
      a3: (PType, Code[A3]) @unchecked)) => impl(r, rt, a1, a2, a3)
    }

  def registerCode[A1, A2, A3, A4](mname: String, mt1: Type, mt2: Type, mt3: Type, mt4: Type, rt: Type, pt: (PType, PType, PType, PType) => PType)
    (impl: (EmitRegion, PType, (PType, Code[A1]), (PType, Code[A2]), (PType, Code[A3]), (PType, Code[A4])) => Code[_]): Unit =
    registerCode(mname, Array(mt1, mt2, mt3, mt4), rt, unwrappedApply(pt)) {
      case (r, rt, Array(
      a1: (PType, Code[A1]) @unchecked,
      a2: (PType, Code[A2]) @unchecked,
      a3: (PType, Code[A3]) @unchecked,
      a4: (PType, Code[A4]) @unchecked)) => impl(r, rt, a1, a2, a3, a4)
    }

  def registerCode[A1, A2, A3, A4, A5](mname: String, mt1: Type, mt2: Type, mt3: Type, mt4: Type, mt5: Type, rt: Type,
    pt: (PType, PType, PType, PType, PType) => PType)
    (impl: (EmitRegion, PType, (PType, Code[A1]), (PType, Code[A2]), (PType, Code[A3]), (PType, Code[A4]), (PType, Code[A5])) => Code[_]): Unit =
    registerCode(mname, Array(mt1, mt2, mt3, mt4, mt5), rt, unwrappedApply(pt)) {
      case (r, rt, Array(
      a1: (PType, Code[A1]) @unchecked,
      a2: (PType, Code[A2]) @unchecked,
      a3: (PType, Code[A3]) @unchecked,
      a4: (PType, Code[A4]) @unchecked,
      a5: (PType, Code[A5]) @unchecked)) => impl(r, rt, a1, a2, a3, a4, a5)
    }

  def registerCodeWithMissingness(mname: String, rt: Type, pt: PType)(impl: EmitRegion => EmitCode): Unit =
    registerCodeWithMissingness(mname, Array[Type](), rt, (_: Seq[PType]) => pt) { case (r, rt, Array()) => impl(r) }

  def registerCodeWithMissingness(mname: String, mt1: Type, rt: Type, pt: PType => PType)
    (impl: (EmitRegion, PType, (PType, EmitCode)) => EmitCode): Unit =
    registerCodeWithMissingness(mname, Array(mt1), rt, unwrappedApply(pt)) { case (r, rt, Array(a1)) => impl(r, rt, a1) }

  def registerCodeWithMissingness(mname: String, mt1: Type, mt2: Type, rt: Type, pt: (PType, PType) => PType)
    (impl: (EmitRegion, PType, (PType, EmitCode), (PType, EmitCode)) => EmitCode): Unit =
    registerCodeWithMissingness(mname, Array(mt1, mt2), rt, unwrappedApply(pt)) { case (r, rt, Array(a1, a2)) => impl(r, rt, a1, a2) }

  def registerCodeWithMissingness(mname: String, mt1: Type, mt2: Type, mt3: Type, mt4: Type, rt: Type, pt: (PType, PType, PType) => PType)
    (impl: (EmitRegion, PType, (PType, EmitCode), (PType, EmitCode), (PType, EmitCode), (PType, EmitCode)) => EmitCode): Unit =
    registerCodeWithMissingness(mname, Array(mt1, mt2, mt3, mt4), rt, unwrappedApply(pt)) { case (r, rt, Array(a1, a2, a3, a4)) => impl(r, rt, a1, a2, a3, a4) }

  def registerCodeWithMissingness(mname: String, mt1: Type, mt2: Type, mt3: Type, mt4: Type, mt5: Type, mt6: Type, rt: Type, pt: (PType, PType, PType) => PType)
    (impl: (EmitRegion, PType, (PType, EmitCode), (PType, EmitCode), (PType, EmitCode), (PType, EmitCode), (PType, EmitCode), (PType, EmitCode)) => EmitCode): Unit =
    registerCodeWithMissingness(mname, Array(mt1, mt2, mt3, mt4, mt5, mt6), rt, unwrappedApply(pt)) { case (r, rt, Array(a1, a2, a3, a4, a5, a6)) => impl(r, rt, a1, a2, a3, a4, a5, a6) }

  def registerIR(mname: String, retType: Type)(f: () => IR): Unit =
    registerIR(mname, Array[Type](), retType) { case Seq() => f() }

  def registerIR(mname: String, mt1: Type, retType: Type)(f: IR => IR): Unit =
    registerIR(mname, Array(mt1), retType) { case Seq(a1) => f(a1) }

  def registerIR(mname: String, mt1: Type, mt2: Type, retType: Type)(f: (IR, IR) => IR): Unit =
    registerIR(mname, Array(mt1, mt2), retType) { case Seq(a1, a2) => f(a1, a2) }

  def registerIR(mname: String, mt1: Type, mt2: Type, mt3: Type, retType: Type)(f: (IR, IR, IR) => IR): Unit =
    registerIR(mname, Array(mt1, mt2, mt3), retType) { case Seq(a1, a2, a3) => f(a1, a2, a3) }

  def registerIR(mname: String, mt1: Type, mt2: Type, mt3: Type, mt4: Type, retType: Type)(f: (IR, IR, IR, IR) => IR): Unit =
    registerIR(mname, Array(mt1, mt2, mt3, mt4), retType) { case Seq(a1, a2, a3, a4) => f(a1, a2, a3, a4) }

  def registerSeeded(mname: String, aTypes: Array[Type], rType: Type, pt: Seq[PType] => PType)
    (impl: (EmitRegion, PType, Long, Array[(PType, Code[_])]) => Code[_]) {

    IRFunctionRegistry.addIRFunction(new SeededIRFunction {
      val isDeterministic: Boolean = false

      override val name: String = mname

      override val argTypes: Seq[Type] = aTypes

      override val returnType: Type = rType

      override def returnPType(argTypes: Seq[PType], returnType: Type): PType = if (pt == null) PType.canonical(returnType) else pt(argTypes)

      def applySeeded(seed: Long, r: EmitRegion, rpt: PType, args: (PType, Code[_])*): Code[_] = {
        unify(args.map(_._1.virtualType))
        impl(r, rpt, seed, args.toArray)
      }

      def applySeeded(seed: Long, r: EmitRegion, rpt: PType, args: (PType, EmitCode)*): EmitCode = {
        val setup = Code(args.map(_._2.setup))
        val rpt = returnPType(args.map(_._1), returnType)
        val missing: Code[Boolean] = if (args.isEmpty) false else args.map(_._2.m).reduce(_ || _)
        val value = applySeeded(seed, r, rpt, args.map { case (t, a) => (t, a.v) }: _*)

        EmitCode(setup, missing, PCode(rpt, value))
      }

      override val isStrict: Boolean = true
    })
  }

  def registerSeeded(mname: String, rType: Type, pt: PType)(impl: (EmitRegion, PType, Long) => Code[_]): Unit =
    registerSeeded(mname, Array[Type](), rType, if (pt == null) null else (_: Seq[PType]) => pt) { case (r, rt, seed, array) => impl(r, rt, seed) }

  def registerSeeded[A1](mname: String, arg1: Type, rType: Type, pt: PType => PType)(impl: (EmitRegion, PType, Long, (PType, Code[A1])) => Code[_]): Unit =
    registerSeeded(mname, Array(arg1), rType, unwrappedApply(pt)) {
      case (r, rt, seed, Array(a1: (PType, Code[A1])@unchecked)) => impl(r, rt, seed, a1)
    }

  def registerSeeded[A1, A2](mname: String, arg1: Type, arg2: Type, rType: Type, pt: (PType, PType) => PType)
    (impl: (EmitRegion, PType, Long, (PType, Code[A1]), (PType, Code[A2])) => Code[_]): Unit =
    registerSeeded(mname, Array(arg1, arg2), rType, unwrappedApply(pt)) { case
      (r, rt, seed, Array(a1: (PType, Code[A1])@unchecked, a2: (PType, Code[A2])@unchecked)) =>
      impl(r, rt, seed, a1, a2)
    }

  def registerSeeded[A1, A2, A3, A4](mname: String, arg1: Type, arg2: Type, arg3: Type, arg4: Type, rType: Type, pt: (PType, PType, PType, PType) => PType)
    (impl: (EmitRegion, PType, Long, (PType, Code[A1]), (PType, Code[A2]), (PType, Code[A3]), (PType, Code[A4])) => Code[_]): Unit =
    registerSeeded(mname, Array(arg1, arg2, arg3, arg4), rType, unwrappedApply(pt)) {
        case (r, rt, seed, Array(
        a1: (PType, Code[A1]) @unchecked,
        a2: (PType, Code[A2]) @unchecked,
        a3: (PType, Code[A3]) @unchecked,
        a4: (PType, Code[A4]) @unchecked)) => impl(r, rt, seed, a1, a2, a3, a4)
    }
}

sealed abstract class IRFunction {
  def name: String

  def argTypes: Seq[Type]

  def apply(mb: EmitRegion, returnType: PType, args: (PType, EmitCode)*): EmitCode

  def getAsMethod[C](cb: EmitClassBuilder[C], rpt: PType, args: PType*): EmitMethodBuilder[C] = ???

  def returnType: Type

  def returnPType(argTypes: Seq[PType], returnType: Type): PType

  override def toString: String = s"$name(${ argTypes.mkString(", ") }): $returnType"

  def unify(concrete: Seq[Type]): Boolean = {
    val types = argTypes :+ returnType
    types.length == concrete.length && {
      types.foreach(_.clear())
      types.zip(concrete).forall { case (i, j) => i.unify(j) }
    }
  }
}

abstract class IRFunctionWithoutMissingness extends IRFunction {
  def name: String

  def argTypes: Seq[Type]

  def apply(r: EmitRegion, returnPType: PType, args: (PType, Code[_])*): Code[_]

  def apply(r: EmitRegion, returnPType: PType, args: (PType, EmitCode)*): EmitCode = {
    val setup = Code(args.map(_._2.setup))
    val missing = args.map(_._2.m).reduce(_ || _)
    val value = apply(r, returnPType, args.map { case (t, a) => (t, a.v) }: _*)

    EmitCode(setup, missing, PCode(returnPType, value))
  }

  override def getAsMethod[C](cb: EmitClassBuilder[C], rpt: PType, args: PType*): EmitMethodBuilder[C] = {
    val unified = unify(args.map(_.virtualType) :+ rpt.virtualType)
    assert(unified)
    val ts = argTypes.map(t => typeToTypeInfo(t.subst()))
    val methodbuilder = cb.genEmitMethod(name, (typeInfo[Region] +: ts).toFastIndexedSeq, typeToTypeInfo(rpt))
    methodbuilder.emit(apply(EmitRegion.default(methodbuilder), rpt, args.zip(ts.zipWithIndex.map { case (a, i) => methodbuilder.getArg(i + 2)(a).load() }): _*))
    methodbuilder
  }

  def returnType: Type
}

abstract class IRFunctionWithMissingness extends IRFunction {
  def name: String

  def argTypes: Seq[Type]

  def apply(r: EmitRegion, rpt: PType, args: (PType, EmitCode)*): EmitCode

  def returnType: Type
}

abstract class SeededIRFunction extends IRFunction {
  def name: String

  def argTypes: Seq[Type]

  private[this] var seed: Long = _

  def setSeed(s: Long): Unit = { seed = s }

  def applySeeded(seed: Long, region: EmitRegion, rpt: PType, args: (PType, EmitCode)*): EmitCode

  def apply(region: EmitRegion, rpt: PType, args: (PType, EmitCode)*): EmitCode =
    applySeeded(seed, region, rpt, args: _*)

  def returnType: Type

  def isStrict: Boolean = false
}
