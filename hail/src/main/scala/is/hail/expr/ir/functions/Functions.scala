package is.hail.expr.ir.functions

import is.hail.annotations._
import is.hail.asm4s._
import is.hail.expr.ir._
import is.hail.types._
import is.hail.utils._
import is.hail.asm4s.coerce
import is.hail.experimental.ExperimentalFunctions
import is.hail.types.physical._
import is.hail.types.physical.stypes.SCode
import is.hail.types.physical.stypes.concrete.{SBaseStructPointer, SBaseStructPointerCode, SStringPointer}
import is.hail.types.physical.stypes.interfaces._
import is.hail.types.physical.stypes.primitives._
import is.hail.types.virtual._
import is.hail.variant.Locus
import org.apache.spark.sql.Row

import scala.collection.JavaConverters._
import scala.collection.mutable
import scala.reflect._

object IRFunctionRegistry {
  private val userAddedFunctions: mutable.Set[(String, (Type, Seq[Type], Seq[Type]))] = mutable.HashSet.empty

  def clearUserFunctions() {
    userAddedFunctions.foreach { case (name, (rt, typeParameters, valueParameterTypes)) =>
      removeJVMFunction(name, rt, typeParameters, valueParameterTypes) }
    userAddedFunctions.clear()
  }

  type IRFunctionSignature = (Seq[Type], Seq[Type], Type, Boolean)
  type IRFunctionImplementation = (Seq[Type], Seq[IR]) => IR

  val irRegistry: mutable.Map[String, mutable.Map[IRFunctionSignature, IRFunctionImplementation]] = new mutable.HashMap()

  val jvmRegistry: mutable.MultiMap[String, JVMFunction] =
    new mutable.HashMap[String, mutable.Set[JVMFunction]] with mutable.MultiMap[String, JVMFunction]

  private[this] def requireJavaIdentifier(name: String): Unit = {
    if (!isJavaIdentifier(name))
      throw new IllegalArgumentException(s"Illegal function name, not Java identifier: ${ name }")
  }

  def addJVMFunction(f: JVMFunction): Unit = {
    requireJavaIdentifier(f.name)
    jvmRegistry.addBinding(f.name, f)
  }

  def addIR(
    name: String,
    typeParameters: Seq[Type],
    valueParameterTypes: Seq[Type],
    returnType: Type,
    alwaysInline: Boolean,
    f: (Seq[Type], Seq[IR]) => IR
  ): Unit = {
    requireJavaIdentifier(name)

    val m = irRegistry.getOrElseUpdate(name, new mutable.HashMap())
    m.update((typeParameters, valueParameterTypes, returnType, alwaysInline), f)
  }

  def pyRegisterIR(
    name: String,
    typeParamStrs: java.util.ArrayList[String],
    argNames: java.util.ArrayList[String],
    argTypeStrs: java.util.ArrayList[String],
    returnType: String,
    body: IR
  ): Unit = {
    requireJavaIdentifier(name)

    val typeParameters = typeParamStrs.asScala.map(IRParser.parseType).toFastIndexedSeq
    val valueParameterTypes = argTypeStrs.asScala.map(IRParser.parseType).toFastIndexedSeq
    userAddedFunctions += ((name, (body.typ, typeParameters, valueParameterTypes)))
    addIR(name,
      typeParameters,
      valueParameterTypes, IRParser.parseType(returnType), false, { (_, args) =>
        Subst(body,
          BindingEnv(Env[IR](argNames.asScala.zip(args): _*)))
      })
  }

  def removeJVMFunction(
    name: String,
    returnType: Type,
    typeParameters: Seq[Type],
    valueParameterTypes: Seq[Type]
  ): Unit = {
    val functions = jvmRegistry(name)
    val toRemove = functions.filter(_.unify(typeParameters, valueParameterTypes, returnType)).toArray
    assert(toRemove.length == 1)
    jvmRegistry.removeBinding(name, toRemove.head)
  }

  def lookupFunction(
    name: String,
    returnType: Type,
    typeParameters: Seq[Type],
    valueParameterTypes: Seq[Type]
  ): Option[JVMFunction] = {
    jvmRegistry.lift(name).map { fs => fs.filter(t => t.unify(typeParameters, valueParameterTypes, returnType)).toSeq }.getOrElse(FastSeq()) match {
      case Seq() => None
      case Seq(f) => Some(f)
      case _ => fatal(s"Multiple functions found that satisfy $name(${ valueParameterTypes.mkString(",") }).")
    }
  }

  def lookupFunctionOrFail(
    name: String,
    returnType: Type,
    typeParameters: Seq[Type],
    valueParameterTypes: Seq[Type]
  ): JVMFunction = {
    jvmRegistry.lift(name) match {
      case None =>
        fatal(s"no functions found with the name ${name}")
      case Some(functions) =>
        functions.filter(t => t.unify(typeParameters, valueParameterTypes, returnType)).toSeq match {
          case Seq() =>
            val prettyFunctionSignature = s"$name[${ typeParameters.mkString(", ") }](${ valueParameterTypes.mkString(", ") }): $returnType"
            val prettyMismatchedFunctionSignatures = functions.map(x => s"  $x").mkString("\n")
            fatal(
              s"No function found with the signature $prettyFunctionSignature.\n" +
              s"However, there are other functions with that name:\n$prettyMismatchedFunctionSignatures")
          case Seq(f) => f
          case _ => fatal(s"Multiple functions found that satisfy $name(${ valueParameterTypes.mkString(", ") }).")
        }
    }
  }

  def lookupIR(
    name: String,
    returnType: Type,
    typeParameters: Seq[Type],
    valueParameterTypes: Seq[Type]
  ): Option[(IRFunctionSignature, IRFunctionImplementation)] = {
    irRegistry.getOrElse(name, Map.empty).filter { case ((typeParametersFound: Seq[Type], valueParameterTypesFound: Seq[Type], _, _), _) =>
      typeParametersFound.length == typeParameters.length && {
        typeParametersFound.foreach(_.clear())
        (typeParametersFound, typeParameters).zipped.forall(_.unify(_))
      } && valueParameterTypesFound.length == valueParameterTypes.length && {
        valueParameterTypesFound.foreach(_.clear())
        (valueParameterTypesFound, valueParameterTypes).zipped.forall(_.unify(_))
      }
    }.toSeq match {
      case Seq() => None
      case Seq(kv) => Some(kv)
      case _ => fatal(s"Multiple functions found that satisfy $name(${valueParameterTypes.mkString(",")}).")
    }
  }

  def lookupSeeded(name: String, seed: Long, returnType: Type, arguments: Seq[Type]): Option[(Seq[IR]) => IR] = {
    lookupFunction(name, returnType, Array.empty[Type], arguments)
      .filter(_.isInstanceOf[SeededJVMFunction])
      .map { case f: SeededJVMFunction =>
        (irArguments: Seq[IR]) => ApplySeeded(name, irArguments, seed, f.returnType.subst())
      }
  }

  def lookupUnseeded(name: String, returnType: Type, arguments: Seq[Type]): Option[(Seq[Type], Seq[IR]) => IR] =
    lookupUnseeded(name, returnType, Array.empty[Type], arguments)

  def lookupUnseeded(name: String, returnType: Type, typeParameters: Seq[Type], arguments: Seq[Type]): Option[(Seq[Type], Seq[IR]) => IR] = {
    val validIR: Option[(Seq[Type], Seq[IR]) => IR] = lookupIR(name, returnType, typeParameters, arguments).map {
      case ((_, _, _, inline), conversion) => (typeParametersPassed, args) =>
        val x = ApplyIR(name, typeParametersPassed, args)
        x.conversion = conversion
        x.inline = inline
        x
    }

    val validMethods = lookupFunction(name, returnType, typeParameters, arguments)
      .filter(!_.isInstanceOf[SeededJVMFunction]).map { f =>
        { (irValueParametersTypes: Seq[Type], irArguments: Seq[IR]) =>
          f match {
            case _: UnseededMissingnessObliviousJVMFunction =>
              Apply(name, irValueParametersTypes, irArguments, f.returnType.subst())
            case _: UnseededMissingnessAwareJVMFunction =>
              ApplySpecial(name, irValueParametersTypes, irArguments, f.returnType.subst())
          }
        }
      }

    (validIR, validMethods) match {
      case (None   , None)    => None
      case (None   , Some(x)) => Some(x)
      case (Some(x), None)    => Some(x)
      case _ => fatal(s"Multiple methods found that satisfy $name(${ arguments.mkString(",") }).")
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
    ExperimentalFunctions,
    ReferenceGenomeFunctions
  ).foreach(_.registerAll())

  def dumpFunctions(): Unit = {
    def dtype(t: Type): String = s"""dtype("${ StringEscapeUtils.escapeString(t.toString) }\")"""

    irRegistry.foreach { case (name, fns) =>
        fns.foreach { case ((typeParameters, valueParameterTypes, returnType, _), f) =>
          println(s"""register_function("${ StringEscapeUtils.escapeString(name) }", (${ typeParameters.map(dtype).mkString(",") }), (${ valueParameterTypes.map(dtype).mkString(",") }), ${ dtype(returnType) })""")
        }
    }

    jvmRegistry.foreach { case (name, fns) =>
        fns.foreach { f =>
          println(s"""${
            if (f.isInstanceOf[SeededJVMFunction])
              "register_seeded_function"
            else
              "register_function"
          }("${ StringEscapeUtils.escapeString(name) }", (${ f.typeParameters.map(dtype).mkString(",") }), (${ f.valueParameterTypes.map(dtype).mkString(",") }), ${ dtype(f.returnType) })""")
        }
    }
  }
}

object RegistryHelpers {
  def stupidUnwrapStruct(r: Region, value: Row, ptype: PType): Long = {
    assert(value != null)
    ptype.unstagedStoreJavaObject(value, r)
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
    case t: PLocus => c => EmitCodeBuilder.scopedCode(r.mb)(cb => PCode(t, c).asLocus.getLocusObj(cb))
    case _ => c =>
      Code.invokeScalaObject3[PType, Region, Long, Any](
        UnsafeRow.getClass, "read",
        r.mb.getPType(t),
        r.region, coerce[Long](c))
  }

  def boxedTypeInfo(t: Type): TypeInfo[_ >: Null] = t match {
    case TBoolean => classInfo[java.lang.Boolean]
    case TInt32 => classInfo[java.lang.Integer]
    case TInt64 => classInfo[java.lang.Long]
    case TFloat32 => classInfo[java.lang.Float]
    case TFloat64 => classInfo[java.lang.Double]
    case TCall => classInfo[java.lang.Integer]
    case TString => classInfo[java.lang.String]
    case _: TLocus => classInfo[Locus]
    case _ => classInfo[AnyRef]
  }

  def scodeToJavaValue(cb: EmitCodeBuilder, r: Value[Region], sc: SCode): Code[AnyRef] = {
    sc.st match {
      case _: SInt32 => Code.boxInt(sc.asInt32.intCode(cb))
      case _: SInt64 => Code.boxLong(sc.asInt64.longCode(cb))
      case _: SFloat32 => Code.boxFloat(sc.asFloat32.floatCode(cb))
      case _: SFloat64 => Code.boxDouble(sc.asFloat64.doubleCode(cb))
      case _: SBoolean => Code.boxBoolean(sc.asBoolean.boolCode(cb))
      case _: SCall => Code.boxInt(coerce[Int](sc.asPCode.code))
      case _: SString => sc.asString.loadString()
      case _: SLocus => sc.asLocus.getLocusObj(cb)
      case t =>
        val pt = t.canonicalPType()
        val addr = pt.store(cb, r, sc, deepCopy = false)
        Code.invokeScalaObject3[PType, Region, Long, AnyRef](
          UnsafeRow.getClass, "readAnyRef",
          cb.emb.getPType(pt),
          r, addr)

    }
  }

  def boxArg(r: EmitRegion, t: PType): Code[_] => Code[AnyRef] = t match {
    case _: PBoolean => c => Code.boxBoolean(coerce[Boolean](c))
    case _: PInt32 => c => Code.boxInt(coerce[Int](c))
    case _: PInt64 => c => Code.boxLong(coerce[Long](c))
    case _: PFloat32 => c => Code.boxFloat(coerce[Float](c))
    case _: PFloat64 => c => Code.boxDouble(coerce[Double](c))
    case _: PCall => c => Code.boxInt(coerce[Int](c))
    case t: PString => c => t.loadString(coerce[Long](c))
    case t: PLocus => c => EmitCodeBuilder.scopedCode(r.mb)(cb => PCode(t, c).asLocus.getLocusObj(cb))
    case _ => c =>
      Code.invokeScalaObject3[PType, Region, Long, AnyRef](
        UnsafeRow.getClass, "readAnyRef",
        r.mb.getPType(t),
        r.region, coerce[Long](c))
  }

  def unwrapReturn(cb: EmitCodeBuilder, r: Value[Region], pt: PType, value: Code[_]): PCode = pt.virtualType match {
    case TBoolean => PCode(pt, value)
    case TInt32 => PCode(pt, value)
    case TInt64 => PCode(pt, value)
    case TFloat32 => PCode(pt, value)
    case TFloat64 => PCode(pt, value)
    case TString =>
      val st = SStringPointer(pt.asInstanceOf[PCanonicalString])
      st.constructFromString(cb, r, coerce[String](value))
    case TCall => PCode(pt, value)
    case TArray(TInt32) =>
      val pca = pt.asInstanceOf[PCanonicalArray]
      val arr = cb.newLocal[IndexedSeq[Int]]("unrwrap_return_array_int32_arr", coerce[IndexedSeq[Int]](value))
      val len = cb.newLocal[Int]("unwrap_return_array_int32_len", arr.invoke[Int]("length"))
      pca.constructFromElements(cb, r, len, deepCopy = false) { (cb, idx) =>
        val elt = cb.newLocal[java.lang.Integer]("unwrap_return_array_int32_elt",
          Code.checkcast[java.lang.Integer](arr.invoke[Int, java.lang.Object]("apply", idx)))
        IEmitCode(cb, elt.isNull, primitive(elt.invoke[Int]("intValue")))
      }
    case TArray(TFloat64) =>
      val pca = pt.asInstanceOf[PCanonicalArray]
      val arr = cb.newLocal[IndexedSeq[Double]]("unrwrap_return_array_float64_arr", coerce[IndexedSeq[Double]](value))
      val len = cb.newLocal[Int]("unwrap_return_array_float64_len", arr.invoke[Int]("length"))
      pca.constructFromElements(cb, r, len, deepCopy = false) { (cb, idx) =>
        val elt = cb.newLocal[java.lang.Double]("unwrap_return_array_float64_elt",
          Code.checkcast[java.lang.Double](arr.invoke[Int, java.lang.Object]("apply", idx)))
        IEmitCode(cb, elt.isNull, primitive(elt.invoke[Double]("doubleValue")))
      }
    case TArray(TString) =>
      val pca = pt.asInstanceOf[PCanonicalArray]
      val arr = cb.newLocal[IndexedSeq[String]]("unrwrap_return_array_str_arr", coerce[IndexedSeq[String]](value))
      val len = cb.newLocal[Int]("unwrap_return_array_str_len", arr.invoke[Int]("length"))
      pca.constructFromElements(cb, r, len, deepCopy = false) { (cb, idx) =>
        val st = SStringPointer(pca.elementType.asInstanceOf[PCanonicalString])
        val elt = cb.newLocal[String]("unwrap_return_array_str_elt",
          Code.checkcast[String](arr.invoke[Int, java.lang.Object]("apply", idx)))
        IEmitCode(cb, elt.isNull, st.constructFromString(cb, r, elt))
      }
    case t: TBaseStruct =>
      val addr = Code.invokeScalaObject3[Region, Row, PType, Long](
        RegistryHelpers.getClass, "stupidUnwrapStruct", r.region, coerce[Row](value), cb.emb.ecb.getPType(pt))
      new SBaseStructPointerCode(SBaseStructPointer(pt.asInstanceOf[PBaseStruct]), addr)
  }

  def registerPCode(
    name: String,
    valueParameterTypes: Array[Type],
    returnType: Type,
    calculateReturnPType: (Type, Seq[PType]) => PType,
    typeParameters: Array[Type] = Array.empty
  )(
    impl: (EmitRegion, EmitCodeBuilder, Seq[Type], PType, Array[PCode]) => PCode
  ) {
    IRFunctionRegistry.addJVMFunction(
      new UnseededMissingnessObliviousJVMFunction(name, typeParameters, valueParameterTypes, returnType, calculateReturnPType) {
        override def apply(r: EmitRegion, cb: EmitCodeBuilder, returnPType: PType, typeParameters: Seq[Type], args: PCode*): PCode =
          impl(r, cb, typeParameters, returnPType, args.toArray)
        override def apply(r: EmitRegion, cb: EmitCodeBuilder, returnPType: PType, typeParameters: Seq[Type], args: (PType, Code[_])*): Code[_] = {
          assert(unify(typeParameters, args.map(_._1.virtualType), returnPType.virtualType))
          apply(r, cb, returnPType, typeParameters, args.map { case (t, a) => PCode(t, a) }: _*).code
        }
      })
  }

  def registerCode(
    name: String,
    valueParameterTypes: Array[Type],
    returnType: Type,
    calculateReturnPType: (Type, Seq[PType]) => PType,
    typeParameters: Array[Type] = Array.empty
  )(
    impl: (EmitRegion, EmitCodeBuilder, PType, Array[Type], Array[(PType, Code[_])]) => Code[_]
  ) {
    IRFunctionRegistry.addJVMFunction(
      new UnseededMissingnessObliviousJVMFunction(name, typeParameters, valueParameterTypes, returnType, calculateReturnPType) {
        override def apply(r: EmitRegion, cb: EmitCodeBuilder, returnPType: PType, typeParameters: Seq[Type], args: (PType, Code[_])*): Code[_] = {
          assert(unify(typeParameters, args.map(_._1.virtualType), returnPType.virtualType))
          impl(r, cb, returnPType, typeParameters.toArray, args.toArray)
        }
      })
  }

  def registerEmitCode(
    name: String,
    valueParameterTypes: Array[Type],
    returnType: Type,
    calculateReturnPType: (Type, Seq[PType]) => PType,
    typeParameters: Array[Type] = Array.empty
  )(
    impl: (EmitRegion,PType, Array[EmitCode]) => EmitCode
  ) {
    IRFunctionRegistry.addJVMFunction(
      new UnseededMissingnessAwareJVMFunction(name, typeParameters, valueParameterTypes, returnType, calculateReturnPType) {
        override def apply(r: EmitRegion, rpt: PType, typeParameters: Seq[Type], args: EmitCode*): EmitCode = {
          assert(unify(typeParameters, args.map(_.pt.virtualType), rpt.virtualType))
          impl(r, rpt, args.toArray)
        }
      })
  }

  def registerIEmitCode(
    name: String,
    valueParameterTypes: Array[Type],
    returnType: Type,
    calculateReturnPType: (Type, Seq[PType]) => PType,
    typeParameters: Array[Type] = Array.empty
  )(
    impl: (EmitCodeBuilder, Value[Region], PType, Array[EmitCode]) => IEmitCode
  ) {
    IRFunctionRegistry.addJVMFunction(
      new UnseededMissingnessAwareJVMFunction(name, typeParameters, valueParameterTypes, returnType, calculateReturnPType) {
        override def apply(
          cb: EmitCodeBuilder,
          r: Value[Region],
          rpt: PType,
          typeParameters: Seq[Type],
          args: EmitCode*
        ): IEmitCode = impl(cb, r, rpt, args.toArray)

        override def apply(r: EmitRegion, rpt: PType, typeParameters: Seq[Type], args: EmitCode*): EmitCode = {
          EmitCode.fromI(r.mb) { cb =>
            apply(cb, r.region, rpt, typeParameters, args: _*)
          }
        }
      })
  }

  def registerScalaFunction(
    name: String,
    valueParameterTypes: Array[Type],
    returnType: Type,
    calculateReturnPType: (Type, Seq[PType]) => PType
  )(
    cls: Class[_],
    method: String
  ) {
    registerCode(name, valueParameterTypes, returnType, calculateReturnPType) { case (r, cb, rt, _, args) =>
      val cts = valueParameterTypes.map(TypeToIRIntermediateClassTag(_).runtimeClass)
      Code.invokeScalaObject(cls, method, cts, args.map(_._2))(TypeToIRIntermediateClassTag(returnType))
    }
  }

  def registerWrappedScalaFunction(
    name: String,
    valueParameterTypes: Array[Type],
    returnType: Type,
    calculateReturnPType: (Type, Seq[PType]) => PType
  )(
    cls: Class[_],
    method: String
  ) {
    def ct(typ: Type): ClassTag[_] = typ match {
      case TString => classTag[String]
      case TArray(TInt32) => classTag[IndexedSeq[Int]]
      case TArray(TFloat64) => classTag[IndexedSeq[Double]]
      case TArray(TString) => classTag[IndexedSeq[String]]
      case TSet(TString) => classTag[Set[String]]
      case TDict(TString, TString) => classTag[Map[String, String]]
      case t => TypeToIRIntermediateClassTag(t)
    }

    registerCode(name, valueParameterTypes, returnType, calculateReturnPType) { case (r, cb, rt, _, args) =>
      val cts = valueParameterTypes.map(ct(_).runtimeClass)
      val out = Code.invokeScalaObject(cls, method, cts, args.map { case (t, a) => wrapArg(r, t)(a) })(ct(returnType))
      unwrapReturn(cb, r.region, rt, out).code
    }
  }

  def registerWrappedScalaFunction1(name: String, a1: Type, returnType: Type,  pt: (Type, PType) => PType)(cls: Class[_], method: String): Unit =
    registerWrappedScalaFunction(name, Array(a1), returnType, unwrappedApply(pt))(cls, method)

  def registerWrappedScalaFunction2(name: String, a1: Type, a2: Type, returnType: Type, pt: (Type, PType, PType) => PType)(cls: Class[_], method: String): Unit =
    registerWrappedScalaFunction(name, Array(a1, a2), returnType, unwrappedApply(pt))(cls, method)

  def registerWrappedScalaFunction3(name: String, a1: Type, a2: Type, a3: Type, returnType: Type,
    pt: (Type, PType, PType, PType) => PType)(cls: Class[_], method: String): Unit =
    registerWrappedScalaFunction(name, Array(a1, a2, a3), returnType, unwrappedApply(pt))(cls, method)

  def registerJavaStaticFunction(name: String, valueParameterTypes: Array[Type], returnType: Type, pt: (Type, Seq[PType]) => PType)(cls: Class[_], method: String) {
    registerCode(name, valueParameterTypes, returnType, pt) { case (r, cb, rt, _, args) =>
      val cts = valueParameterTypes.map(TypeToIRIntermediateClassTag(_).runtimeClass)
      Code.invokeStatic(cls, method, cts, args.map(_._2))(TypeToIRIntermediateClassTag(returnType))
    }
  }

  def registerIR(name: String, valueParameterTypes: Array[Type], returnType: Type, inline: Boolean = false, typeParameters: Array[Type] = Array.empty)(f: (Seq[Type], Seq[IR]) => IR): Unit =
    IRFunctionRegistry.addIR(name, typeParameters, valueParameterTypes, returnType, inline, f)

  def registerPCode1(name: String, mt1: Type, rt: Type, pt: (Type, PType) => PType)(impl: (EmitRegion, EmitCodeBuilder, PType, PCode) => PCode): Unit =
    registerPCode(name, Array(mt1), rt, unwrappedApply(pt)) {
      case (r, cb, _, rt, Array(a1)) => impl(r, cb, rt, a1)
    }

  def registerPCode2(name: String, mt1: Type, mt2: Type, rt: Type, pt: (Type, PType, PType) => PType)
    (impl: (EmitRegion, EmitCodeBuilder, PType, PCode, PCode) => PCode): Unit =
    registerPCode(name, Array(mt1, mt2), rt, unwrappedApply(pt)) {
      case (r, cb, _, rt, Array(a1, a2)) => impl(r, cb, rt, a1, a2)
    }

  def registerPCode3(name: String, mt1: Type, mt2: Type, mt3: Type, rt: Type, pt: (Type, PType, PType, PType) => PType)
    (impl: (EmitRegion, EmitCodeBuilder, PType, PCode, PCode, PCode) => PCode): Unit =
    registerPCode(name, Array(mt1, mt2, mt3), rt, unwrappedApply(pt)) {
      case (r, cb, _, rt, Array(a1, a2, a3)) => impl(r, cb, rt, a1, a2, a3)
    }

  def registerPCode4(name: String, mt1: Type, mt2: Type, mt3: Type, mt4: Type, rt: Type, pt: (Type, PType, PType, PType, PType) => PType)
    (impl: (EmitRegion, EmitCodeBuilder, PType, PCode, PCode, PCode, PCode) => PCode): Unit =
    registerPCode(name, Array(mt1, mt2, mt3, mt4), rt, unwrappedApply(pt)) {
      case (r, cb, _, rt, Array(a1, a2, a3, a4)) => impl(r, cb, rt, a1, a2, a3, a4)
    }

  def registerPCode4t(name: String, typeParams: Array[Type], mt1: Type, mt2: Type, mt3: Type, mt4: Type, rt: Type,
    pt: (Type, PType, PType, PType, PType) => PType)
    (impl: (EmitRegion, EmitCodeBuilder, Seq[Type], PType, PCode, PCode, PCode, PCode) => PCode): Unit =
    registerPCode(name, Array(mt1, mt2, mt3, mt4), rt, unwrappedApply(pt), typeParams) {
      case (r, cb, typeParams, rt, Array(a1, a2, a3, a4)) => impl(r, cb, typeParams, rt, a1, a2, a3, a4)
    }


  def registerPCode5(name: String, mt1: Type, mt2: Type, mt3: Type, mt4: Type, mt5: Type, rt: Type, pt: (Type, PType, PType, PType, PType, PType) => PType)
    (impl: (EmitRegion, EmitCodeBuilder, PType, PCode, PCode, PCode, PCode, PCode) => PCode): Unit =
    registerPCode(name, Array(mt1, mt2, mt3, mt4, mt5), rt, unwrappedApply(pt)) {
      case (r, cb, _, rt, Array(a1, a2, a3, a4, a5)) => impl(r, cb, rt, a1, a2, a3, a4, a5)
    }

  def registerCode1[A1](name: String, mt1: Type, rt: Type, pt: (Type, PType) => PType)(impl: (EmitRegion, PType, (PType, Code[A1])) => Code[_]): Unit =
    registerCode(name, Array(mt1), rt, unwrappedApply(pt)) {
      case (r, xb, rt, _, Array(a1: (PType, Code[A1]) @unchecked)) => impl(r, rt, a1)
    }

  def registerCode1t[A1](name: String, typeParam: Type, mt1: Type, rt: Type, pt: (Type, PType) => PType)(impl: (EmitRegion, PType, Type, (PType, Code[A1])) => Code[_]): Unit =
    registerCode(name, Array(mt1), rt, unwrappedApply(pt), typeParameters = Array(typeParam)) {
      case (r, cb, rt, Array(t), Array(a1: (PType, Code[A1]) @unchecked)) => impl(r, rt, t, a1)
    }


  def registerCode2[A1, A2](name: String, mt1: Type, mt2: Type, rt: Type, pt: (Type, PType, PType) => PType)
    (impl: (EmitRegion, PType, (PType, Code[A1]), (PType, Code[A2])) => Code[_]): Unit =
    registerCode(name, Array(mt1, mt2), rt, unwrappedApply(pt)) {
      case (r, cb, rt, _, Array(
      a1: (PType, Code[A1]) @unchecked,
      a2: (PType, Code[A2]) @unchecked)) => impl(r, rt, a1, a2)
    }

  def registerCode2t[A1, A2](name: String, typeParam1: Type, arg1: Type, arg2: Type, rt: Type, pt: (Type, PType, PType) => PType)
    (impl: (EmitRegion, PType, Type, (PType, Code[A1]), (PType, Code[A2])) => Code[_]): Unit =
    registerCode(name, Array(arg1, arg2), rt, unwrappedApply(pt), Array(typeParam1)) {
      case (r, cb, rt, Array(t1), Array(a1: (PType, Code[A1]) @unchecked, a2: (PType, Code[A2]) @unchecked)) => impl(r, rt, t1, a1, a2)
    }

  def registerCode3[A1, A2, A3](name: String, mt1: Type, mt2: Type, mt3: Type, rt: Type, pt: (Type, PType, PType, PType) => PType)
    (impl: (EmitRegion, PType, (PType, Code[A1]), (PType, Code[A2]), (PType, Code[A3])) => Code[_]): Unit =
    registerCode(name, Array(mt1, mt2, mt3), rt, unwrappedApply(pt)) {
      case (r, cb, rt, _, Array(
      a1: (PType, Code[A1]) @unchecked,
      a2: (PType, Code[A2]) @unchecked,
      a3: (PType, Code[A3]) @unchecked)) => impl(r, rt, a1, a2, a3)
    }

  def registerIEmitCode1(name: String, mt1: Type, rt: Type, pt: (Type, PType) => PType)
    (impl: (EmitCodeBuilder, Value[Region], PType, EmitCode) => IEmitCode): Unit =
    registerIEmitCode(name, Array(mt1), rt, unwrappedApply(pt)) { case (cb, r, rt, Array(a1)) =>
      impl(cb, r, rt, a1)
    }

  def registerIEmitCode2(name: String, mt1: Type, mt2: Type, rt: Type, pt: (Type, PType, PType) => PType)
    (impl: (EmitCodeBuilder, Value[Region], PType, EmitCode, EmitCode) => IEmitCode): Unit =
    registerIEmitCode(name, Array(mt1, mt2), rt, unwrappedApply(pt)) { case (cb, r, rt, Array(a1, a2)) =>
      impl(cb, r, rt, a1, a2)
    }

  def registerIEmitCode4(name: String, mt1: Type, mt2: Type, mt3: Type, mt4: Type, rt: Type, pt: (Type, PType, PType, PType, PType) => PType)
    (impl: (EmitCodeBuilder, Value[Region], PType, EmitCode, EmitCode, EmitCode, EmitCode) => IEmitCode): Unit =
    registerIEmitCode(name, Array(mt1, mt2, mt3, mt4), rt, unwrappedApply(pt)) { case (cb, r, rt, Array(a1, a2, a3, a4)) =>
      impl(cb, r, rt, a1, a2, a3, a4)
    }

  def registerIEmitCode6(name: String, mt1: Type, mt2: Type, mt3: Type, mt4: Type, mt5: Type, mt6: Type, rt: Type, pt: (Type, PType, PType, PType, PType, PType, PType) => PType)
    (impl: (EmitCodeBuilder, Value[Region], PType, EmitCode, EmitCode, EmitCode, EmitCode, EmitCode, EmitCode) => IEmitCode): Unit =
    registerIEmitCode(name, Array(mt1, mt2, mt3, mt4, mt5, mt6), rt, unwrappedApply(pt)) { case (cb, r, rt, Array(a1, a2, a3, a4, a5, a6)) =>
      impl(cb, r, rt, a1, a2, a3, a4, a5, a6)
    }

  def registerEmitCode2(name: String, mt1: Type, mt2: Type, rt: Type, pt: (Type, PType, PType) => PType)
    (impl: (EmitRegion, PType, EmitCode, EmitCode) => EmitCode): Unit =
    registerEmitCode(name, Array(mt1, mt2), rt, unwrappedApply(pt)) { case (r, rt, Array(a1, a2)) => impl(r, rt, a1, a2) }

  def registerIR1(name: String, mt1: Type, returnType: Type, typeParameters: Array[Type] = Array.empty)(f: (Seq[Type], IR) => IR): Unit =
    registerIR(name, Array(mt1), returnType, typeParameters = typeParameters) { case (t, Seq(a1)) => f(t, a1) }

  def registerIR2(name: String, mt1: Type, mt2: Type, returnType: Type, typeParameters: Array[Type] = Array.empty)(f: (Seq[Type], IR, IR) => IR): Unit =
    registerIR(name, Array(mt1, mt2), returnType, typeParameters = typeParameters) { case (t, Seq(a1, a2)) => f(t, a1, a2) }

  def registerIR3(name: String, mt1: Type, mt2: Type, mt3: Type, returnType: Type, typeParameters: Array[Type] = Array.empty)(f: (Seq[Type], IR, IR, IR) => IR): Unit =
    registerIR(name, Array(mt1, mt2, mt3), returnType, typeParameters = typeParameters) { case (t, Seq(a1, a2, a3)) => f(t, a1, a2, a3) }

  def registerIR4(name: String, mt1: Type, mt2: Type, mt3: Type, mt4: Type, returnType: Type, typeParameters: Array[Type] = Array.empty)(f: (Seq[Type], IR, IR, IR, IR) => IR): Unit =
    registerIR(name, Array(mt1, mt2, mt3, mt4), returnType, typeParameters = typeParameters) { case (t, Seq(a1, a2, a3, a4)) => f(t, a1, a2, a3, a4) }

  def registerSeeded(
    name: String,
    valueParameterTypes: Array[Type],
    returnType: Type,
    calculateReturnPType: (Type, Seq[PType]) => PType
  )(
    impl: (EmitCodeBuilder, Value[Region], PType, Long, Array[SCode]) => SCode
  ) {
    IRFunctionRegistry.addJVMFunction(
      new SeededMissingnessObliviousJVMFunction(name, valueParameterTypes, returnType, calculateReturnPType) {
        val isDeterministic: Boolean = false

        def applySeeded(cb: EmitCodeBuilder, seed: Long, r: Value[Region], rpt: PType, args: SCode*): SCode = {
          assert(unify(Array.empty[Type], args.map(_.st.virtualType), rpt.virtualType))
          impl(cb, r, rpt, seed, args.toArray)
        }

        def applySeededI(seed: Long, cb: EmitCodeBuilder, r: Value[Region], rpt: PType, args: (PType, EmitCode)*): IEmitCode = {
          IEmitCode.multiMapEmitCodes(cb, args.map(_._2).toFastIndexedSeq) {
            argPCs => applySeeded(cb, seed, r, rpt, argPCs: _*).asPCode
          }
        }

        override val isStrict: Boolean = true
      })
  }

  def registerSeeded0(name: String, returnType: Type, pt: PType)(impl: (EmitCodeBuilder, Value[Region], PType, Long) => SCode): Unit =
    registerSeeded(name, Array[Type](), returnType, if (pt == null) null else (_: Type, _: Seq[PType]) => pt) { case (cb, r, rt, seed, _) => impl(cb, r, rt, seed) }

  def registerSeeded1(name: String, arg1: Type, returnType: Type, pt: (Type, PType) => PType)(impl: (EmitCodeBuilder, Value[Region], PType, Long, SCode) => SCode): Unit =
    registerSeeded(name, Array(arg1), returnType, unwrappedApply(pt)) {
      case (cb, r, rt, seed, Array(a1)) => impl(cb, r, rt, seed, a1)
    }

  def registerSeeded2(name: String, arg1: Type, arg2: Type, returnType: Type, pt: (Type, PType, PType) => PType)
    (impl: (EmitCodeBuilder, Value[Region], PType, Long, SCode, SCode) => SCode): Unit =
    registerSeeded(name, Array(arg1, arg2), returnType, unwrappedApply(pt)) { case
      (cb, r, rt, seed, Array(a1, a2)) =>
      impl(cb, r, rt, seed, a1, a2)
    }

  def registerSeeded4(name: String, arg1: Type, arg2: Type, arg3: Type, arg4: Type, returnType: Type, pt: (Type, PType, PType, PType, PType) => PType)
    (impl: (EmitCodeBuilder, Value[Region], PType, Long, SCode, SCode, SCode, SCode) => SCode): Unit =
    registerSeeded(name, Array(arg1, arg2, arg3, arg4), returnType, unwrappedApply(pt)) {
      case (cb, r, rt, seed, Array(a1, a2, a3, a4)) => impl(cb, r, rt, seed, a1, a2, a3, a4)
    }
}

sealed abstract class JVMFunction {
  def name: String

  def typeParameters: Seq[Type]

  def valueParameterTypes: Seq[Type]

  def returnType: Type

  def returnPType(returnType: Type, valueParameterTypes: Seq[PType]): PType

  def apply(mb: EmitRegion, returnType: PType, typeParameters: Seq[Type], args: EmitCode*): EmitCode

  def getAsMethod[C](cb: EmitClassBuilder[C], rpt: PType, typeParameters: Seq[Type], args: PType*): EmitMethodBuilder[C] = ???

  override def toString: String = s"$name[${ typeParameters.mkString(", ") }](${ valueParameterTypes.mkString(", ") }): $returnType"

  def unify(typeArguments: Seq[Type], valueArgumentTypes: Seq[Type], returnTypeIn: Type): Boolean = {
    val concrete = (typeArguments ++ valueArgumentTypes) :+ returnTypeIn
    val types = (typeParameters ++ valueParameterTypes) :+ returnType
    types.length == concrete.length && {
      types.foreach(_.clear())
      types.zip(concrete).forall { case (i, j) => i.unify(j) }
    }
  }
}

object MissingnessObliviousJVMFunction {
  def returnPType(calculateReturnPType: (Type, Seq[PType]) => PType)(returnType: Type, valueParameterTypes: Seq[PType]): PType = {
    val returnPType =
      if (calculateReturnPType == null) PType.canonical(returnType)
      else calculateReturnPType(returnType, valueParameterTypes)
    returnPType.setRequired(valueParameterTypes.forall(_.required))
  }
}

abstract class UnseededMissingnessObliviousJVMFunction (
  override val name: String,
  override val typeParameters: Seq[Type],
  override val valueParameterTypes: Seq[Type],
  override val returnType: Type,
  missingnessObliviousReturnPType: (Type, Seq[PType]) => PType
) extends JVMFunction {
  override def returnPType(returnType: Type, valueParameterTypes: Seq[PType]): PType =
    MissingnessObliviousJVMFunction.returnPType(missingnessObliviousReturnPType)(returnType, valueParameterTypes)

  def apply(r: EmitRegion, cb: EmitCodeBuilder, returnPType: PType, typeParameters: Seq[Type], args: (PType, Code[_])*): Code[_]

  def apply(r: EmitRegion, cb: EmitCodeBuilder, returnPType: PType, typeParameters: Seq[Type], args: PCode*): PCode =
    PCode(returnPType, apply(r, cb, returnPType, typeParameters, args.map(pc => pc.pt -> pc.code): _*))

  def apply(r: EmitRegion, returnPType: PType, typeParameters: Seq[Type], args: EmitCode*): EmitCode = {
    EmitCode.fromI(r.mb)(cb => IEmitCode.multiMapEmitCodes(cb, args.toFastIndexedSeq) { args =>
      apply(r, cb, returnPType, typeParameters, args: _*)
    })
  }

  override def getAsMethod[C](cb: EmitClassBuilder[C], rpt: PType, typeParameters: Seq[Type], args: PType*): EmitMethodBuilder[C] = {
    val unified = unify(typeParameters, args.map(_.virtualType), rpt.virtualType)
    assert(unified)
    val argTIs = args.toFastIndexedSeq.map(typeToTypeInfo)
    val methodbuilder = cb.genEmitMethod(name, (typeInfo[Region] +: argTIs).map(ti => ti: CodeParamType), typeToTypeInfo(rpt))
    methodbuilder.emitWithBuilder(cb => apply(EmitRegion.default(methodbuilder),
      cb,
      rpt,
      typeParameters,
      args.zip(argTIs.zipWithIndex.map { case (ti, i) =>
        methodbuilder.getCodeParam(i + 2)(ti).get
      }): _*))
    methodbuilder
  }
}

object MissingnessAwareJVMFunction {
  def returnPType(calculateReturnPType: (Type, Seq[PType]) => PType)(returnType: Type, valueParameterTypes: Seq[PType]): PType=
    if (calculateReturnPType == null) PType.canonical(returnType)
    else calculateReturnPType(returnType, valueParameterTypes)
}

abstract class UnseededMissingnessAwareJVMFunction (
  override val name: String,
  override val typeParameters: Seq[Type],
  override val valueParameterTypes: Seq[Type],
  override val returnType: Type,
  missingnessAwareReturnPType: (Type, Seq[PType]) => PType
) extends JVMFunction {
  override def returnPType(returnType: Type, valueParameterTypes: Seq[PType]): PType =
    MissingnessAwareJVMFunction.returnPType(missingnessAwareReturnPType)(returnType, valueParameterTypes)

  def apply(cb: EmitCodeBuilder,
    r: Value[Region],
    rpt: PType,
    typeParameters: Seq[Type],
    args: EmitCode*
  ): IEmitCode = {
    ???
  }
}

abstract class SeededJVMFunction (
  override val name: String,
  override val valueParameterTypes: Seq[Type],
  override val returnType: Type
) extends JVMFunction {
  def typeParameters: Seq[Type] = Seq.empty[Type]

  private[this] var seed: Long = _

  def setSeed(s: Long): Unit = { seed = s }

  def applySeededI(seed: Long, cb: EmitCodeBuilder, region: Value[Region], rpt: PType, args: (PType, EmitCode)*): IEmitCode

  def apply(region: EmitRegion, rpt: PType, typeParameters: Seq[Type], args: EmitCode*): EmitCode =
    fatal("seeded functions must go through IEmitCode path")

  def apply(region: EmitRegion, rpt: PType, args: EmitCode*): EmitCode =
    fatal("seeded functions must go through IEmitCode path")

  def isStrict: Boolean = false
}

abstract class SeededMissingnessObliviousJVMFunction (
  override val name: String,
  override val valueParameterTypes: Seq[Type],
  override val returnType: Type,
  missingnessObliviousReturnPType: (Type, Seq[PType]) => PType
) extends SeededJVMFunction(name, valueParameterTypes, returnType) {
  override def returnPType(returnType: Type, valueParameterTypes: Seq[PType]): PType =
    MissingnessObliviousJVMFunction.returnPType(missingnessObliviousReturnPType)(returnType, valueParameterTypes)
}

abstract class SeededMissingnessAwareJVMFunction (
  override val name: String,
  override val valueParameterTypes: Seq[Type],
  override val returnType: Type,
  missingnessAwareReturnPType: (Type, Seq[PType]) => PType
) extends SeededJVMFunction(name, valueParameterTypes, returnType) {
  override def returnPType(returnType: Type, valueParameterTypes: Seq[PType]): PType =
    MissingnessAwareJVMFunction.returnPType(missingnessAwareReturnPType)(returnType, valueParameterTypes)
}
