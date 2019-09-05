package is.hail.expr.ir

import java.io._

import is.hail.annotations.{CodeOrdering, Region, RegionValueBuilder}
import is.hail.{HailContext, asm4s}
import is.hail.asm4s._
import is.hail.backend.BackendUtils
import is.hail.expr.Parser
import is.hail.expr.ir.functions.IRRandomness
import is.hail.expr.types.physical.{PTuple, PType}
import is.hail.expr.types.virtual.{TStruct, TTuple, Type}
import is.hail.io.{CodecSpec, Decoder, PackCodecSpec}
import is.hail.utils._
import is.hail.variant.ReferenceGenome
import is.hail.io.fs.FS
import org.apache.spark.TaskContext
import org.objectweb.asm.tree.AbstractInsnNode

import scala.collection.generic.Growable
import scala.collection.mutable
import scala.reflect.ClassTag

object EmitFunctionBuilder {
  def apply[R: TypeInfo](): EmitFunctionBuilder[AsmFunction0[R]] =
    new EmitFunctionBuilder[AsmFunction0[R]](Array[MaybeGenericTypeInfo[_]](), GenericTypeInfo[R])

  def apply[A: TypeInfo, R: TypeInfo]: EmitFunctionBuilder[AsmFunction1[A, R]] =
    new EmitFunctionBuilder[AsmFunction1[A, R]](Array(GenericTypeInfo[A]), GenericTypeInfo[R])

  def apply[A: TypeInfo, B: TypeInfo, R: TypeInfo]: EmitFunctionBuilder[AsmFunction2[A, B, R]] =
    new EmitFunctionBuilder[AsmFunction2[A, B, R]](Array(GenericTypeInfo[A], GenericTypeInfo[B]), GenericTypeInfo[R])

  def apply[A: TypeInfo, B: TypeInfo, C: TypeInfo, R: TypeInfo]: EmitFunctionBuilder[AsmFunction3[A, B, C, R]] =
    new EmitFunctionBuilder[AsmFunction3[A, B, C, R]](Array(GenericTypeInfo[A], GenericTypeInfo[B], GenericTypeInfo[C]), GenericTypeInfo[R])

  def apply[A: TypeInfo, B: TypeInfo, C: TypeInfo, D: TypeInfo, R: TypeInfo]: EmitFunctionBuilder[AsmFunction4[A, B, C, D, R]] =
    new EmitFunctionBuilder[AsmFunction4[A, B, C, D, R]](Array(GenericTypeInfo[A], GenericTypeInfo[B], GenericTypeInfo[C], GenericTypeInfo[D]), GenericTypeInfo[R])

  def apply[A: TypeInfo, B: TypeInfo, C: TypeInfo, D: TypeInfo, E: TypeInfo, R: TypeInfo]: EmitFunctionBuilder[AsmFunction5[A, B, C, D, E, R]] =
    new EmitFunctionBuilder[AsmFunction5[A, B, C, D, E, R]](Array(GenericTypeInfo[A], GenericTypeInfo[B], GenericTypeInfo[C], GenericTypeInfo[D], GenericTypeInfo[E]), GenericTypeInfo[R])

  def apply[A: TypeInfo, B: TypeInfo, C: TypeInfo, D: TypeInfo, E: TypeInfo, F: TypeInfo, R: TypeInfo]: EmitFunctionBuilder[AsmFunction6[A, B, C, D, E, F, R]] =
    new EmitFunctionBuilder[AsmFunction6[A, B, C, D, E, F, R]](Array(GenericTypeInfo[A], GenericTypeInfo[B], GenericTypeInfo[C], GenericTypeInfo[D], GenericTypeInfo[E], GenericTypeInfo[F]), GenericTypeInfo[R])

  def apply[A: TypeInfo, B: TypeInfo, C: TypeInfo, D: TypeInfo, E: TypeInfo, F: TypeInfo, G: TypeInfo, R: TypeInfo]: EmitFunctionBuilder[AsmFunction7[A, B, C, D, E, F, G, R]] =
    new EmitFunctionBuilder[AsmFunction7[A, B, C, D, E, F, G, R]](Array(GenericTypeInfo[A], GenericTypeInfo[B], GenericTypeInfo[C], GenericTypeInfo[D], GenericTypeInfo[E], GenericTypeInfo[F], GenericTypeInfo[G]), GenericTypeInfo[R])
}

trait FunctionWithFS {
  def addFS(fs: FS): Unit
}

trait FunctionWithAggRegion {
  def getAggOffset(): Long

  def setAggState(region: Region, offset: Long): Unit

  def newAggState(region: Region): Unit

  def setNumSerialized(i: Int): Unit

  def setSerializedAgg(i: Int, b: Array[Byte]): Unit

  def getSerializedAgg(i: Int): Array[Byte]
}

trait FunctionWithLiterals {
  def addLiterals(lit: Array[Byte], r: Region): Unit
}

trait FunctionWithSeededRandomness {
  def setPartitionIndex(idx: Int): Unit
}

trait FunctionWithBackend {
  def setBackend(spark: BackendUtils): Unit
}

class EmitMethodBuilder(
  override val fb: EmitFunctionBuilder[_],
  mname: String,
  parameterTypeInfo: Array[TypeInfo[_]],
  returnTypeInfo: TypeInfo[_]
) extends MethodBuilder(fb, mname, parameterTypeInfo, returnTypeInfo) {

  def numReferenceGenomes: Int = fb.numReferenceGenomes

  def getReferenceGenome(rg: ReferenceGenome): Code[ReferenceGenome] =
    fb.getReferenceGenome(rg)

  def numTypes: Int = fb.numTypes

  def getType(t: Type): Code[Type] = fb.getType(t)

  def getPType(t: PType): Code[PType] = fb.getPType(t)

  def getCodeOrdering[T](t: PType, op: CodeOrdering.Op): CodeOrdering.F[T] =
    getCodeOrdering[T](t, op, ignoreMissingness = false)

  def getCodeOrdering[T](t: PType, op: CodeOrdering.Op, ignoreMissingness: Boolean): CodeOrdering.F[T] =
    fb.getCodeOrdering[T](t, op, ignoreMissingness)

  def getCodeOrdering[T](t1: PType, t2: PType, op: CodeOrdering.Op): CodeOrdering.F[T] =
    fb.getCodeOrdering[T](t1, t2, op, ignoreMissingness = false)

  def getCodeOrdering[T](t1: PType, t2: PType, op: CodeOrdering.Op, ignoreMissingness: Boolean): CodeOrdering.F[T] =
    fb.getCodeOrdering[T](t1, t2, op, ignoreMissingness)

  def newRNG(seed: Long): Code[IRRandomness] = fb.newRNG(seed)
}

class DependentEmitFunction[F >: Null <: AnyRef : TypeInfo : ClassTag](
  parentfb: EmitFunctionBuilder[_],
  parameterTypeInfo: Array[MaybeGenericTypeInfo[_]],
  returnTypeInfo: MaybeGenericTypeInfo[_],
  packageName: String = "is/hail/codegen/generated"
) extends EmitFunctionBuilder[F](parameterTypeInfo, returnTypeInfo, packageName) with DependentFunction[F] {

  private[this] val rgMap: mutable.Map[ReferenceGenome, Code[ReferenceGenome]] =
    mutable.Map[ReferenceGenome, Code[ReferenceGenome]]()

  private[this] val typMap: mutable.Map[Type, Code[Type]] =
    mutable.Map[Type, Code[Type]]()

  private[this] val literalsMap: mutable.Map[(Type, Any), Code[_]] =
    mutable.Map[(Type, Any), Code[_]]()

  override def getReferenceGenome(rg: ReferenceGenome): Code[ReferenceGenome] =
    rgMap.getOrElseUpdate(rg, {
      val fromParent = parentfb.getReferenceGenome(rg)
      val field = addField[ReferenceGenome](fromParent)
      field.load()
    })

  override def getType(t: Type): Code[Type] =
    typMap.getOrElseUpdate(t, {
      val fromParent = parentfb.getType(t)
      val field = addField[Type](fromParent)
      field.load()
    })

  override def addLiteral(v: Any, t: Type, region: Code[Region]): Code[_] = {
    assert(v != null)
    literalsMap.getOrElseUpdate(t -> v, {
      val fromParent = parentfb.addLiteral(v, t, region)
      val ti: TypeInfo[_] = typeToTypeInfo(t)
      val field = addField(fromParent, dummy = true)(ti)
      field.load()
    })
  }
}

class EmitFunctionBuilder[F >: Null](
  parameterTypeInfo: Array[MaybeGenericTypeInfo[_]],
  returnTypeInfo: MaybeGenericTypeInfo[_],
  packageName: String = "is/hail/codegen/generated"
)(implicit interfaceTi: TypeInfo[F]) extends FunctionBuilder[F](parameterTypeInfo, returnTypeInfo, packageName) {

  private[this] val rgMap: mutable.Map[ReferenceGenome, Code[ReferenceGenome]] =
    mutable.Map[ReferenceGenome, Code[ReferenceGenome]]()

  private[this] val typMap: mutable.Map[Type, Code[Type]] =
    mutable.Map[Type, Code[Type]]()

  private[this] val pTypeMap: mutable.Map[PType, Code[PType]] = mutable.Map[PType, Code[PType]]()

  private[this] val compareMap: mutable.Map[(PType, PType, CodeOrdering.Op, Boolean), CodeOrdering.F[_]] =
    mutable.Map[(PType, PType, CodeOrdering.Op, Boolean), CodeOrdering.F[_]]()

  def numReferenceGenomes: Int = rgMap.size

  def getReferenceGenome(rg: ReferenceGenome): Code[ReferenceGenome] =
    rgMap.getOrElseUpdate(rg, newLazyField[ReferenceGenome](rg.codeSetup(this)))

  def numTypes: Int = typMap.size

  private[this] def addReferenceGenome(rg: ReferenceGenome): Code[Unit] = {
    val rgExists = Code.invokeScalaObject[String, Boolean](ReferenceGenome.getClass, "hasReference", const(rg.name))
    val addRG = Code.invokeScalaObject[ReferenceGenome, Unit](ReferenceGenome.getClass, "addReference", getReferenceGenome(rg))
    rgExists.mux(Code._empty, addRG)
  }

  private[this] val literalsMap: mutable.Map[(Type, Any), ClassFieldRef[_]] =
    mutable.Map[(Type, Any), ClassFieldRef[_]]()
  private[this] lazy val encLitField: ClassFieldRef[Array[Byte]] = newField[Array[Byte]]

  def addLiteral(v: Any, t: Type, region: Code[Region]): Code[_] = {
    assert(v != null)
    val f = literalsMap.getOrElseUpdate(t -> v, newField("literal")(typeToTypeInfo(t)))
    f.load()
  }

  private[this] def encodeLiterals(): Array[Byte] = {
    val literals = literalsMap.toArray
    val litType = PType.canonical(TTuple(literals.map { case ((t, _), _) => t }: _*)).asInstanceOf[PTuple]
    val spec = CodecSpec.defaultUncompressed.makeCodecSpec2(litType)

    val (litRType, dec) = spec.buildEmitDecoderF[Long](litType.virtualType, this)
    assert(litRType == litType)
    cn.interfaces.asInstanceOf[java.util.List[String]].add(typeInfo[FunctionWithLiterals].iname)
    val mb2 = new EmitMethodBuilder(this, "addLiterals", Array(typeInfo[Array[Byte]], typeInfo[Region]), typeInfo[Unit])
    val off = mb2.newLocal[Long]
    val storeFields = literals.zipWithIndex.map { case (((_, _), f), i) =>
      f.storeAny(mb2.getArg[Region](2).load().loadIRIntermediate(litType.types(i))(litType.fieldOffset(off, i)))
    }

    mb2.emit(Code(
      encLitField := mb2.getArg[Array[Byte]](1),
      off := dec(mb2.getArg[Region](2).load(),
        spec.buildCodeInputBuffer(Code.newInstance[ByteArrayInputStream, Array[Byte]](encLitField))),
      Code(storeFields: _*)
    ))
    methods.append(mb2)

    val baos = new ByteArrayOutputStream()
    val enc = spec.buildEncoder(litType, litType)(baos)
    Region.scoped { region =>
      val rvb = new RegionValueBuilder(region)
      rvb.start(litType)
      rvb.startTuple()
      literals.foreach { case ((typ, a), _) => rvb.addAnnotation(typ, a) }
      rvb.endTuple()
      enc.writeRegionValue(region, rvb.end())
    }
    enc.flush()
    enc.close()
    baos.toByteArray
  }

  private[this] var _hfs: FS = _
  private[this] var _hfield: ClassFieldRef[FS] = _

  private[this] var _mods: ArrayBuilder[(String, (Int, Region) => AsmFunction3[Region, Array[Byte], Array[Byte], Array[Byte]])] = new ArrayBuilder()
  private[this] var _backendField: ClassFieldRef[BackendUtils] = _

  private[this] var _aggSigs: Array[AggSignature2] = _
  private[this] var _aggRegion: ClassFieldRef[Region] = _
  private[this] var _aggOff: ClassFieldRef[Long] = _
  private[this] var _aggState: agg.StateContainer = _
  private[this] var _nSerialized: Int = 0
  private[this] var _aggSerialized: ClassFieldRef[Array[Array[Byte]]] = _

  def addAggStates(aggSigs: Array[AggSignature2]): (agg.StateContainer, Code[Long]) = {
    if (_aggSigs != null) {
      assert(aggSigs sameElements _aggSigs)
      return _aggState -> _aggOff
    }
    cn.interfaces.asInstanceOf[java.util.List[String]].add(typeInfo[FunctionWithAggRegion].iname)
    _aggSigs = aggSigs
    _aggRegion = newField[Region]
    _aggOff = newField[Long]
    _aggState = agg.StateContainer(aggSigs.map(a => agg.Extract.getAgg(a).createState(this)).toArray, _aggRegion)
    _aggSerialized = newField[Array[Array[Byte]]]

    val newF = new EmitMethodBuilder(this, "newAggState", Array(typeInfo[Region]), typeInfo[Unit])
    val setF = new EmitMethodBuilder(this, "setAggState", Array(typeInfo[Region], typeInfo[Long]), typeInfo[Unit])
    val getF = new EmitMethodBuilder(this, "getAggOffset", Array(), typeInfo[Long])
    val setNSer = new EmitMethodBuilder(this, "setNumSerialized", Array(typeInfo[Int]), typeInfo[Unit])
    val setSer = new EmitMethodBuilder(this, "setSerializedAgg", Array(typeInfo[Int], typeInfo[Array[Byte]]), typeInfo[Unit])
    val getSer = new EmitMethodBuilder(this, "getSerializedAgg", Array(typeInfo[Int]), typeInfo[Array[Byte]])

    methods += newF
    methods += setF
    methods += getF
    methods += setNSer
    methods += setSer
    methods += getSer

    newF.emit(
      Code(_aggRegion := newF.getArg[Region](1),
        _aggState.topRegion.setNumParents(aggSigs.length),
        _aggState.toCode((i, s) => s.createState),
        _aggOff := _aggRegion.load().allocate(_aggState.typ.alignment, _aggState.typ.byteSize),
        _aggState.newStates))

    setF.emit(
      Code(
        _aggRegion := setF.getArg[Region](1),
        _aggState.topRegion.setNumParents(aggSigs.length),
        _aggState.toCode((i, s) => s.createState),
        _aggOff := setF.getArg[Long](2),
        _aggState.load(0, _aggOff)))

    getF.emit(Code(_aggState.store(0, _aggOff), _aggOff))

    setNSer.emit(_aggSerialized := Code.newArray[Array[Byte]](setNSer.getArg[Int](1)))

    setSer.emit(_aggSerialized.load().update(setSer.getArg[Int](1), setSer.getArg[Array[Byte]](2)))

    getSer.emit(_aggSerialized.load()(getSer.getArg[Int](1)))

    _aggState -> _aggOff
  }

  def getSerializedAgg(i: Int): Code[Array[Byte]] = {
    if (_nSerialized <= i)
      _nSerialized = i + 1
    _aggSerialized.load()(i)
  }

  def setSerializedAgg(i: Int, b: Code[Array[Byte]]): Code[Unit] = {
    if (_nSerialized <= i)
      _nSerialized = i + 1
    _aggSerialized.load().update(i, b)
  }

  def backend(): Code[BackendUtils] = {
    if (_backendField == null) {
      cn.interfaces.asInstanceOf[java.util.List[String]].add(typeInfo[FunctionWithBackend].iname)
      val backendField = newField[BackendUtils]
      val mb = new EmitMethodBuilder(this, "setBackend", Array(typeInfo[BackendUtils]), typeInfo[Unit])
      methods.append(mb)
      mb.emit(backendField := mb.getArg[BackendUtils](1))
      _backendField = backendField
    }
    _backendField
  }

  def addModule(name: String, mod: (Int, Region) => AsmFunction3[Region, Array[Byte], Array[Byte], Array[Byte]]): Unit = {
    _mods += name -> mod
  }

  def getFS: Code[FS] = {
    if (_hfs == null) {
      cn.interfaces.asInstanceOf[java.util.List[String]].add(typeInfo[FunctionWithFS].iname)
      val confField = newField[FS]
      val mb = new EmitMethodBuilder(this, "addFS", Array(typeInfo[FS]), typeInfo[Unit])
      methods.append(mb)
      mb.emit(confField := mb.getArg[FS](1))
      _hfs = HailContext.sFS
      _hfield = confField
    }

    assert(_hfs == HailContext.sFS && _hfield != null)
    _hfield.load()
  }

  def getUnsafeReader(path: Code[String], checkCodec: Code[Boolean]): Code[InputStream] =
     getFS.invoke[String, Boolean, InputStream]("unsafeReader", path, checkCodec)

  def getUnsafeWriter(path: Code[String]): Code[OutputStream] =
    getFS.invoke[String, OutputStream]("unsafeWriter", path)

  def getPType(t: PType): Code[PType] = {
    val references = ReferenceGenome.getReferences(t.virtualType).toArray
    val setup = Code(Code(references.map(addReferenceGenome): _*),
      Code.invokeScalaObject[String, PType](
        IRParser.getClass, "parsePType", const(t.parsableString())))
    pTypeMap.getOrElseUpdate(t,
      newLazyField[PType](setup))
  }

  def getType(t: Type): Code[Type] = {
    val references = ReferenceGenome.getReferences(t).toArray
    val setup = Code(Code(references.map(addReferenceGenome): _*),
      Code.invokeScalaObject[String, Type](
        IRParser.getClass, "parseType", const(t.parsableString())))
    typMap.getOrElseUpdate(t,
      newLazyField[Type](setup))
  }

  def getCodeOrdering[T](t1: PType, t2: PType, op: CodeOrdering.Op, ignoreMissingness: Boolean): CodeOrdering.F[T] = {
    val f = compareMap.getOrElseUpdate((t1, t2, op, ignoreMissingness), {
      val ti = typeToTypeInfo(t1)
      val rt = if (op == CodeOrdering.compare) typeInfo[Int] else typeInfo[Boolean]

      val newMB = if (ignoreMissingness) {
        val newMB = newMethod(Array[TypeInfo[_]](ti, ti), rt)
        val ord = t1.codeOrdering(newMB, t2)
        val v1 = newMB.getArg(1)(ti)
        val v2 = newMB.getArg(3)(ti)
        val c: Code[_] = op match {
          case CodeOrdering.compare => ord.compareNonnull(coerce[ord.T](v1), coerce[ord.T](v2))
          case CodeOrdering.equiv => ord.equivNonnull(coerce[ord.T](v1), coerce[ord.T](v2))
          case CodeOrdering.lt => ord.ltNonnull(coerce[ord.T](v1), coerce[ord.T](v2))
          case CodeOrdering.lteq => ord.lteqNonnull(coerce[ord.T](v1), coerce[ord.T](v2))
          case CodeOrdering.gt => ord.gtNonnull(coerce[ord.T](v1), coerce[ord.T](v2))
          case CodeOrdering.gteq => ord.gteqNonnull(coerce[ord.T](v1), coerce[ord.T](v2))
          case CodeOrdering.neq => !ord.equivNonnull(coerce[ord.T](v1), coerce[ord.T](v2))
        }
        newMB.emit(c)
        newMB
      } else {
        val newMB = newMethod(Array[TypeInfo[_]](typeInfo[Boolean], ti, typeInfo[Boolean], ti), rt)
        val ord = t1.codeOrdering(newMB, t2)
        val m1 = newMB.getArg[Boolean](1)
        val v1 = newMB.getArg(2)(ti)
        val m2 = newMB.getArg[Boolean](3)
        val v2 = newMB.getArg(4)(ti)
        val c: Code[_] = op match {
          case CodeOrdering.compare => ord.compare((m1, coerce[ord.T](v1)), (m2, coerce[ord.T](v2)))
          case CodeOrdering.equiv => ord.equiv((m1, coerce[ord.T](v1)), (m2, coerce[ord.T](v2)))
          case CodeOrdering.lt => ord.lt((m1, coerce[ord.T](v1)), (m2, coerce[ord.T](v2)))
          case CodeOrdering.lteq => ord.lteq((m1, coerce[ord.T](v1)), (m2, coerce[ord.T](v2)))
          case CodeOrdering.gt => ord.gt((m1, coerce[ord.T](v1)), (m2, coerce[ord.T](v2)))
          case CodeOrdering.gteq => ord.gteq((m1, coerce[ord.T](v1)), (m2, coerce[ord.T](v2)))
          case CodeOrdering.neq => !ord.equiv((m1, coerce[ord.T](v1)), (m2, coerce[ord.T](v2)))
        }
        newMB.emit(c)
        newMB
      }
      val f = { (x: (Code[Boolean], Code[_]), y: (Code[Boolean], Code[_])) =>
        if (ignoreMissingness)
          newMB.invoke(x._2, y._2)
        else
          newMB.invoke(x._1, x._2, y._1, y._2)
      }
      f
    })
    (v1: (Code[Boolean], Code[_]), v2: (Code[Boolean], Code[_])) => coerce[T](f(v1, v2))
  }

  def getCodeOrdering[T](t: PType, op: CodeOrdering.Op, ignoreMissingness: Boolean): CodeOrdering.F[T] =
    getCodeOrdering[T](t, t, op, ignoreMissingness)

  override val apply_method: EmitMethodBuilder = {
    val m = new EmitMethodBuilder(this, "apply", parameterTypeInfo.map(_.base), returnTypeInfo.base)
    if (parameterTypeInfo.exists(_.isGeneric) || returnTypeInfo.isGeneric) {
      val generic = new MethodBuilder(this, "apply", parameterTypeInfo.map(_.generic), returnTypeInfo.generic)
      methods.append(generic)
      generic.emit(
        new Code[Unit] {
          def emit(il: Growable[AbstractInsnNode]) {
            returnTypeInfo.castToGeneric(
              m.invoke(parameterTypeInfo.zipWithIndex.map { case (ti, i) =>
                ti.castFromGeneric(generic.getArg(i + 1)(ti.generic))
              }: _*)).emit(il)
          }
        }
      )
    }
    m
  }

  def newMethod(prefix: String, argsInfo: Array[TypeInfo[_]], returnInfo: TypeInfo[_]): EmitMethodBuilder = {
    val mb = new EmitMethodBuilder(this, s"${prefix}_${ methods.size }", argsInfo, returnInfo)
    methods.append(mb)
    mb
  }

  override def newMethod(argsInfo: Array[TypeInfo[_]], returnInfo: TypeInfo[_]): EmitMethodBuilder =
    newMethod("method", argsInfo, returnInfo)

  override def newMethod[R: TypeInfo]: EmitMethodBuilder =
    newMethod(Array[TypeInfo[_]](), typeInfo[R])

  override def newMethod[A: TypeInfo, R: TypeInfo]: EmitMethodBuilder =
    newMethod(Array[TypeInfo[_]](typeInfo[A]), typeInfo[R])

  override def newMethod[A: TypeInfo, B: TypeInfo, R: TypeInfo]: EmitMethodBuilder =
    newMethod(Array[TypeInfo[_]](typeInfo[A], typeInfo[B]), typeInfo[R])

  override def newMethod[A: TypeInfo, B: TypeInfo, C: TypeInfo, R: TypeInfo]: EmitMethodBuilder =
    newMethod(Array[TypeInfo[_]](typeInfo[A], typeInfo[B], typeInfo[C]), typeInfo[R])

  override def newMethod[A: TypeInfo, B: TypeInfo, C: TypeInfo, D: TypeInfo, R: TypeInfo]: EmitMethodBuilder =
    newMethod(Array[TypeInfo[_]](typeInfo[A], typeInfo[B], typeInfo[C], typeInfo[D]), typeInfo[R])

  override def newMethod[A: TypeInfo, B: TypeInfo, C: TypeInfo, D: TypeInfo, E: TypeInfo, R: TypeInfo]: EmitMethodBuilder =
    newMethod(Array[TypeInfo[_]](typeInfo[A], typeInfo[B], typeInfo[C], typeInfo[D], typeInfo[E]), typeInfo[R])

  def newDependentFunction[A1: TypeInfo, A2: TypeInfo, R: TypeInfo]: DependentEmitFunction[AsmFunction2[A1, A2, R]] = {
    val df = new DependentEmitFunction[AsmFunction2[A1, A2, R]](
      this, Array(GenericTypeInfo[A1], GenericTypeInfo[A2]), GenericTypeInfo[R])
    children += df
    df
  }

  def newDependentFunction[A1: TypeInfo, A2: TypeInfo, A3: TypeInfo, R: TypeInfo]: DependentEmitFunction[AsmFunction3[A1, A2, A3, R]] = {
    val df = new DependentEmitFunction[AsmFunction3[A1, A2, A3, R]](
      this, Array(GenericTypeInfo[A1], GenericTypeInfo[A2], GenericTypeInfo[A3]), GenericTypeInfo[R])
    children += df
    df
  }

  val rngs: ArrayBuilder[(ClassFieldRef[IRRandomness], Code[IRRandomness])] = new ArrayBuilder()

  def makeRNGs() {
    cn.interfaces.asInstanceOf[java.util.List[String]].add(typeInfo[FunctionWithSeededRandomness].iname)

    val initialized = newField[Boolean]
    val mb = new EmitMethodBuilder(this, "setPartitionIndex", Array(typeInfo[Int]), typeInfo[Unit])
    methods += mb

    val rngFields = rngs.result()
    val initialize = Code(rngFields.map { case (field, initialization) =>
        field := initialization
    }: _*)

    val reseed = Code(rngFields.map { case (field, _) =>
      field.invoke[Int, Unit]("reset", mb.getArg[Int](1))
    }: _*)

    mb.emit(Code(
      initialized.mux(
        Code._empty,
        Code(initialize, initialized := true)),
      reseed))
  }

  def newRNG(seed: Long): Code[IRRandomness] = {
    val rng = newField[IRRandomness]
    rngs += rng -> Code.newInstance[IRRandomness, Long](seed)
    rng
  }

  def resultWithIndex(print: Option[PrintWriter] = None): (Int, Region) => F = {
    makeRNGs()
    val childClasses = children.result().map(f => (f.name.replace("/","."), f.classAsBytes(print)))

    val hasLiterals: Boolean = literalsMap.nonEmpty
    val literals: Array[Byte] = if (hasLiterals) encodeLiterals() else Array()

    val literalsBc = HailContext.get.backend.broadcast(literals)

    val bytes = classAsBytes(print)
    val n = name.replace("/",".")
    val localFS = _hfs

    val nSerializedAggs = _nSerialized

    val useBackend = _backendField != null
    val backend = if (useBackend) new BackendUtils(_mods.result()) else null

    assert(TaskContext.get() == null,
      "FunctionBuilder emission should happen on master, but happened on worker")

    new ((Int, Region) => F) with java.io.Serializable {
      @transient @volatile private var theClass: Class[_] = null

      def apply(idx: Int, region: Region): F = {
        try {
          if (theClass == null) {
            this.synchronized {
              if (theClass == null) {
                childClasses.foreach { case (fn, b) => loadClass(fn, b) }
                theClass = loadClass(n, bytes)
              }
            }
          }
          val f = theClass.newInstance().asInstanceOf[F]
          if (localFS != null)
            f.asInstanceOf[FunctionWithFS].addFS(localFS)
          if (useBackend)
            f.asInstanceOf[FunctionWithBackend].setBackend(backend)
          if (hasLiterals)
            f.asInstanceOf[FunctionWithLiterals].addLiterals(literalsBc.value, region)
          if (nSerializedAggs != 0)
            f.asInstanceOf[FunctionWithAggRegion].setNumSerialized(nSerializedAggs)
          f.asInstanceOf[FunctionWithSeededRandomness].setPartitionIndex(idx)
          f
        } catch {
          //  only triggers on classloader
          case e@(_: Exception | _: LinkageError) =>
            FunctionBuilder.bytesToBytecodeString(bytes, FunctionBuilder.stderrAndLoggerErrorOS)
            throw e
        }
      }
    }
  }
}
