package is.hail.expr.ir

import java.io._

import is.hail.{HailContext, lir}
import is.hail.annotations.{CodeOrdering, Region, RegionValueBuilder}
import is.hail.asm4s._
import is.hail.backend.BackendUtils
import is.hail.expr.ir.functions.IRRandomness
import is.hail.expr.types.physical.{PCanonicalTuple, PType}
import is.hail.expr.types.virtual.Type
import is.hail.io.{BufferSpec, TypedCodecSpec}
import is.hail.io.fs.FS
import is.hail.utils._
import is.hail.variant.ReferenceGenome
import org.apache.spark.TaskContext

import scala.collection.mutable
import scala.reflect.ClassTag

class EmitModuleBuilder(val modb: ModuleBuilder) extends WrappedModuleBuilder {
  def newEmitClass[C](name: String)(implicit cti: TypeInfo[C]): EmitClassBuilder[C] =
    new EmitClassBuilder(this, modb.newClass(name))

  def genEmitClass[C](baseName: String)(implicit cti: TypeInfo[C]): EmitClassBuilder[C] =
    newEmitClass[C](genName("C", baseName))
}

trait WrappedEmitModuleBuilder extends WrappedModuleBuilder {
  def emodb: EmitModuleBuilder

  def modb: ModuleBuilder

  def newEmitClass[C](name: String)(implicit cti: TypeInfo[C]): EmitClassBuilder[C] = emodb.newEmitClass[C](name)

  def genEmitClass[C](baseName: String)(implicit cti: TypeInfo[C]): EmitClassBuilder[C] = emodb.genEmitClass[C](baseName)
}

trait WrappedEmitClassBuilder[C] extends WrappedEmitModuleBuilder with WrappedClassBuilder[C] {
  def ecb: EmitClassBuilder[C]

  def emodb: EmitModuleBuilder = ecb.emodb

  def cb: ClassBuilder[C]

  def getFS: Code[FS] = ecb.getFS

  def getSerializedAgg(i: Int): Code[Array[Byte]] = ecb.getSerializedAgg(i)

  def setSerializedAgg(i: Int, b: Code[Array[Byte]]): Code[Unit] = ecb.setSerializedAgg(i, b)

  def backend(): Code[BackendUtils] = ecb.backend()

  def addModule(name: String, mod: (Int, Region) => AsmFunction3[Region, Array[Byte], Array[Byte], Array[Byte]]): Unit =
    ecb.addModule(name, mod)

  def wrapVoids(x: Seq[Code[Unit]], prefix: String, size: Int = 32): Code[Unit] = ecb.wrapVoids(x, prefix, size)

  def wrapVoidsWithArgs(x: Seq[Seq[Code[_]] => Code[Unit]],
    suffix: String,
    argTypes: IndexedSeq[TypeInfo[_]],
    args: IndexedSeq[Code[_]],
    size: Int = 32): Code[Unit] = ecb.wrapVoidsWithArgs(x, suffix, argTypes, args, size)

  def getReferenceGenome(rg: ReferenceGenome): Value[ReferenceGenome] = ecb.getReferenceGenome(rg)

  def partitionRegion: Settable[Region] = ecb.partitionRegion

  def addLiteral(v: Any, t: PType): Code[_] = ecb.addLiteral(v, t)

  def getPType(t: PType): Code[PType] = ecb.getPType(t)

  def getType(t: Type): Code[Type] = ecb.getType(t)

  def newEmitMethod(name: String, argsInfo: IndexedSeq[TypeInfo[_]], returnInfo: TypeInfo[_]): EmitMethodBuilder[C] =
    ecb.newEmitMethod(name, argsInfo, returnInfo)

  def newEmitMethod(name: String, argsInfo: IndexedSeq[MaybeGenericTypeInfo[_]], returnInfo: MaybeGenericTypeInfo[_]): EmitMethodBuilder[C] =
    ecb.newEmitMethod(name, argsInfo, returnInfo)

  def genEmitMethod(baseName: String, argsInfo: IndexedSeq[TypeInfo[_]], returnInfo: TypeInfo[_]): EmitMethodBuilder[C] =
    ecb.genEmitMethod(baseName, argsInfo, returnInfo)

  def getCodeOrdering(
    t1: PType, t2: PType, sortOrder: SortOrder, op: CodeOrdering.Op, ignoreMissingness: Boolean
  ): CodeOrdering.F[op.ReturnType] =
    ecb.getCodeOrdering(t1, t2, sortOrder, op, ignoreMissingness)

  def addAggStates(aggSigs: Array[AggStatePhysicalSignature]): agg.TupleAggregatorState = ecb.addAggStates(aggSigs)

  def genDependentFunction[F](baseName: String,
    maybeGenericParameterTypeInfo: IndexedSeq[MaybeGenericTypeInfo[_]],
    maybeGenericReturnTypeInfo: MaybeGenericTypeInfo[_])(implicit fti: TypeInfo[F]): DependentEmitFunction[F] =
    ecb.genDependentFunction(baseName, maybeGenericParameterTypeInfo, maybeGenericReturnTypeInfo)

  def newRNG(seed: Long): Value[IRRandomness] = ecb.newRNG(seed)

  def resultWithIndex(print: Option[PrintWriter] = None): (Int, Region) => C = ecb.resultWithIndex(print)

  def getOrDefineEmitMethod(
    baseName: String, key: Any, argsInfo: IndexedSeq[TypeInfo[_]], returnInfo: TypeInfo[_]
  )(body: EmitMethodBuilder[C] => Unit): EmitMethodBuilder[C] = ecb.getOrGenEmitMethod(baseName, key, argsInfo, returnInfo)(body)

  // derived functions
  def getCodeOrdering(t: PType, op: CodeOrdering.Op): CodeOrdering.F[op.ReturnType] =
    getCodeOrdering(t, t, sortOrder = Ascending, op, ignoreMissingness = false)

  def getCodeOrdering(t: PType, op: CodeOrdering.Op, ignoreMissingness: Boolean): CodeOrdering.F[op.ReturnType] =
    getCodeOrdering(t, t, sortOrder = Ascending, op, ignoreMissingness)

  def getCodeOrdering(t1: PType, t2: PType, op: CodeOrdering.Op): CodeOrdering.F[op.ReturnType] =
    getCodeOrdering(t1, t2, sortOrder = Ascending, op, ignoreMissingness = false)

  def getCodeOrdering(t1: PType, t2: PType, op: CodeOrdering.Op, ignoreMissingness: Boolean): CodeOrdering.F[op.ReturnType] =
    getCodeOrdering(t1, t2, sortOrder = Ascending, op, ignoreMissingness)

  def getCodeOrdering(
    t1: PType, t2: PType, sortOrder: SortOrder, op: CodeOrdering.Op
  ): CodeOrdering.F[op.ReturnType] =
    getCodeOrdering(t1, t2, sortOrder, op, ignoreMissingness = false)

  def genEmitMethod[R: TypeInfo](baseName: String): EmitMethodBuilder[C] =
    ecb.genEmitMethod[R](baseName)

  def genEmitMethod[A: TypeInfo, R: TypeInfo](baseName: String): EmitMethodBuilder[C] =
    ecb.genEmitMethod[A, R](baseName)

  def genEmitMethod[A1: TypeInfo, A2: TypeInfo, R: TypeInfo](baseName: String): EmitMethodBuilder[C] =
    ecb.genEmitMethod[A1, A2, R](baseName)

  def genEmitMethod[A1: TypeInfo, A2: TypeInfo, A3: TypeInfo, R: TypeInfo](baseName: String): EmitMethodBuilder[C] =
    ecb.genEmitMethod[A1, A2, A3, R](baseName)

  def geEmitMethod[A1: TypeInfo, A2: TypeInfo, A3: TypeInfo, A4: TypeInfo, R: TypeInfo](baseName: String): EmitMethodBuilder[C] =
    ecb.genEmitMethod[A1, A2, A3, A4, R](baseName)

  def genEmitMethod[A1: TypeInfo, A2: TypeInfo, A3: TypeInfo, A4: TypeInfo, A5: TypeInfo, R: TypeInfo](baseName: String): EmitMethodBuilder[C] =
    ecb.genEmitMethod[A1, A2, A3, A4, A5, R](baseName)

  def wrapInEmitMethod[R: TypeInfo](
    baseName: String,
    body: (EmitMethodBuilder[_]) => Code[R]): Code[R] = ecb.wrapInEmitMethod[R](baseName, body)

  def wrapInEmitMethod[A: TypeInfo, R: TypeInfo](
    baseName: String,
    body: (EmitMethodBuilder[_], Code[A]) => Code[R]
  ): Code[A] => Code[R] = ecb.wrapInEmitMethod[A, R](baseName, body)

  def wrapInEmitMethod[A1: TypeInfo, A2: TypeInfo, R: TypeInfo](
    baseName: String,
    body: (EmitMethodBuilder[_], Code[A1], Code[A2]) => Code[R]
  ): (Code[A1], Code[A2]) => Code[R] = ecb.wrapInEmitMethod[A1, A2, R](baseName, body)

  def getUnsafeReader(path: Code[String], checkCodec: Code[Boolean]): Code[InputStream] =
    getFS.invoke[String, Boolean, InputStream]("unsafeReader", path, checkCodec)

  def getUnsafeWriter(path: Code[String]): Code[OutputStream] =
    getFS.invoke[String, OutputStream]("unsafeWriter", path)

  def genDependentFunction[A1: TypeInfo, A2: TypeInfo, R: TypeInfo](
    baseName: String = null
  ): DependentEmitFunction[AsmFunction2[A1, A2, R]] =
    genDependentFunction[AsmFunction2[A1, A2, R]](baseName, Array(GenericTypeInfo[A1], GenericTypeInfo[A2]), GenericTypeInfo[R])

  def genDependentFunction[A1: TypeInfo, A2: TypeInfo, A3: TypeInfo, R: TypeInfo]: DependentEmitFunction[AsmFunction3[A1, A2, A3, R]] =
    genDependentFunction[AsmFunction3[A1, A2, A3, R]](null, Array(GenericTypeInfo[A1], GenericTypeInfo[A2], GenericTypeInfo[A3]), GenericTypeInfo[R])
}

class EmitClassBuilder[C](
  val emodb: EmitModuleBuilder,
  val cb: ClassBuilder[C]
) extends WrappedEmitModuleBuilder with WrappedClassBuilder[C] { self =>

  private[this] val rgMap: mutable.Map[ReferenceGenome, Value[ReferenceGenome]] =
    mutable.Map[ReferenceGenome, Value[ReferenceGenome]]()

  private[this] val typMap: mutable.Map[Type, Value[Type]] =
    mutable.Map[Type, Value[Type]]()

  private[this] val pTypeMap: mutable.Map[PType, Value[PType]] = mutable.Map[PType, Value[PType]]()

  private[this] type CompareMapKey = (PType, PType, CodeOrdering.Op, SortOrder, Boolean)
  private[this] val compareMap: mutable.Map[CompareMapKey, CodeOrdering.F[_]] =
    mutable.Map[CompareMapKey, CodeOrdering.F[_]]()

  def numReferenceGenomes: Int = rgMap.size

  def getReferenceGenome(rg: ReferenceGenome): Value[ReferenceGenome] =
    rgMap.getOrElseUpdate(rg, genLazyFieldThisRef[ReferenceGenome](rg.codeSetup(this)))

  def numTypes: Int = typMap.size

  private[this] def addReferenceGenome(rg: ReferenceGenome): Code[Unit] = {
    val rgExists = Code.invokeScalaObject[String, Boolean](ReferenceGenome.getClass, "hasReference", rg.name)
    val addRG = Code.invokeScalaObject[ReferenceGenome, Unit](ReferenceGenome.getClass, "addReference", getReferenceGenome(rg))
    rgExists.mux(Code._empty, addRG)
  }

  private[this] val literalsMap: mutable.Map[(PType, Any), Settable[_]] =
    mutable.Map[(PType, Any), Settable[_]]()
  private[this] lazy val encLitField: Settable[Array[Byte]] = genFieldThisRef[Array[Byte]]("encodedLiterals")

  lazy val partitionRegion: Settable[Region] = genFieldThisRef[Region]("partitionRegion")

  def addLiteral(v: Any, t: PType): Code[_] = {
    assert(v != null)
    assert(t.isCanonical)
    val f = literalsMap.getOrElseUpdate(t -> v, genFieldThisRef("literal")(typeToTypeInfo(t)))
    f.load()
  }

  private[this] def encodeLiterals(): Array[Byte] = {
    val literals = literalsMap.toArray
    val litType = PCanonicalTuple(true, literals.map(_._1._1): _*)
    val spec = TypedCodecSpec(litType, BufferSpec.defaultUncompressed)

    val (litRType, dec) = spec.buildEmitDecoderF[Long](litType.virtualType, this)
    assert(litRType == litType)
    cb.addInterface(typeInfo[FunctionWithLiterals].iname)
    val mb2 = newEmitMethod("addLiterals", Array(typeInfo[Array[Byte]]), typeInfo[Unit])
    val off = mb2.newLocal[Long]()
    val storeFields = literals.zipWithIndex.map { case (((_, _), f), i) =>
      f.storeAny(Region.loadIRIntermediate(litType.types(i))(litType.fieldOffset(off, i)))
    }

    mb2.emit(Code(
      encLitField := mb2.getArg[Array[Byte]](1),
      off := Code.memoize(spec.buildCodeInputBuffer(Code.newInstance[ByteArrayInputStream, Array[Byte]](encLitField)), "enc_lit_ib") { ib =>
        dec(partitionRegion, ib)
      },
      Code(storeFields)
    ))

    val baos = new ByteArrayOutputStream()
    val enc = spec.buildEncoder(litType)(baos)
    Region.scoped { region =>
      val rvb = new RegionValueBuilder(region)
      rvb.start(litType)
      rvb.startTuple()
      literals.foreach { case ((typ, a), _) => rvb.addAnnotation(typ.virtualType, a) }
      rvb.endTuple()
      enc.writeRegionValue(region, rvb.end())
    }
    enc.flush()
    enc.close()
    baos.toByteArray
  }

  private[this] var _hfs: FS = _
  private[this] var _hfield: Settable[FS] = _

  private[this] var _mods: ArrayBuilder[(String, (Int, Region) => AsmFunction3[Region, Array[Byte], Array[Byte], Array[Byte]])] = new ArrayBuilder()
  private[this] var _backendField: Settable[BackendUtils] = _

  private[this] var _aggSigs: Array[AggStatePhysicalSignature] = _
  private[this] var _aggRegion: Settable[Region] = _
  private[this] var _aggOff: Settable[Long] = _
  private[this] var _aggState: agg.TupleAggregatorState = _
  private[this] var _nSerialized: Int = 0
  private[this] var _aggSerialized: Settable[Array[Array[Byte]]] = _

  def addAggStates(aggSigs: Array[AggStatePhysicalSignature]): agg.TupleAggregatorState = {
    if (_aggSigs != null) {
      assert(aggSigs sameElements _aggSigs)
      return _aggState
    }
    cb.addInterface(typeInfo[FunctionWithAggRegion].iname)
    _aggSigs = aggSigs
    _aggRegion = genFieldThisRef[Region]("agg_top_region")
    _aggOff = genFieldThisRef[Long]("agg_off")
    val states = agg.StateTuple(aggSigs.map(a => agg.Extract.getAgg(a, a.default).createState(this)).toArray)
    _aggState = new agg.TupleAggregatorState(this, states, _aggRegion, _aggOff)
    _aggSerialized = genFieldThisRef[Array[Array[Byte]]]("agg_serialized")

    val newF = newEmitMethod("newAggState", FastIndexedSeq(typeInfo[Region]), typeInfo[Unit])
    val setF = newEmitMethod("setAggState", FastIndexedSeq(typeInfo[Region], typeInfo[Long]), typeInfo[Unit])
    val getF = newEmitMethod("getAggOffset", FastIndexedSeq(), typeInfo[Long])
    val setNSer = newEmitMethod("setNumSerialized", FastIndexedSeq(typeInfo[Int]), typeInfo[Unit])
    val setSer = newEmitMethod("setSerializedAgg", FastIndexedSeq(typeInfo[Int], typeInfo[Array[Byte]]), typeInfo[Unit])
    val getSer = newEmitMethod("getSerializedAgg", FastIndexedSeq(typeInfo[Int]), typeInfo[Array[Byte]])

    newF.emit(
      Code(_aggRegion := newF.getArg[Region](1),
        _aggState.topRegion.setNumParents(aggSigs.length),
        _aggOff := _aggRegion.load().allocate(states.storageType.alignment, states.storageType.byteSize),
        states.createStates(this),
        _aggState.newState))

    setF.emit(
      Code(
        _aggRegion := setF.getArg[Region](1),
        _aggState.topRegion.setNumParents(aggSigs.length),
        states.createStates(this),
        _aggOff := setF.getArg[Long](2),
        _aggState.load))

    getF.emit(Code(_aggState.store, _aggOff))

    setNSer.emit(_aggSerialized := Code.newArray[Array[Byte]](setNSer.getArg[Int](1)))

    setSer.emit(_aggSerialized.load().update(setSer.getArg[Int](1), setSer.getArg[Array[Byte]](2)))

    getSer.emit(_aggSerialized.load()(getSer.getArg[Int](1)))

    _aggState
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
      cb.addInterface(typeInfo[FunctionWithBackend].iname)
      val backendField = genFieldThisRef[BackendUtils]()
      val mb = newEmitMethod("setBackend", Array(typeInfo[BackendUtils]), typeInfo[Unit])
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
      cb.addInterface(typeInfo[FunctionWithFS].iname)
      val confField = genFieldThisRef[FS]()
      val mb = newEmitMethod("addFS", Array(typeInfo[FS]), typeInfo[Unit])
      mb.emit(confField := mb.getArg[FS](1))
      _hfs = HailContext.sFS
      _hfield = confField
    }

    assert(_hfs == HailContext.sFS && _hfield != null)
    _hfield.load()
  }

  def getPType(t: PType): Code[PType] = {
    val references = ReferenceGenome.getReferences(t.virtualType).toArray
    val setup = Code(Code(references.map(addReferenceGenome)),
      Code.invokeScalaObject[String, PType](
        IRParser.getClass, "parsePType", t.toString))
    pTypeMap.getOrElseUpdate(t,
      genLazyFieldThisRef[PType](setup))
  }

  def getType(t: Type): Code[Type] = {
    val references = ReferenceGenome.getReferences(t).toArray
    val setup = Code(Code(references.map(addReferenceGenome)),
      Code.invokeScalaObject[String, Type](
        IRParser.getClass, "parseType", t.parsableString()))
    typMap.getOrElseUpdate(t,
      genLazyFieldThisRef[Type](setup))
  }

  def getCodeOrdering(
    t1: PType,
    t2: PType,
    sortOrder: SortOrder,
    op: CodeOrdering.Op,
    ignoreMissingness: Boolean
  ): CodeOrdering.F[op.ReturnType] = {
    val f = compareMap.getOrElseUpdate((t1, t2, op, sortOrder, ignoreMissingness), {
      val ti = typeToTypeInfo(t1)
      val rt = if (op == CodeOrdering.compare) typeInfo[Int] else typeInfo[Boolean]

      val newMB = if (ignoreMissingness) {
        val newMB = genEmitMethod("cord", FastIndexedSeq[TypeInfo[_]](ti, ti), rt)
        val ord = t1.codeOrdering(newMB, t2, sortOrder)
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
        val newMB = genEmitMethod("cord", FastIndexedSeq[TypeInfo[_]](typeInfo[Boolean], ti, typeInfo[Boolean], ti), rt)
        val ord = t1.codeOrdering(newMB, t2, sortOrder)
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
    (v1: (Code[Boolean], Code[_]), v2: (Code[Boolean], Code[_])) => coerce[op.ReturnType](f(v1, v2))
  }

  def getCodeOrdering(
    t: PType,
    op: CodeOrdering.Op,
    sortOrder: SortOrder,
    ignoreMissingness: Boolean
  ): CodeOrdering.F[op.ReturnType] =
    getCodeOrdering(t, t, sortOrder, op, ignoreMissingness)

  def wrapVoids(x: Seq[Code[Unit]], prefix: String, size: Int = 32): Code[Unit] =
    wrapVoidsWithArgs(x.map { c => (s: Seq[Code[_]]) => c }, prefix, FastIndexedSeq(), FastIndexedSeq(), size)

  def wrapVoidsWithArgs(x: Seq[Seq[Code[_]] => Code[Unit]],
    suffix: String,
    argTypes: IndexedSeq[TypeInfo[_]],
    args: IndexedSeq[Code[_]],
    size: Int = 32): Code[Unit] = {
    val argTmps: IndexedSeq[Settable[Any]] = argTypes.zipWithIndex.map { case (ti, i) =>
      new LocalRef(new lir.Local(null, s"wvwa_arg$i", ti))(ti).asInstanceOf[Settable[Any]]
    }

    Code(
      Code((argTmps, args).zipped.map { case (t, arg) => t.storeAny(arg) }),
      Code(x.grouped(size).zipWithIndex.map { case (codes, i) =>
        val mb = genEmitMethod(suffix + s"_group$i", argTypes, UnitInfo)
        val methodArgs = argTypes.zipWithIndex.map { case (a, i) => mb.getArg(i + 1)(a) }
        mb.emit(Code(codes.map(_.apply(methodArgs.map(_.get)))))
        mb.invoke[Unit](argTmps.map(_.get): _*)
      }.toArray))
  }

  def newEmitMethod(name: String, argsInfo: IndexedSeq[TypeInfo[_]], returnInfo: TypeInfo[_]): EmitMethodBuilder[C] = {
    new EmitMethodBuilder[C](
      this,
      newMethod(name, argsInfo, returnInfo))
  }

  def newEmitMethod(name: String, argsInfo: IndexedSeq[MaybeGenericTypeInfo[_]], returnInfo: MaybeGenericTypeInfo[_]): EmitMethodBuilder[C] = {
    new EmitMethodBuilder[C](
      this,
      newMethod(name, argsInfo, returnInfo))
  }

  def genDependentFunction[F](baseName: String,
    maybeGenericParameterTypeInfo: IndexedSeq[MaybeGenericTypeInfo[_]],
    maybeGenericReturnTypeInfo: MaybeGenericTypeInfo[_])(implicit fti: TypeInfo[F]): DependentEmitFunction[F] = {
    val depCB = emodb.genEmitClass[F](baseName)
    new DependentEmitFunction[F](
      depCB,
      new EmitMethodBuilder[F](
        depCB,
        depCB.cb.newMethod("apply", maybeGenericParameterTypeInfo, maybeGenericReturnTypeInfo)))
  }

  val rngs: ArrayBuilder[(Settable[IRRandomness], Code[IRRandomness])] = new ArrayBuilder()

  def makeAddPartitionRegion(): Unit = {
    cb.addInterface(typeInfo[FunctionWithPartitionRegion].iname)
    val mb = newEmitMethod("addPartitionRegion", Array(typeInfo[Region]), typeInfo[Unit])
    mb.emit(partitionRegion := mb.getArg[Region](1))
  }

  def makeRNGs() {
    cb.addInterface(typeInfo[FunctionWithSeededRandomness].iname)

    val initialized = genFieldThisRef[Boolean]()
    val mb = newEmitMethod("setPartitionIndex", IndexedSeq(typeInfo[Int]), typeInfo[Unit])

    val rngFields = rngs.result()
    val initialize = Code(rngFields.map { case (field, initialization) =>
      field := initialization
    })

    val reseed = Code(rngFields.map { case (field, _) =>
      field.invoke[Int, Unit]("reset", mb.getArg[Int](1))
    })

    mb.emit(Code(
      initialized.mux(
        Code._empty,
        Code(initialize, initialized := true)),
      reseed))
  }

  def newRNG(seed: Long): Value[IRRandomness] = {
    val rng = genFieldThisRef[IRRandomness]()
    rngs += rng -> Code.newInstance[IRRandomness, Long](seed)
    rng
  }

  def resultWithIndex(print: Option[PrintWriter] = None): (Int, Region) => C = {
    makeRNGs()
    makeAddPartitionRegion()

    val hasLiterals: Boolean = literalsMap.nonEmpty

    val literalsBc = if (hasLiterals) {
      HailContext.get.backend.broadcast(encodeLiterals())
    } else {
      // if there are no literals, there might not be a HailContext
      null
    }

    val localFS = _hfs

    val nSerializedAggs = _nSerialized

    val useBackend = _backendField != null
    val backend = if (useBackend) new BackendUtils(_mods.result()) else null

    assert(TaskContext.get() == null,
      "FunctionBuilder emission should happen on master, but happened on worker")

    val n = name.replace("/", ".")
    val classesBytes = modb.classesBytes()

    new ((Int, Region) => C) with java.io.Serializable {
      @transient @volatile private var theClass: Class[_] = null

      def apply(idx: Int, region: Region): C = {
        if (theClass == null) {
          this.synchronized {
            if (theClass == null) {
              classesBytes.load()
              theClass = loadClass(n)
            }
          }
        }
        val f = theClass.newInstance().asInstanceOf[C]
        f.asInstanceOf[FunctionWithPartitionRegion].addPartitionRegion(region)
        if (localFS != null)
          f.asInstanceOf[FunctionWithFS].addFS(localFS)
        if (useBackend)
          f.asInstanceOf[FunctionWithBackend].setBackend(backend)
        if (hasLiterals)
          f.asInstanceOf[FunctionWithLiterals].addLiterals(literalsBc.value)
        if (nSerializedAggs != 0)
          f.asInstanceOf[FunctionWithAggRegion].setNumSerialized(nSerializedAggs)
        f.asInstanceOf[FunctionWithSeededRandomness].setPartitionIndex(idx)
        f
      }
    }
  }

  private[this] val methodMemo: mutable.Map[Any, EmitMethodBuilder[C]] = mutable.Map()

  def getOrGenEmitMethod(
    baseName: String, key: Any, argsInfo: IndexedSeq[TypeInfo[_]], returnInfo: TypeInfo[_]
  )(body: EmitMethodBuilder[C] => Unit): EmitMethodBuilder[C] = {
    methodMemo.getOrElseUpdate(key, {
      val mb = new EmitMethodBuilder(self, genMethod(baseName, argsInfo, returnInfo))
      body(mb)
      mb
    })
  }

  // derived functions
  def getCodeOrdering(t: PType, op: CodeOrdering.Op): CodeOrdering.F[op.ReturnType] =
    getCodeOrdering(t, t, sortOrder = Ascending, op, ignoreMissingness = false)

  def getCodeOrdering(t: PType, op: CodeOrdering.Op, ignoreMissingness: Boolean): CodeOrdering.F[op.ReturnType] =
    getCodeOrdering(t, t, sortOrder = Ascending, op, ignoreMissingness)

  def getCodeOrdering(t1: PType, t2: PType, op: CodeOrdering.Op): CodeOrdering.F[op.ReturnType] =
    getCodeOrdering(t1, t2, sortOrder = Ascending, op, ignoreMissingness = false)

  def getCodeOrdering(t1: PType, t2: PType, op: CodeOrdering.Op, ignoreMissingness: Boolean): CodeOrdering.F[op.ReturnType] =
    getCodeOrdering(t1, t2, sortOrder = Ascending, op, ignoreMissingness)

  def getCodeOrdering(
    t1: PType, t2: PType, sortOrder: SortOrder, op: CodeOrdering.Op
  ): CodeOrdering.F[op.ReturnType] =
    getCodeOrdering(t1, t2, sortOrder, op, ignoreMissingness = false)

  def genEmitMethod(baseName: String, argsInfo: IndexedSeq[TypeInfo[_]], returnInfo: TypeInfo[_]): EmitMethodBuilder[C] =
    newEmitMethod(genName("m", baseName), argsInfo, returnInfo)

  def genEmitMethod[R: TypeInfo](baseName: String): EmitMethodBuilder[C] =
    genEmitMethod(baseName, FastIndexedSeq[TypeInfo[_]](), typeInfo[R])

  def genEmitMethod[A: TypeInfo, R: TypeInfo](baseName: String): EmitMethodBuilder[C] =
    genEmitMethod(baseName, FastIndexedSeq[TypeInfo[_]](typeInfo[A]), typeInfo[R])

  def genEmitMethod[A: TypeInfo, B: TypeInfo, R: TypeInfo](baseName: String): EmitMethodBuilder[C] =
    genEmitMethod(baseName, FastIndexedSeq[TypeInfo[_]](typeInfo[A], typeInfo[B]), typeInfo[R])

  def genEmitMethod[A1: TypeInfo, A2: TypeInfo, A3: TypeInfo, R: TypeInfo](baseName: String): EmitMethodBuilder[C] =
    genEmitMethod(baseName, FastIndexedSeq[TypeInfo[_]](typeInfo[A1], typeInfo[A2], typeInfo[A3]), typeInfo[R])

  def genEmitMethod[A1: TypeInfo, A2: TypeInfo, A3: TypeInfo, A4: TypeInfo, R: TypeInfo](baseName: String): EmitMethodBuilder[C] =
    genEmitMethod(baseName, FastIndexedSeq[TypeInfo[_]](typeInfo[A1], typeInfo[A2], typeInfo[A3], typeInfo[A4]), typeInfo[R])

  def genEmitMethod[A1: TypeInfo, A2: TypeInfo, A3: TypeInfo, A4: TypeInfo, A5: TypeInfo, R: TypeInfo](baseName: String): EmitMethodBuilder[C] =
    genEmitMethod(baseName, FastIndexedSeq[TypeInfo[_]](typeInfo[A1], typeInfo[A2], typeInfo[A3], typeInfo[A4], typeInfo[A5]), typeInfo[R])

  def wrapInEmitMethod[R: TypeInfo](
    baseName: String,
    body: (EmitMethodBuilder[_]) => Code[R]): Code[R] = {
    val mb = genEmitMethod(baseName, FastIndexedSeq[TypeInfo[_]](), typeInfo[R])
    mb.emit(body(mb))
    mb.invoke[R]()
  }

  def wrapInEmitMethod[A: TypeInfo, R: TypeInfo](
    baseName: String,
    body: (EmitMethodBuilder[_], Code[A]) => Code[R]
  ): Code[A] => Code[R] = {
    val mb = genEmitMethod(baseName, FastIndexedSeq[TypeInfo[_]](typeInfo[A]), typeInfo[R])
    mb.emit(body(mb, mb.getArg[A](1)))
    a => mb.invoke[R](a)
  }

  def wrapInEmitMethod[A1: TypeInfo, A2: TypeInfo, R: TypeInfo](
    baseName: String,
    body: (EmitMethodBuilder[_], Code[A1], Code[A2]) => Code[R]
  ): (Code[A1], Code[A2]) => Code[R] = {
    val mb = genEmitMethod(baseName, FastIndexedSeq[TypeInfo[_]](typeInfo[A1], typeInfo[A2]), typeInfo[R])
    mb.emit(body(mb, mb.getArg[A1](1), mb.getArg[A2](2)))
    (a1, a2) => mb.invoke[R](a1, a2)
  }

  def getUnsafeReader(path: Code[String], checkCodec: Code[Boolean]): Code[InputStream] =
    getFS.invoke[String, Boolean, InputStream]("unsafeReader", path, checkCodec)

  def getUnsafeWriter(path: Code[String]): Code[OutputStream] =
    getFS.invoke[String, OutputStream]("unsafeWriter", path)

  def genDependentFunction[A1: TypeInfo, A2: TypeInfo, R: TypeInfo](
    baseName: String = null
  ): DependentEmitFunction[AsmFunction2[A1, A2, R]] =
    genDependentFunction[AsmFunction2[A1, A2, R]](baseName, Array(GenericTypeInfo[A1], GenericTypeInfo[A2]), GenericTypeInfo[R])

  def genDependentFunction[A1: TypeInfo, A2: TypeInfo, A3: TypeInfo, R: TypeInfo]: DependentEmitFunction[AsmFunction3[A1, A2, A3, R]] =
    genDependentFunction[AsmFunction3[A1, A2, A3, R]](null, Array(GenericTypeInfo[A1], GenericTypeInfo[A2], GenericTypeInfo[A3]), GenericTypeInfo[R])
}

object EmitFunctionBuilder {
  def apply[F](
    baseName: String, argInfo: IndexedSeq[MaybeGenericTypeInfo[_]], returnInfo: MaybeGenericTypeInfo[_]
  )(implicit fti: TypeInfo[F]): EmitFunctionBuilder[F] = {
    val modb = new EmitModuleBuilder(new ModuleBuilder())
    val cb = modb.genEmitClass[F](baseName)
        val apply = cb.newEmitMethod("apply", argInfo, returnInfo)
    new EmitFunctionBuilder(apply)
  }

  def apply[R: TypeInfo](baseName: String): EmitFunctionBuilder[AsmFunction0[R]] =
    EmitFunctionBuilder[AsmFunction0[R]](baseName, FastIndexedSeq[MaybeGenericTypeInfo[_]](), GenericTypeInfo[R])

  def apply[A: TypeInfo, R: TypeInfo](baseName: String): EmitFunctionBuilder[AsmFunction1[A, R]] =
    EmitFunctionBuilder[AsmFunction1[A, R]](baseName, Array(GenericTypeInfo[A]), GenericTypeInfo[R])

  def apply[A: TypeInfo, B: TypeInfo, R: TypeInfo](baseName: String): EmitFunctionBuilder[AsmFunction2[A, B, R]] =
    EmitFunctionBuilder[AsmFunction2[A, B, R]](baseName, Array(GenericTypeInfo[A], GenericTypeInfo[B]), GenericTypeInfo[R])

  def apply[A: TypeInfo, B: TypeInfo, C: TypeInfo, R: TypeInfo](baseName: String): EmitFunctionBuilder[AsmFunction3[A, B, C, R]] =
    EmitFunctionBuilder[AsmFunction3[A, B, C, R]](baseName, Array(GenericTypeInfo[A], GenericTypeInfo[B], GenericTypeInfo[C]), GenericTypeInfo[R])

  def apply[A: TypeInfo, B: TypeInfo, C: TypeInfo, D: TypeInfo, R: TypeInfo](baseName: String): EmitFunctionBuilder[AsmFunction4[A, B, C, D, R]] =
    EmitFunctionBuilder[AsmFunction4[A, B, C, D, R]](baseName, Array(GenericTypeInfo[A], GenericTypeInfo[B], GenericTypeInfo[C], GenericTypeInfo[D]), GenericTypeInfo[R])

  def apply[A: TypeInfo, B: TypeInfo, C: TypeInfo, D: TypeInfo, E: TypeInfo, R: TypeInfo](baseName: String): EmitFunctionBuilder[AsmFunction5[A, B, C, D, E, R]] =
    EmitFunctionBuilder[AsmFunction5[A, B, C, D, E, R]](baseName, Array(GenericTypeInfo[A], GenericTypeInfo[B], GenericTypeInfo[C], GenericTypeInfo[D], GenericTypeInfo[E]), GenericTypeInfo[R])

  def apply[A: TypeInfo, B: TypeInfo, C: TypeInfo, D: TypeInfo, E: TypeInfo, F: TypeInfo, R: TypeInfo](baseName: String): EmitFunctionBuilder[AsmFunction6[A, B, C, D, E, F, R]] =
    EmitFunctionBuilder[AsmFunction6[A, B, C, D, E, F, R]](baseName, Array(GenericTypeInfo[A], GenericTypeInfo[B], GenericTypeInfo[C], GenericTypeInfo[D], GenericTypeInfo[E], GenericTypeInfo[F]), GenericTypeInfo[R])

  def apply[A: TypeInfo, B: TypeInfo, C: TypeInfo, D: TypeInfo, E: TypeInfo, F: TypeInfo, G: TypeInfo, R: TypeInfo](baseName: String): EmitFunctionBuilder[AsmFunction7[A, B, C, D, E, F, G, R]] =
    EmitFunctionBuilder[AsmFunction7[A, B, C, D, E, F, G, R]](baseName, Array(GenericTypeInfo[A], GenericTypeInfo[B], GenericTypeInfo[C], GenericTypeInfo[D], GenericTypeInfo[E], GenericTypeInfo[F], GenericTypeInfo[G]), GenericTypeInfo[R])
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

trait FunctionWithPartitionRegion {
  def addPartitionRegion(r: Region): Unit
}

trait FunctionWithLiterals {
  def addLiterals(lit: Array[Byte]): Unit
}

trait FunctionWithSeededRandomness {
  def setPartitionIndex(idx: Int): Unit
}

trait FunctionWithBackend {
  def setBackend(spark: BackendUtils): Unit
}

class EmitMethodBuilder[C](
  val ecb: EmitClassBuilder[C],
  val mb: MethodBuilder[C]
) extends WrappedEmitClassBuilder[C] with WrappedMethodBuilder[C] {
  def getArgEV(i: Int, _pt: PType): EmitValue = new EmitValue {
    def pt: PType = _pt

    def get: EmitCode = {
      EmitCode(Code._empty,
        getArg[Boolean](2 + 2 * i + 1),
        PCode(pt, getArg(2 + 2 * i)(typeToTypeInfo(_pt)).load()))
    }
  }

  def newPSettable(_pt: PType, s: Settable[_]): PSettable = new PSettable {
    def pt: PType = _pt

    def get: PCode = PCode(_pt, s)

    def store(v: PCode): Code[Unit] = s.storeAny(v.code)
  }

  def newPLocal(pt: PType): PSettable = newPSettable(pt, newLocal()(typeToTypeInfo(pt)))

  def newPLocal(name: String, pt: PType): PSettable = newPSettable(pt, newLocal(name)(typeToTypeInfo(pt)))

  def newPField(pt: PType): PSettable = newPSettable(pt, genFieldThisRef()(typeToTypeInfo(pt)))

  def newPField(name: String, pt: PType): PSettable = newPSettable(pt, genFieldThisRef(name)(typeToTypeInfo(pt)))

  def newEmitSettable(_pt: PType, ms: Settable[Boolean], vs: PSettable): EmitSettable = new EmitSettable {
    def pt: PType = _pt

    def get: EmitCode = EmitCode(Code._empty,
      if (_pt.required) false else ms.get,
      vs.get)

    def store(ec: EmitCode): Code[Unit] =
      if (_pt.required)
        Code(ec.setup,
          // FIXME put this under control of a debugging option
          ec.m.mux(
            Code._fatal[Unit](s"Required EmitSettable cannot be missing ${ _pt }"),
            Code._empty),
          vs := ec.pv)
      else
        Code(ec.setup, ec.m.mux(ms := true, Code(ms := false, vs := ec.pv)))
  }

  def newEmitLocal(pt: PType): EmitSettable =
    newEmitSettable(pt, if (pt.required) null else newLocal[Boolean](), newPLocal(pt))

  def newEmitLocal(name: String, pt: PType): EmitSettable =
    newEmitSettable(pt, if (pt.required) null else newLocal[Boolean](name + "_missing"), newPLocal(name, pt))

  def newEmitField(pt: PType): EmitSettable =
    newEmitSettable(pt, genFieldThisRef[Boolean](), newPField(pt))

  def newEmitField(name: String, pt: PType): EmitSettable =
    newEmitSettable(pt, genFieldThisRef[Boolean](name + "_missing"), newPField(name, pt))

  def newPresentEmitSettable(_pt: PType, ps: PSettable): PresentEmitSettable = new PresentEmitSettable {
    def pt: PType = _pt

    def get: EmitCode = EmitCode(Code._empty, const(false), ps.load())

    def store(pv: PCode): Code[Unit] = ps := pv
  }

  def newPresentEmitLocal(pt: PType): PresentEmitSettable =
    newPresentEmitSettable(pt, newPLocal(pt))

  def newPresentEmitField(pt: PType): PresentEmitSettable =
    newPresentEmitSettable(pt, newPField(pt))

  def newPresentEmitField(name: String, pt: PType): PresentEmitSettable =
    newPresentEmitSettable(pt, newPField(name, pt))
}

trait WrappedEmitMethodBuilder[C] extends WrappedEmitClassBuilder[C] with WrappedMethodBuilder[C] {
  def emb: EmitMethodBuilder[C]

  def ecb: EmitClassBuilder[C] = emb.ecb

  def mb: MethodBuilder[C] = emb.mb

  def getArgEV(i: Int, _pt: PType): EmitValue = emb.getArgEV(i, _pt)

  def newPLocal(pt: PType): PSettable = emb.newPLocal(pt)

  def newPLocal(name: String, pt: PType): PSettable = emb.newPLocal(name, pt)

  def newPField(pt: PType): PSettable = emb.newPField(pt)

  def newPField(name: String, pt: PType): PSettable = emb.newPField(name, pt)

  def newEmitLocal(pt: PType): EmitSettable = emb.newEmitLocal(pt)

  def newEmitLocal(name: String, pt: PType): EmitSettable = emb.newEmitLocal(name, pt)

  def newEmitField(pt: PType): EmitSettable = emb.newEmitField(pt)

  def newEmitField(name: String, pt: PType): EmitSettable = emb.newEmitField(name, pt)

  def newPresentEmitLocal(pt: PType): PresentEmitSettable = emb.newPresentEmitLocal(pt)

  def newPresentEmitField(pt: PType): PresentEmitSettable = emb.newPresentEmitField(pt)

  def newPresentEmitField(name: String, pt: PType): PresentEmitSettable = emb.newPresentEmitField(name, pt)
}

class DependentEmitFunction[F](
  parentcb: EmitClassBuilder[_],
  val apply_method: EmitMethodBuilder[F]
) extends WrappedEmitMethodBuilder[F] with DependentFunction[F] {
  def emb: EmitMethodBuilder[F] = apply_method

  private[this] val rgMap: mutable.Map[ReferenceGenome, Value[ReferenceGenome]] =
    mutable.Map[ReferenceGenome, Value[ReferenceGenome]]()

  private[this] val typMap: mutable.Map[Type, Value[Type]] =
    mutable.Map[Type, Value[Type]]()

  private[this] val literalsMap: mutable.Map[(PType, Any), Value[_]] =
    mutable.Map[(PType, Any), Value[_]]()

  override def getReferenceGenome(rg: ReferenceGenome): Value[ReferenceGenome] =
    rgMap.getOrElseUpdate(rg, {
      val fromParent = parentcb.getReferenceGenome(rg)
      val field = newDepField[ReferenceGenome](fromParent)
      field
    })

  override def getType(t: Type): Code[Type] =
    typMap.getOrElseUpdate(t, {
      val fromParent = parentcb.getType(t)
      val field = newDepField[Type](fromParent)
      field
    })

  override def addLiteral(v: Any, t: PType): Code[_] = {
    assert(v != null)
    literalsMap.getOrElseUpdate(t -> v, {
      val fromParent = parentcb.addLiteral(v, t)
      val ti: TypeInfo[_] = typeToTypeInfo(t)
      val field = newDepFieldAny(fromParent)(ti)
      field
    })
  }

  def newDepEmitField(ec: EmitCode): EmitValue = {
    val _pt = ec.pt
    val ti = typeToTypeInfo(_pt)
    val m = genFieldThisRef[Boolean]()
    val v = genFieldThisRef()(ti)
    setFields += { (obj: lir.ValueX) =>
      val setup = ec.setup
      setup.end.append(lir.putField(name, m.name, typeInfo[Boolean], obj, ec.m.v))
      setup.end.append(lir.putField(name, v.name, ti, obj, ec.v.v))
      val newC = new VCode(setup.start, setup.end, null)
      setup.clear()
      ec.m.clear()
      ec.v.clear()
      newC
    }
    new EmitValue {
      def pt: PType = _pt

      def get: EmitCode = EmitCode(Code._empty, m.load(), PCode(_pt, v.load()))
    }
  }
}

class EmitFunctionBuilder[F](val apply_method: EmitMethodBuilder[F]) extends WrappedEmitMethodBuilder[F] {
  def emb: EmitMethodBuilder[F] = apply_method
}
