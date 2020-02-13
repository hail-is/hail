package is.hail.expr.ir

import java.io.{ByteArrayInputStream, ByteArrayOutputStream}

import is.hail.HailContext
import is.hail.annotations._
import is.hail.asm4s.{Code, _}
import is.hail.expr.ir.functions.{MathFunctions, StringFunctions}
import is.hail.expr.types.physical._
import is.hail.expr.types.virtual._
import is.hail.io.{BufferSpec, InputBuffer, OutputBuffer, TypedCodecSpec}
import is.hail.linalg.{BLAS, LAPACK, LinalgCodeUtils}
import is.hail.utils._

import scala.collection.mutable
import scala.language.{existentials, postfixOps}

object Emit {
  type E = Env[(TypeInfo[_], Code[Boolean], Code[_])]

  type F = (Code[Boolean], Code[_]) => Code[Unit]

  def apply(ctx: ExecuteContext, ir: IR, fb: EmitFunctionBuilder[_], aggs: Option[Array[AggStatePhysicalSignature]] = None) {
    val triplet = emit(ctx, ir, fb, Env.empty, aggs)
    typeToTypeInfo(ir.typ) match {
      case ti: TypeInfo[t] =>
        fb.emit(Code(triplet.setup, triplet.m.mux(
          Code._throw(Code.newInstance[RuntimeException, String]("cannot return empty")),
          coerce[t](triplet.v))))
    }
  }

  private def emit(
    ctx: ExecuteContext,
    ir: IR,
    fb: EmitFunctionBuilder[_],
    env: E,
    aggs: Option[Array[AggStatePhysicalSignature]]): EmitTriplet = {
    TypeCheck(ir)
    val container = aggs.map { a =>
      val c = fb.addAggStates(a)
      AggContainer(a, c)
    }

    val baseTriplet = new Emit(ctx: ExecuteContext, fb.apply_method).emit(ir, env, EmitRegion.default(fb.apply_method), container = container)

    EmitTriplet(
      baseTriplet.setup,
      baseTriplet.m,
      baseTriplet.v)
  }
}

object AggContainer {
  def fromVars(aggs: Array[AggStatePhysicalSignature], fb: EmitFunctionBuilder[_], region: ClassFieldRef[Region], off: ClassFieldRef[Long]): (AggContainer, Code[Unit], Code[Unit]) = {
    val states = agg.StateTuple(aggs.map(a => agg.Extract.getAgg(a, a.default).createState(fb)).toArray)
    val aggState = new agg.TupleAggregatorState(fb, states, region, off)

    val setup = Code(
      region := Region.stagedCreate(Region.REGULAR),
      region.load().setNumParents(aggs.length),
      off := region.load().allocate(aggState.storageType.alignment, aggState.storageType.byteSize),
      states.createStates(fb),
      aggState.newState)

    val cleanup = Code(
      region.load().invalidate(),
      region := Code._null)

    (AggContainer(aggs, aggState), setup, cleanup)
  }
  def fromFunctionBuilder(aggs: Array[AggStatePhysicalSignature], fb: EmitFunctionBuilder[_], varPrefix: String): (AggContainer, Code[Unit], Code[Unit]) =
    fromVars(aggs, fb, fb.newField[Region](s"${varPrefix}_top_region"), fb.newField[Long](s"${varPrefix}_off"))
}

case class AggContainer(aggs: Array[AggStatePhysicalSignature], container: agg.TupleAggregatorState) {

  def nested(i: Int, init: Boolean): Option[AggContainer] = {
    aggs(i).nested.map { n =>
      aggs(i).default match {
        case AggElementsLengthCheck() =>
          val state = container.states(i).asInstanceOf[agg.ArrayElementState]
          if (init)
            AggContainer(n.toArray, state.initContainer)
          else
            AggContainer(n.toArray, state.container)
        case Group() =>
          val state = container.states(i).asInstanceOf[agg.DictState]
          if (init)
            AggContainer(n.toArray, state.initContainer)
          else
            AggContainer(n.toArray, state.keyed.container)
      }
    }
  }
}

object EmitRegion {
  def default(mb: EmitMethodBuilder): EmitRegion = EmitRegion(mb, mb.getArg[Region](1))
}

case class EmitRegion(mb: EmitMethodBuilder, region: Code[Region]) {
  def baseRegion: Code[Region] = mb.getArg[Region](1)
}

case class EmitTriplet(setup: Code[Unit], m: Code[Boolean], v: Code[_]) {
  def value[T]: Code[T] = coerce[T](v)
}

case class EmitArrayTriplet(setup: Code[Unit], m: Option[Code[Boolean]], addElements: Code[Unit])

case class ArrayIteratorTriplet(calcLength: Code[Unit], length: Option[Code[Int]], arrayEmitter: Emit.F => EmitArrayTriplet) {
  def arrayEmitterFromBuilder(sab: StagedArrayBuilder): EmitArrayTriplet = {
    arrayEmitter( { (m: Code[Boolean], v: Code[_]) => m.mux(sab.addMissing(), sab.add(v)) } )
  }

  def toEmitTriplet(mb: MethodBuilder, aTyp: PArray): EmitTriplet = {
    val srvb = new StagedRegionValueBuilder(mb, aTyp)

    length match {
      case Some(len) =>
        val cont = { (m: Code[Boolean], v: Code[_]) =>
          coerce[Unit](
            Code(
              m.mux(
                srvb.setMissing(),
                srvb.addIRIntermediate(aTyp.elementType)(v)),
              srvb.advance()))
        }
        val processAElts = arrayEmitter(cont)
        EmitTriplet(processAElts.setup, processAElts.m.getOrElse(const(false)), Code(
          calcLength,
          srvb.start(len, init = true),
          processAElts.addElements,
          srvb.offset
        ))

      case None =>
        val len = mb.newLocal[Int]
        val i = mb.newLocal[Int]
        val vab = new StagedArrayBuilder(aTyp.elementType, mb, 16)
        val processArrayElts = arrayEmitter { (m: Code[Boolean], v: Code[_]) => m.mux(vab.addMissing(), vab.add(v)) }
        EmitTriplet(Code(vab.clear, processArrayElts.setup), processArrayElts.m.getOrElse(const(false)), Code(
          calcLength,
          processArrayElts.addElements,
          len := vab.size,
          srvb.start(len, init = true),
          i := 0,
          Code.whileLoop(i < len,
            vab.isMissing(i).mux(
              srvb.setMissing(),
              srvb.addIRIntermediate(aTyp.elementType)(vab(i))),
            i := i + 1,
            srvb.advance()),
          srvb.offset))
    }
  }
}

case class LoopRef(m: ClassFieldRef[Boolean], v: ClassFieldRef[_], tempM: LocalRef[Boolean], tempV: LocalRef[_])

abstract class MethodBuilderLike[M <: MethodBuilderLike[M]] {
  type MB <: MethodBuilder

  def mb: MB

  def newMethod(paramInfo: Array[TypeInfo[_]], returnInfo: TypeInfo[_]): M
}

class MethodBuilderSelfLike(val mb: MethodBuilder) extends MethodBuilderLike[MethodBuilderSelfLike] {
  type MB = MethodBuilder

  def newMethod(paramInfo: Array[TypeInfo[_]], returnInfo: TypeInfo[_]): MethodBuilderSelfLike =
    new MethodBuilderSelfLike(mb.fb.newMethod(paramInfo, returnInfo))
}

abstract class EstimableEmitter[M <: MethodBuilderLike[M]] {
  def emit(mb: M): Code[Unit]

  def estimatedSize: Int
}

object EmitUtils {
  private val maxBytecodeSizeTarget: Int = 4096

  def getChunkBounds(sizes: Seq[Int]): Array[Int] = {
    val ab = new ArrayBuilder[Int]()
    ab += 0
    var total = sizes.head
    sizes.zipWithIndex.tail.foreach { case (size, i) =>
      if (total + size <= maxBytecodeSizeTarget)
        total += size
      else {
        ab += i
        total = size
      }
    }
    ab += sizes.size
    ab.result()
  }

  def wrapToMethod[T, M <: MethodBuilderLike[M]](items: Seq[EstimableEmitter[M]], mbLike: M): Code[Unit] = {
    if (items.isEmpty)
      return Code._empty

    val sizes = items.map(_.estimatedSize)
    if (sizes.sum < 100)
      return coerce[Unit](Code(items.map(_.emit(mbLike)): _*))

    val chunkBounds = getChunkBounds(sizes)
    assert(chunkBounds(0) == 0 && chunkBounds.last == sizes.length)

    val chunks = chunkBounds.zip(chunkBounds.tail).map { case (start, end) =>
      assert(start < end)
      val newMBLike = mbLike.newMethod(mbLike.mb.parameterTypeInfo, typeInfo[Unit])
      val c = items.slice(start, end)
      newMBLike.mb.emit(Code(c.map(_.emit(newMBLike)): _*))
      new EstimableEmitter[M] {
        def emit(mbLike: M): Code[Unit] = {
          val args = mbLike.mb.parameterTypeInfo.zipWithIndex.map { case (ti, i) => mbLike.mb.getArg(i + 1)(ti).load() }
          coerce[Unit](newMBLike.mb.invoke(args: _*))
        }

        def estimatedSize: Int = 5
      }
    }
    wrapToMethod(chunks, mbLike)
  }
}

private class Emit(
  val ctx: ExecuteContext,
  val mb: EmitMethodBuilder) {

  val resultRegion: EmitRegion = EmitRegion.default(mb)
  val region: Code[Region] = mb.getArg[Region](1)
  val methods: mutable.Map[String, Seq[(Seq[PType], PType, EmitMethodBuilder)]] = mutable.Map().withDefaultValue(FastSeq())

  import Emit.{E, F}

  class EmitMethodBuilderLike(val emit: Emit) extends MethodBuilderLike[EmitMethodBuilderLike] {
    type MB = EmitMethodBuilder

    def mb: MB = emit.mb

    def newMethod(paramInfo: Array[TypeInfo[_]], returnInfo: TypeInfo[_]): EmitMethodBuilderLike = {
      val newMB = emit.mb.fb.newMethod(paramInfo, returnInfo)
      val newEmitter = new Emit(ctx, newMB)
      new EmitMethodBuilderLike(newEmitter)
    }
  }

  private def wrapToMethod(irs: Seq[IR], env: E, container: Option[AggContainer])(useValues: (EmitMethodBuilder, PType, EmitTriplet) => Code[Unit]): Code[Unit] = {
    val opSize: Int = 20
    val items = irs.map { ir =>
      new EstimableEmitter[EmitMethodBuilderLike] {
        def estimatedSize: Int = ir.size * opSize

        def emit(mbLike: EmitMethodBuilderLike): Code[Unit] =
          // wrapped methods can't contain uses of Recur
          useValues(mbLike.mb, ir.pType, mbLike.emit.emit(ir, env, EmitRegion.default(mbLike.emit.mb), container, None))
      }
    }

    EmitUtils.wrapToMethod(items, new EmitMethodBuilderLike(this))
  }

  /**
    * Invariants of the Returned Triplet
    * ----------------------------------
    *
    * The elements of the triplet are called (precompute, missingness, value)
    *
    *  1. evaluate each returned Code[_] at most once
    *  2. evaluate precompute *on all static code-paths* leading to missingness or value
    *  3. guard the the evaluation of value by missingness
    *
    * Triplets returning values cannot have side-effects.  For void triplets, precompute
    * contains the side effect, missingness is false, and value is {@code Code._empty}.
    *
    * JVM gotcha:
    * a variable must be initialized on all static code-paths prior to its use (ergo defaultValue)
    *
    * Argument Convention
    * -------------------
    *
    * {@code In(i)} occupies two argument slots, one for the value and one for a
    * missing bit. The value for {@code  In(0)} is passed as argument
    * {@code  nSpecialArguments + 1}. The missingness bit is the subsequent
    * argument. In general, the value for {@code  In(i)} appears at
    * {@code  nSpecialArguments + 1 + 2 * i}.
    *
    * There must always be at least one special argument: a {@code  Region} in
    * which the IR can allocate memory.
    *
    * When compiling an aggregation expression, {@code AggIn} refers to the first
    * argument {@code In(0)} whose type must be of type
    * {@code tAggIn.elementType}.  {@code tAggIn.symTab} is not used by Emit.
    *
    **/
  private[ir] def emit(ir: IR, env: E, er: EmitRegion, container: Option[AggContainer]): EmitTriplet =
    emit(ir, env, er, container, None)

  private def emit(ir: IR, env: E, er: EmitRegion, container: Option[AggContainer], loopEnv: Option[Env[Array[LoopRef]]]): EmitTriplet = {

    def emit(ir: IR, env: E = env, er: EmitRegion = er, container: Option[AggContainer] = container, loopEnv: Option[Env[Array[LoopRef]]] = loopEnv): EmitTriplet =
      this.emit(ir, env, er, container, loopEnv)

    def wrapToMethod(irs: Seq[IR], env: E = env, container: Option[AggContainer] = container)(useValues: (EmitMethodBuilder, PType, EmitTriplet) => Code[Unit]): Code[Unit] =
      this.wrapToMethod(irs, env, container)(useValues)

    def emitArrayIterator(ir: IR, env: E = env, container: Option[AggContainer] = container) = this.emitArrayIterator(ir, env, er, container)

    def emitDeforestedNDArray(ir: IR) =
      deforestNDArray(resultRegion, ir, env).emit(coerce[PNDArray](ir.pType))

    def emitNDArrayStandardStrides(ir: IR) =
      // Currently relying on the fact that emitDeforestedNDArray always emits standard striding.
      emitDeforestedNDArray(ir)

    val region = er.region

    (ir: @unchecked) match {
      case I32(x) =>
        present(const(x))
      case I64(x) =>
        present(const(x))
      case F32(x) =>
        present(const(x))
      case F64(x) =>
        present(const(x))
      case s@Str(x) =>
        present(mb.fb.addLiteral(x, coerce[PString](s.pType), er.baseRegion))
      case x@Literal(t, v) =>
        if (v == null)
          emit(NA(t))
        else
          present(mb.fb.addLiteral(v, x.pType, er.baseRegion))
      case True() =>
        present(const(true))
      case False() =>
        present(const(false))
      case Void() =>
        EmitTriplet(Code._empty, const(false), Code._empty)

      case Cast(v, typ) =>
        val codeV = emit(v)
        val cast = Casts.get(v.typ, typ)
        EmitTriplet(codeV.setup, codeV.m, cast(codeV.v))

      case CastRename(v, _typ) =>
        emit(v)

      case NA(typ) =>
        EmitTriplet(Code._empty, const(true), defaultValue(typ))
      case IsNA(v) =>
        val codeV = emit(v)
        EmitTriplet(codeV.setup, const(false), codeV.m)

      case Coalesce(values) =>
        val va = values.toArray.map(emit(_))
        val mbs = Array.fill(va.length - 1)(mb.newLocal[Boolean])

        val outType = values.head.typ
        val mout = mb.newLocal[Boolean]()
        val out = coerce[Any](mb.newLocal()(typeToTypeInfo(outType)))

        val setup = va.indices
          .init
          .foldRight(Code(
            mout := va.last.m,
            out := defaultValue(outType),
            mout.mux(Code._empty, out := ir.pType.copyFromTypeAndStackValue(mb, er.region, values.last.pType, va.last.v)))) { case (i, comb) =>
            Code(
              mbs(i) := va(i).m,
              mbs(i).mux(
                comb,
                Code(
                  mout := false,
                  out := ir.pType.copyFromTypeAndStackValue(mb, er.region, values(i).pType, va(i).v))))
          }

        EmitTriplet(
          setup = Code(
            Code(va.map(_.setup): _*),
            setup),
          m = mout,
          v = out
        )

      case If(cond, cnsq, altr) =>
        assert(cnsq.typ isOfType altr.typ)

        if (cnsq.typ == TVoid) {
          val codeCond = emit(cond)
          val codeCnsq = emit(cnsq)
          val codeAltr = emit(altr)
          EmitTriplet(
            Code(
              codeCond.setup,
              codeCond.m.mux(
                Code._empty,
                codeCond.value[Boolean].mux(
                  codeCnsq.setup,
                  codeAltr.setup))),
            const(false),
            Code._empty)
        } else {
          val typ = ir.typ
          val codeCond = emit(cond)
          val out = coerce[Any](mb.newLocal()(typeToTypeInfo(typ)))
          val mout = mb.newLocal[Boolean]()
          val codeCnsq = emit(cnsq)
          val codeAltr = emit(altr)

          val setup = Code(
            codeCond.setup,
            codeCond.m.mux(
              Code(mout := true, out := defaultValue(typ)),
              coerce[Boolean](codeCond.v).mux(
                Code(codeCnsq.setup, mout := codeCnsq.m, out := mout.mux(defaultValue(typ), ir.pType.copyFromTypeAndStackValue(mb, er.region, cnsq.pType, codeCnsq.v))),
                Code(codeAltr.setup, mout := codeAltr.m, out := mout.mux(defaultValue(typ), ir.pType.copyFromTypeAndStackValue(mb, er.region, altr.pType, codeAltr.v))))))

          EmitTriplet(setup, mout, out)
        }

      case Let(name, value, body) =>
        val vti = typeToTypeInfo(value.typ)
        val mx = mb.newField[Boolean]()
        val x = coerce[Any](mb.newField(name)(vti))
        val storeV = wrapToMethod(FastIndexedSeq(value)) { (_, _, codeV) =>
          Code(codeV.setup,
            mx := codeV.m,
            x := mx.mux(defaultValue(value.typ), codeV.v))
        }
        val bodyenv = env.bind(name, (vti, mx.load(), x.load()))
        val codeBody = emit(body, env = bodyenv)
        val setup = Code(
          storeV,
          codeBody.setup)

        EmitTriplet(setup, codeBody.m, codeBody.v)
      case Ref(name, typ) =>
        val ti = typeToTypeInfo(typ)
        val (t, m, v) = env.lookup(name)
        assert(t == ti, s"$name type annotation, $typ, $t doesn't match typeinfo: $ti")
        EmitTriplet(Code._empty, m, v)

      case ApplyBinaryPrimOp(op, l, r) =>
        val typ = ir.typ
        val codeL = emit(l)
        val codeR = emit(r)
        strict(BinaryOp.emit(op, l.typ, r.typ, codeL.v, codeR.v), codeL, codeR)
      case ApplyUnaryPrimOp(op, x) =>
        val typ = ir.typ
        val v = emit(x)
        strict(UnaryOp.emit(op, x.typ, v.v), v)
      case ApplyComparisonOp(op, l, r) =>
        val f = op.codeOrdering(mb, l.pType, r.pType)
        val codeL = emit(l)
        val codeR = emit(r)
        if (op.strict) {
          strict(f((false, codeL.v), (false, codeR.v)),
            codeL, codeR)
        } else {
          val lm = mb.newLocal[Boolean]
          val rm = mb.newLocal[Boolean]
          present(Code(
            codeL.setup,
            codeR.setup,
            lm := codeL.m,
            rm := codeR.m,
            f((lm, lm.mux(defaultValue(l.typ), codeL.v)),
              (rm, rm.mux(defaultValue(r.typ), codeR.v)))))
        }

      case x@MakeArray(args, _) =>
        val pType = x.pType.asInstanceOf[PArray]
        val srvb = new StagedRegionValueBuilder(mb, pType)
        val addElement = srvb.addIRIntermediate(pType.elementType)

        val addElts = { (newMB: EmitMethodBuilder, pt: PType, v: EmitTriplet) =>
          Code(
            v.setup,
            v.m.mux(srvb.setMissing(), addElement(pType.elementType.copyFromTypeAndStackValue(newMB, er.region, pt, v.v))),
            srvb.advance())
        }
        present(Code(srvb.start(args.size, init = true), wrapToMethod(args)(addElts), srvb.offset))
      case x@ArrayRef(a, i, s) =>
        val typ = x.typ
        val pArray = coerce[PStreamable](a.pType).asPArray
        val ati = coerce[Long](typeToTypeInfo(pArray))
        val codeA = emit(a)
        val codeI = emit(i)
        val errorTransformer: Code[String] => Code[String] = s match {
          case Str("") =>
            val prettied = Pretty.short(x)
            (c: Code[String]) =>
              c.concat("\n----------\nIR:\n").concat(prettied)
          case Str(s) => (c: Code[String]) => c.concat("\n----------\nPython traceback:\n").concat(const(s))
          case s =>
            val codeS = emit(s)
            (c: Code[String]) =>
              Code(codeS.setup,
                codeS.m.mux(c, c
                  .concat("\n----------\nPython traceback:\n")
                  .concat(s.pType.asInstanceOf[PString].loadString(coerce[Long](codeS.v)))))
        }
        val xma = mb.newLocal[Boolean]()
        val xa = mb.newLocal()(ati)
        val xi = mb.newLocal[Int]
        val len = mb.newLocal[Int]
        val xmi = mb.newLocal[Boolean]()
        val xmv = mb.newLocal[Boolean]()
        val setup = Code(
          codeA.setup,
          xma := codeA.m,
          xa := coerce[Long](defaultValue(pArray)),
          codeI.setup,
          xmi := codeI.m,
          xi := coerce[Int](defaultValue(TInt32())),
          len := coerce[Int](defaultValue(TInt32())),
          (xmi || xma).mux(
            xmv := const(true),
            Code(
              xa := coerce[Long](codeA.v),
              xi := coerce[Int](codeI.v),
              len := pArray.loadLength(xa),
              (xi < len && xi >= 0).mux(
                xmv := !pArray.isElementDefined(xa, xi),
                Code._fatal(errorTransformer(
                  const("array index out of bounds: index=")
                    .concat(xi.load().toS)
                    .concat(", length=")
                    .concat(len.load().toS)))))))

        EmitTriplet(setup, xmv, Code(
          Region.loadIRIntermediate(x.pType)(pArray.elementOffset(xa, len, xi))))
      case ArrayLen(a) =>
        val codeA = emit(a)
        strict(a.pType.asInstanceOf[PArray].loadLength(coerce[Long](codeA.v)), codeA)

      case x@(_: ArraySort | _: ToSet | _: ToDict) =>
        val atyp = coerce[PIterable](x.pType)
        val eltType = -atyp.elementType
        val eltVType = eltType.virtualType
        val vab = new StagedArrayBuilder(atyp.elementType, mb, 16)
        val sorter = new ArraySorter(er, vab)

        val (array, compare, distinct) = (x: @unchecked) match {
          case ArraySort(a, l, r, comp) => (a, Subst(comp, BindingEnv(Env[IR](l -> In(0, eltType), r -> In(1, eltType)))), Code._empty[Unit])
          case ToSet(a) =>
            val discardNext = mb.fb.newMethod(Array[TypeInfo[_]](typeInfo[Region], sorter.ti, typeInfo[Boolean], sorter.ti, typeInfo[Boolean]), typeInfo[Boolean])
            val EmitTriplet(s, m, v) = new Emit(ctx, discardNext).emit(ApplyComparisonOp(EQWithNA(eltVType), In(0, eltType), In(1, eltType)), Env.empty, er, container)
            discardNext.emit(Code(s, m || coerce[Boolean](v)))
            (a, ApplyComparisonOp(Compare(eltVType), In(0, eltType), In(1, eltType)) < 0, sorter.distinctFromSorted(discardNext.invoke(_, _, _, _, _)))
          case ToDict(a) =>
            val elementType = a.pType.asInstanceOf[PStreamable].elementType
            val (k0, k1, keyType) = elementType match {
              case t: PStruct => (GetField(In(0, elementType.virtualType), "key"), GetField(In(1, elementType.virtualType), "key"), t.fieldType("key"))
              case t: PTuple => (GetTupleElement(In(0, elementType.virtualType), 0), GetTupleElement(In(1, elementType.virtualType), 0), t.types(0))
            }
            val discardNext = mb.fb.newMethod(Array[TypeInfo[_]](typeInfo[Region], sorter.ti, typeInfo[Boolean], sorter.ti, typeInfo[Boolean]), typeInfo[Boolean])
            val EmitTriplet(s, m, v) = new Emit(ctx, discardNext).emit(ApplyComparisonOp(EQWithNA(keyType.virtualType), k0, k1), Env.empty, er, container)
            discardNext.emit(Code(s, m || coerce[Boolean](v)))
            (a, ApplyComparisonOp(Compare(keyType.virtualType), k0, k1) < 0, Code(sorter.pruneMissing, sorter.distinctFromSorted(discardNext.invoke(_, _, _, _, _))))
        }

        val compF = vab.ti match {
          case BooleanInfo => sorter.sort(makeDependentSortingFunction[Boolean](compare, env))
          case IntInfo => sorter.sort(makeDependentSortingFunction[Int](compare, env))
          case LongInfo => sorter.sort(makeDependentSortingFunction[Long](compare, env))
          case FloatInfo => sorter.sort(makeDependentSortingFunction[Float](compare, env))
          case DoubleInfo => sorter.sort(makeDependentSortingFunction[Double](compare, env))
        }

        val aout = emitArrayIterator(array)

        val processArrayElts = aout.arrayEmitterFromBuilder(vab)
        EmitTriplet(
          Code(
            vab.clear,
            processArrayElts.setup),
          processArrayElts.m.getOrElse(const(false)),
          Code(
            aout.calcLength,
            processArrayElts.addElements,
            compF,
            distinct,
            sorter.toRegion()))

      case ToArray(a) =>
        emit(a)

      case ToStream(a) =>
        emit(a)

      case x@LowerBoundOnOrderedCollection(orderedCollection, elem, onKey) =>
        val typ: PContainer = coerce[PIterable](orderedCollection.pType).asPContainer
        val a = emit(orderedCollection)
        val e = emit(elem)
        val bs = new BinarySearch(mb, typ, keyOnly = onKey)

        val localA = mb.newLocal[Long]()
        val localElementMB = mb.newLocal[Boolean]()
        val localElementValue = mb.newLocal()(typeToTypeInfo(elem.pType))
        EmitTriplet(
          Code(a.setup, e.setup),
          a.m,
          Code(
            localA := a.value[Long],
            localElementMB := e.m,
            localElementMB.mux(localElementValue.storeAny(defaultValue(elem.pType)), localElementValue.storeAny(e.v)),
            bs.getClosestIndex(localA, localElementMB, localElementValue)))

      case GroupByKey(collection) =>
        //sort collection by group
        val atyp = coerce[PStreamable](collection.pType).asPArray
        val etyp = coerce[PBaseStruct](atyp.elementType)
        val ktyp = etyp.types(0)
        val vtyp = etyp.types(1)
        val eltOut = coerce[PDict](ir.pType).elementType

        val aout = emitArrayIterator(collection)

        val eab = new StagedArrayBuilder(etyp, mb, 16)
        val sorter = new ArraySorter(er, eab)

        val (k1, k2) = etyp match {
          case t: PStruct => GetField(In(0, t), "key") -> GetField(In(1, t), "key")
          case t: PTuple =>
            assert(t.fields(0).index == 0)
            GetTupleElement(In(0, t), 0) -> GetTupleElement(In(1, t), 0)
        }

        val compare = ApplyComparisonOp(Compare(etyp.types(0).virtualType), k1, k2) < 0
        val compF = eab.ti match {
          case BooleanInfo => sorter.sort(makeDependentSortingFunction[Boolean](compare, env))
          case IntInfo => sorter.sort(makeDependentSortingFunction[Int](compare, env))
          case LongInfo => sorter.sort(makeDependentSortingFunction[Long](compare, env))
          case FloatInfo => sorter.sort(makeDependentSortingFunction[Float](compare, env))
          case DoubleInfo => sorter.sort(makeDependentSortingFunction[Double](compare, env))
        }

        val nab = new StagedArrayBuilder(PInt32(), mb, 16)
        val i = mb.newLocal[Int]

        def loadKey(n: Code[Int]): Code[_] =
          Region.loadIRIntermediate(ktyp)(etyp.fieldOffset(coerce[Long](eab(n)), 0))

        def loadValue(n: Code[Int]): Code[_] =
          Region.loadIRIntermediate(vtyp)(etyp.fieldOffset(coerce[Long](eab(n)), 1))

        val srvb = new StagedRegionValueBuilder(mb, ir.pType)

        type E = Env[(TypeInfo[_], Code[Boolean], Code[_])]

        val (lastKey, currKey) = (etyp.virtualType: @unchecked) match {
          case ts: TStruct =>
            GetField(Ref("i-1", ts), ts.fieldNames(0)) -> GetField(Ref("i", ts), ts.fieldNames(0))
          case tt: TTuple =>
            GetTupleElement(Ref("i-1", tt), tt.fields(0).index) -> GetTupleElement(Ref("i", tt), tt.fields(0).index)
        }

        val isSame = emit(
          ApplyComparisonOp(EQWithNA(ktyp.virtualType),
            lastKey,
            currKey),
          Env(
            ("i-1", (typeInfo[Long], eab.isMissing(i-1), eab.apply(i-1))),
            ("i", (typeInfo[Long], eab.isMissing(i), eab.apply(i)))))

        val processArrayElts = aout.arrayEmitterFromBuilder(eab)
        EmitTriplet(Code(eab.clear, processArrayElts.setup), processArrayElts.m.getOrElse(const(false)), Code(
          nab.clear,
          aout.calcLength,
          processArrayElts.addElements,
          compF,
          sorter.pruneMissing,
          eab.size.ceq(0).mux(
            Code(srvb.start(0), srvb.offset),
            Code(
              i := 1,
              nab.add(1),
              Code.whileLoop(i < eab.size,
                isSame.setup,
                (isSame.m || isSame.value[Boolean]).mux(
                  nab.update(nab.size - 1, coerce[Int](nab(nab.size - 1)) + 1),
                  nab.add(1)),
                i += 1),
              i := 0,
              srvb.start(nab.size),
              Code.whileLoop(srvb.arrayIdx < nab.size,
                srvb.addBaseStruct(eltOut, { structbuilder =>
                  Code(
                    structbuilder.start(),
                    structbuilder.addIRIntermediate(ktyp)(loadKey(i)),
                    structbuilder.advance(),
                    structbuilder.addArray(coerce[PStreamable](eltOut.types(1)).asPArray, { arraybuilder =>
                      Code(
                        arraybuilder.start(coerce[Int](nab(srvb.arrayIdx))),
                        Code.whileLoop(arraybuilder.arrayIdx < coerce[Int](nab(srvb.arrayIdx)),
                          etyp.isFieldMissing(coerce[Long](eab(i)), 1).mux(
                            arraybuilder.setMissing(),
                            arraybuilder.addIRIntermediate(etyp.types(1))(loadValue(i))
                          ),
                          i += 1,
                          arraybuilder.advance()
                        ))
                    }))
                }),
                srvb.advance()
              ),
              srvb.offset
            ))))

      case _: ArrayMap | _: ArrayZip | _: ArrayFilter | _: ArrayRange | _: ArrayFlatMap | _: ArrayScan | _: ArrayLeftJoinDistinct | _: RunAggScan | _: ReadPartition =>
        emitArrayIterator(ir).toEmitTriplet(mb, PArray(coerce[PStreamable](ir.pType).elementType))

      case ArrayFold(a, zero, name1, name2, body) =>
        val typ = ir.typ
        val tarray = coerce[TStreamable](a.typ)
        val tti = typeToTypeInfo(typ)
        val eti = typeToTypeInfo(tarray.elementType)
        val xmv = mb.newField[Boolean](name2 + "_missing")
        val xvv = coerce[Any](mb.newField(name2)(eti))
        val xmbody = mb.newField[Boolean](name1 + "_missing_tmp")
        val xmaccum = mb.newField[Boolean](name1 + "_missing")
        val xvaccum = coerce[Any](mb.newField(name1)(tti))
        val bodyenv = env.bind(
          (name1, (tti, xmaccum.load(), xvaccum.load())),
          (name2, (eti, xmv.load(), xvv.load())))

        val codeZ = emit(zero)
        val codeB = emit(body, env = bodyenv)

        val aBase = emitArrayIterator(a)

        val cont = { (m: Code[Boolean], v: Code[_]) =>
          Code(
            xmv := m,
            xvv := xmv.mux(defaultValue(tarray.elementType), v),
            codeB.setup,
            xmbody := codeB.m,
            xvaccum := xmbody.mux(defaultValue(typ), codeB.v),
            xmaccum := xmbody)
        }

        val processAElts = aBase.arrayEmitter(cont)
        val marray = processAElts.m.getOrElse(const(false))

        EmitTriplet(Code(
          codeZ.setup,
          xmaccum := codeZ.m,
          xvaccum := xmaccum.mux(defaultValue(typ), codeZ.v),
          processAElts.setup,
          marray.mux(
            Code(
              xmaccum := true,
              xvaccum := defaultValue(typ)),
            Code(
              aBase.calcLength,
              processAElts.addElements))),
          xmaccum, xvaccum)

      case ArrayFold2(a, acc, valueName, seq, res) =>
        val typ = ir.typ
        val tarray = coerce[TStreamable](a.typ)
        val tti = typeToTypeInfo(typ)
        val eti = typeToTypeInfo(tarray.elementType)
        val xmv = mb.newField[Boolean](valueName + "_missing")
        val xvv = coerce[Any](mb.newField(valueName)(eti))
        val accVars = acc.map { case (name, value) =>
          val ti = typeToTypeInfo(value.typ)
          (name, (ti, mb.newField[Boolean](s"${name}_missing"), mb.newField(name)(ti)))}
        val xmtmp = mb.newField[Boolean]("arrayfold2_missing_tmp")

        val resEnv = env.bindIterable(accVars.map { case (name, (ti, xm, xv)) => (name, (ti, xm.load(), xv.load())) })
        val seqEnv = resEnv.bind(valueName, (eti, xmv.load(), xvv.load()))

        val codeZ = acc.map { case (_, value) => emit(value) }
        val codeSeq = seq.map(emit(_, env = seqEnv))

        val aBase = emitArrayIterator(a)

        val cont = { (m: Code[Boolean], v: Code[_]) =>
          Code(
            xmv := m,
            xvv := xmv.mux(defaultValue(tarray.elementType), v),
            Code(codeSeq.map(_.setup): _*),
            coerce[Unit](Code(codeSeq.zipWithIndex.map { case (et, i) =>
              val (_, (_, accm, accv)) = accVars(i)
              Code(
                xmtmp := et.m,
                accv.storeAny(xmtmp.mux(defaultValue(acc(i)._2.typ): Code[_], et.v)),
                accm := xmtmp
              )
            }: _*)))
        }

        val processAElts = aBase.arrayEmitter(cont)
        val marray = processAElts.m.getOrElse(const(false))

        val xresm = mb.newField[Boolean]
        val xresv = mb.newField(typeToTypeInfo(res.typ))
        val codeR = emit(res, env = resEnv)

        EmitTriplet(Code(
          codeZ.map(_.setup),
          accVars.zipWithIndex.map { case ((_, (ti, xm, xv)), i) =>
            Code(xm := codeZ(i).m, xv.storeAny(xm.mux(defaultValue(acc(i)._2.typ), codeZ(i).v)))
          },
          processAElts.setup,
          marray.mux(
            Code(
              xresm := true,
              xresv.storeAny(defaultValue(res.typ))),
            Code(
              aBase.calcLength,
              processAElts.addElements,
              codeR.setup,
              xresm := codeR.m,
              xresv.storeAny(codeR.v)))),
          xresm, xresv)

      case ArrayFor(a, valueName, body) =>
        val tarray = coerce[TStreamable](a.typ)
        val eti = typeToTypeInfo(tarray.elementType)
        val xmv = mb.newField[Boolean]()
        val xvv = coerce[Any](mb.newField(valueName)(eti))
        val bodyenv = env.bind(
          (valueName, (eti, xmv.load(), xvv.load())))
        val codeB = emit(body, env = bodyenv)
        val aBase = emitArrayIterator(a)
        val cont = { (m: Code[Boolean], v: Code[_]) =>
          Code(
            xmv := m,
            xvv := xmv.mux(defaultValue(tarray.elementType), v),
            codeB.setup)
        }

        val processAElts = aBase.arrayEmitter(cont)
        val ma = processAElts.m.getOrElse(const(false))
        EmitTriplet(
          Code(
            processAElts.setup,
            ma.mux(
              Code._empty,
              Code(aBase.calcLength, processAElts.addElements))),
          const(false),
          Code._empty)

      case x@RunAgg(body, result, _) =>
        val aggs = x.physicalSignatures
        val (newContainer, aggSetup, aggCleanup) = AggContainer.fromFunctionBuilder(aggs, mb.fb, "run_agg")
        val codeBody = emit(body, env = env, container = Some(newContainer))
        val codeRes = emit(result, env = env, container = Some(newContainer))
        val resm = mb.newField[Boolean]()
        val resv = mb.newField("run_agg_result")(typeToTypeInfo(result.pType))
        val aggregation = Code(
          aggSetup,
          codeBody.setup,
          codeRes.setup,
          resm := codeRes.m,
          resv.storeAny(resm.mux(defaultValue(result.pType), codeRes.v)),
          aggCleanup)

        EmitTriplet(aggregation, resm, resv)

      case x@InitOp(i, args, _, op) =>
        val AggContainer(aggs, sc) = container.get
        val physicalSignature = aggs(i).lookup(op)
        args.map(_.pType).zip(physicalSignature.physicalInitOpArgs).foreach { case (l, r) => assert(l == r, s"$l, $r") }
        val rvAgg = agg.Extract.getAgg(aggs(i), op)

        val argVars = args.map(a => emit(a, container = container.flatMap(_.nested(i, init = true)))).toArray
        void(
          sc.newState(i),
          rvAgg.initOp(sc.states(i), argVars))

      case x@SeqOp(i, args, _, op) =>
        val AggContainer(aggs, sc) = container.get
        val aggSig = aggs(i)
        args.map(_.pType).zip(aggSig.lookup(op).physicalSeqOpArgs).foreach { case (l, r) => assert(l == r, s"$l, $r") }
        val rvAgg = agg.Extract.getAgg(aggSig, op)

        val argVars = args.map(a => emit(a, container = container.flatMap(_.nested(i, init = false)))).toArray
        void(rvAgg.seqOp(sc.states(i), argVars))

      case x@CombOp(i1, i2, _) =>
        val AggContainer(aggs, sc) = container.get
        val aggSig = aggs(i1)
        assert(agg.Extract.compatible(aggs(i2), aggSig), s"${ aggs(i2) } vs $aggSig")
        val rvAgg = agg.Extract.getAgg(aggSig, aggSig.default)

        void(rvAgg.combOp(sc.states(i1), sc.states(i2)))

      case x@ResultOp(start, _) =>
        val newRegion = mb.newField[Region]
        val AggContainer(aggs, sc) = container.get
        val srvb = new StagedRegionValueBuilder(EmitRegion(mb, newRegion), x.pType)
        val addFields = mb.fb.wrapVoids(Array.tabulate(aggs.length) { j =>
          val idx = start + j
          val rvAgg = agg.Extract.getAgg(aggs(j), aggs(j).default)
          Code(
            rvAgg.result(sc.states(idx), srvb),
            srvb.advance())
        }, "aggs_result")

        present(Code(
          newRegion := region,
          srvb.start(),
          addFields,
          sc.store,
          srvb.offset))

      case x@CombOpValue(i, value, sig) =>
        val AggContainer(aggs, sc) = container.get
        val aggSig = aggs(i)
        val rvAgg = agg.Extract.getAgg(aggSig, aggSig.default)
        val newState = rvAgg.createState(mb.fb)

        val t = value.pType.asInstanceOf[PBinary]
        val xValue = emit(value)
        EmitTriplet(
          Code(xValue.setup,
          xValue.m.mux(
            Code._fatal("cannot combOp a missing value"),
            Code(
              newState.createState,
              newState.deserializeFromBytes(t, coerce[Long](xValue.v)),
              rvAgg.combOp(sc.states(i), newState)
            ))),
          const(false),
          Code._empty[Unit])

      case x@AggStateValue(i, _) =>
        val AggContainer(_, sc) = container.get
        present(sc.states(i).serializeToRegion(coerce[PBinary](x.pType), region))

      case x@SerializeAggs(start, sIdx, spec, sigs) =>
        val AggContainer(_, sc) = container.get
        val ob = mb.newField[OutputBuffer]
        val baos = mb.newField[ByteArrayOutputStream]

        val serialize = Array.range(start, start + sigs.length)
          .map { idx => sc.states(idx).serialize(spec)(ob) }

        void(
          baos := Code.newInstance[ByteArrayOutputStream](),
          ob := spec.buildCodeOutputBuffer(baos),
          mb.fb.wrapVoids(serialize, "serialize_aggs"),
          ob.invoke[Unit]("flush"),
          ob.invoke[Unit]("close"),
          mb.fb.setSerializedAgg(sIdx, baos.invoke[Array[Byte]]("toByteArray")),
          sc.store)

      case DeserializeAggs(start, sIdx, spec, sigs) =>
        val AggContainer(_, sc) = container.get
        val ib = mb.newField[InputBuffer]

        val ns = sigs.length
        val deserializers = sc.states.states
          .slice(start, start + ns)
          .map(sc => sc.deserialize(BufferSpec.defaultUncompressed))

        val init = coerce[Unit](Code(Array.range(start, start + ns)
          .map(i => sc.newState(i)): _*))

        val unserialize = Array.tabulate(ns) { j =>
          deserializers(j)(ib)
        }

        void(
          init,
          ib := spec.buildCodeInputBuffer(
            Code.newInstance[ByteArrayInputStream, Array[Byte]](
              mb.fb.getSerializedAgg(sIdx))),
          mb.fb.wrapVoids(unserialize, "deserialize_aggs"))

      case Begin(xs) =>
        EmitTriplet(
          wrapToMethod(xs) { case (_, t, code) =>
            code.setup
          },
          const(false),
          Code._empty)

      case x@MakeStruct(fields) =>
        val srvb = new StagedRegionValueBuilder(mb, x.pType)
        val addFields = { (newMB: EmitMethodBuilder, t: PType, v: EmitTriplet) =>
          Code(
            v.setup,
            v.m.mux(srvb.setMissing(), srvb.addIRIntermediate(t)(v.v)),
            srvb.advance())
        }
        present(Code(srvb.start(init = true), wrapToMethod(fields.map(_._2))(addFields), srvb.offset))

      case x@SelectFields(oldStruct, fields) =>
        val old = emit(oldStruct)
        val oldt = coerce[PStruct](oldStruct.pType)
        val oldv = mb.newField[Long]
        val srvb = new StagedRegionValueBuilder(mb, x.pType)

        val addFields = fields.map { name =>
          new EstimableEmitter[EmitMethodBuilderLike] {
            def estimatedSize: Int = 20

            def emit(mbLike: EmitMethodBuilderLike): Code[Unit] = {
              val i = oldt.fieldIdx(name)
              val t = oldt.types(i)
              val fieldMissing = oldt.isFieldMissing(oldv, i)
              val fieldValue = Region.loadIRIntermediate(t)(oldt.fieldOffset(oldv, i))
              Code(
                fieldMissing.mux(
                  srvb.setMissing(),
                  srvb.addIRIntermediate(t)(fieldValue)),
                srvb.advance())
            }
          }
        }

        EmitTriplet(
          old.setup,
          old.m,
          Code(
            oldv := old.value[Long],
            srvb.start(),
            EmitUtils.wrapToMethod(addFields, new EmitMethodBuilderLike(this)),
            srvb.offset))


      case x@InsertFields(old, fields, fieldOrder) =>
        if (fields.isEmpty)
          emit(old)
        else
          old.pType match {
            case oldtype: PStruct =>
              val codeOld = emit(old)
              val xo = mb.newField[Long]
              val updateMap = Map(fields: _*)
              val srvb = new StagedRegionValueBuilder(mb, x.pType)

              val addFields = { (newMB: EmitMethodBuilder, t: PType, v: EmitTriplet) =>
                Code(
                  v.setup,
                  v.m.mux(srvb.setMissing(), srvb.addIRIntermediate(t)(v.v)),
                  srvb.advance())
              }

              val opSize: Int = 20
              val items = x.pType.fields.map { f =>
                updateMap.get(f.name) match {
                  case Some(vir) =>
                    new EstimableEmitter[EmitMethodBuilderLike] {
                      def estimatedSize: Int = vir.size * opSize

                      def emit(mbLike: EmitMethodBuilderLike): Code[Unit] =
                        addFields(mbLike.mb, vir.pType, mbLike.emit.emit(vir, env, EmitRegion.default(mbLike.mb), container))
                    }
                  case None =>
                    val oldField = oldtype.field(f.name)
                    new EstimableEmitter[EmitMethodBuilderLike] {
                      def estimatedSize: Int = 20

                      def emit(mbLike: EmitMethodBuilderLike): Code[Unit] =
                        Code(
                          oldtype.isFieldMissing(xo, oldField.index).mux(
                            srvb.setMissing(),
                            srvb.addIRIntermediate(f.typ)(Region.loadIRIntermediate(oldField.typ)(oldtype.fieldOffset(xo, oldField.index)))),
                          srvb.advance())
                    }
                }
              }

              EmitTriplet(
                codeOld.setup,
                codeOld.m,
                Code(
                  srvb.start(init = true),
                  xo := coerce[Long](codeOld.v),
                  EmitUtils.wrapToMethod(items, new EmitMethodBuilderLike(this)),
                  srvb.offset))
            case _ =>
              val newIR = MakeStruct(fields)
              emit(newIR)
          }

      case GetField(o, name) =>
        val t = coerce[PStruct](o.pType)
        val fieldIdx = t.fieldIdx(name)
        val codeO = emit(o)
        val xmo = mb.newLocal[Boolean]()
        val xo = mb.newLocal[Long]
        val setup = Code(
          codeO.setup,
          xmo := codeO.m,
          xo := coerce[Long](xmo.mux(defaultValue(t), codeO.v)))
        EmitTriplet(setup,
          xmo || !t.isFieldDefined(xo, fieldIdx),
          Region.loadIRIntermediate(t.types(fieldIdx))(t.fieldOffset(xo, fieldIdx)))

      case x@MakeTuple(fields) =>
        val srvb = new StagedRegionValueBuilder(mb, x.pType)
        val addFields = { (newMB: EmitMethodBuilder, t: PType, v: EmitTriplet) =>
          Code(
            v.setup,
            v.m.mux(srvb.setMissing(), srvb.addIRIntermediate(t)(v.v)),
            srvb.advance())
        }
        present(Code(srvb.start(init = true), wrapToMethod(fields.map(_._2))(addFields), srvb.offset))

      case GetTupleElement(o, i) =>
        val t = coerce[PTuple](o.pType)
        val idx = t.fieldIndex(i)
        val codeO = emit(o)
        val xmo = mb.newLocal[Boolean]()
        val xo = mb.newLocal[Long]
        val setup = Code(
          codeO.setup,
          xmo := codeO.m,
          xo := coerce[Long](xmo.mux(defaultValue(t), codeO.v)))
        EmitTriplet(setup,
          xmo || !t.isFieldDefined(xo, idx),
          Region.loadIRIntermediate(t.types(idx))(t.fieldOffset(xo, idx)))

      case In(i, typ) =>
        normalArgument(i, typ)
      case Die(m, typ) =>
        val cm = emit(m)
        EmitTriplet(
          Code(
            cm.setup,
            Code._throw(Code.newInstance[HailException, String](
              cm.m.mux[String](
                "<exception message missing>",
                coerce[String](StringFunctions.wrapArg(er, m.pType)(cm.v)))))),
          false,
          defaultValue(typ))
      case ir@ApplyIR(fn, args) =>
        assert(!ir.inline)
        val mfield = mb.newField[Boolean]
        val vfield = mb.newField()(typeToTypeInfo(ir.typ))

        val addFields = { (newMB: EmitMethodBuilder, t: PType, v: EmitTriplet) =>
          Code(
            v.setup,
            mfield := v.m,
            mfield.mux(
              vfield.storeAny(defaultValue(t)),
              vfield.storeAny(v.v)))
        }

        EmitTriplet(
          wrapToMethod(FastSeq(ir.explicitNode))(addFields),
          mfield, vfield)

      case ir@Apply(fn, args, rt) =>
        val impl = ir.implementation
        val unified = impl.unify(args.map(_.typ) :+ rt)
        assert(unified)

        val argPTypes = args.map(_.pType)
        val meth =
          methods(fn).filter { case (argt, rtExpected, _) =>
            argt.zip(argPTypes).forall { case (t1, t2) => t1 == t2 } &&
              rtExpected == ir.pType
          } match {
            case Seq((_, _, funcMB)) =>
              funcMB
            case Seq() =>
              val methodbuilder = impl.getAsMethod(mb.fb, rt, argPTypes: _*)
              methods.update(fn, methods(fn) :+ ((argPTypes, ir.pType, methodbuilder)))
              methodbuilder
          }
        val codeArgs = args.map(emit(_))
        val vars = args.map { a => coerce[Any](mb.newLocal()(typeToTypeInfo(a.typ))) }
        val ins = vars.zip(codeArgs.map(_.v)).map { case (l, i) => l := i }
        val value = Code(ins :+ meth.invoke(mb.getArg[Region](1).load() +: vars.map { a => a.load() }: _*): _*)
        strict(value, codeArgs: _*)
      case x@ApplySeeded(fn, args, seed, rt) =>
        val codeArgs = args.map(a => (a.pType, emit(a)))
        val impl = x.implementation
        val unified = impl.unify(args.map(_.typ) :+ rt)
        assert(unified)
        impl.setSeed(seed)
        impl.apply(er, codeArgs: _*)
      case x@ApplySpecial(_, args, rt) =>
        val codeArgs = args.map(a => (a.pType, emit(a)))
        val impl = x.implementation
        impl.argTypes.foreach(_.clear())
        val unified = impl.unify(args.map(_.typ) :+ rt)
        assert(unified)
        impl.apply(er, codeArgs: _*)
      case x@MakeNDArray(dataIR, shapeIR, rowMajorIR) =>
        val xP = x.pType
        val dataContainer = dataIR.pType
        val shapePType = coerce[PTuple](shapeIR.pType)
        val dataPType = xP.data.pType
        val nDims = shapePType.size

        val datat = emit(dataIR)
        val shapet = emit(shapeIR)
        val rowMajort = emit(rowMajorIR)

        val requiredData = dataPType.checkedConvertFrom(mb, region, datat.value[Long], coerce[PArray](dataContainer), "NDArray cannot have missing data")
        val shapeAddress = mb.newField[Long]

        val shapeTuple = new CodePTuple(shapePType, shapeAddress)

        val shapeVariables = (0 until nDims).map(_ => mb.newLocal[Long]).toArray

        def shapeBuilder(srvb: StagedRegionValueBuilder): Code[Unit] = {
          Code(
            srvb.start(),
            Code.foreach(0 until nDims) { index =>
              Code(
                srvb.addLong(shapeVariables(index)),
                srvb.advance()
              )
            }
          )
        }

        val setup = Code(
          shapet.setup,
          datat.setup,
          rowMajort.setup
        )
        val result = Code(
          shapeAddress := shapet.value[Long],
          Code.foreach(0 until nDims) {index =>
            shapeTuple.isMissing(index).mux[Unit](
              Code._fatal(s"shape missing at index $index"),
              shapeVariables(index) := shapeTuple(index)
            )
          },
          xP.construct(0, 0, shapeBuilder, xP.makeDefaultStridesBuilder(shapeVariables.map(_.load()), mb), requiredData, mb)
        )
        EmitTriplet(setup, datat.m || shapet.m, result)
      case NDArrayShape(ndIR) =>
        val ndt = emit(ndIR)
        val ndP = ndIR.pType.asInstanceOf[PNDArray]

        EmitTriplet(ndt.setup, ndt.m, ndP.shape.load(ndt.value[Long]))
      case NDArrayRef(nd, idxs) =>
        val ndt = emit(nd)
        val idxst = idxs.map(emit(_))
        val childPType = coerce[PNDArray](nd.pType)
        val ndAddress = mb.newField[Long]
        val overallMissing = mb.newField[Boolean]

        val idxFields = idxst.map(_ => mb.newField[Long])
        val idxFieldsBinding = Code(
          idxFields.zip(idxst).map{ case (field, idxTriplet) =>
            field := idxTriplet.value[Long]
          }
        )
        val cachedIdxVals = idxFields.map(_.load()).toArray

        val targetElementPosition = childPType.getElementAddress(cachedIdxVals, ndAddress, mb)

        val setup = coerce[Unit](Code(
          ndt.setup,
          overallMissing := ndt.m,
          Code(idxst.map(_.setup)),
          Code.foreach(idxst.map(_.m)){ idxMissingness =>
            overallMissing := overallMissing || idxMissingness
          }
        ))

        val value = Code(
          ndAddress := ndt.value[Long],
          idxFieldsBinding,
          childPType.outOfBounds(cachedIdxVals, ndAddress, mb).orEmpty(Code._fatal("Index out of bounds")),
          Region.loadIRIntermediate(childPType.data.pType.elementType)(targetElementPosition)
        )

        EmitTriplet(setup, overallMissing, value)
      case x@NDArrayReindex(child, indexMap) =>
        val childt = emit(child)
        val childAddress = mb.newField[Long]
        val childPType = coerce[PNDArray](child.pType)

        val childShape = new CodePTuple(childPType.shape.pType, childPType.shape.load(childAddress))
        val childStrides = new CodePTuple(childPType.strides.pType, childPType.strides.load(childAddress))

        val setup = childt.setup
        val value = Code(
          childAddress := childt.value[Long],
          x.pType.construct(
            childPType.flags.load(childAddress),
            childPType.offset.load(childAddress),
            { srvb =>
              Code(
                srvb.start(),
                Code.foreach(indexMap) {childIndex =>
                  Code(
                    srvb.addLong(if (childIndex < childPType.nDims) childShape(childIndex) else 1L),
                    srvb.advance()
                  )
                }
              )
            },
            { srvb =>
              Code(
                srvb.start(),
                Code.foreach(indexMap) {index =>
                  Code(
                    srvb.addLong(if (index < childPType.nDims) childStrides(index) else 0L),
                    srvb.advance()
                  )
                }
              )
            },
            childPType.data.load(childAddress),
            mb
          )
        )
        EmitTriplet(setup, childt.m, value)
      case x: NDArrayMap  =>  emitDeforestedNDArray(x)
      case x: NDArrayMap2 =>  emitDeforestedNDArray(x)
      case x: NDArrayReshape => emitDeforestedNDArray(x)
      case x: NDArrayConcat => emitDeforestedNDArray(x)
      case x: NDArraySlice => emitDeforestedNDArray(x)

      case NDArrayMatMul(lChild, rChild) =>
        val lT = emitNDArrayStandardStrides(lChild)
        val rT = emitNDArrayStandardStrides(rChild)

        val lPType = coerce[PNDArray](lChild.pType)
        val rPType = coerce[PNDArray](rChild.pType)

        val leftND = mb.newField[Long]
        val rightND = mb.newField[Long]

        val leftShape = lPType.shape.load(leftND)
        val rightShape = rPType.shape.load(rightND)

        val lShapeTuple = new CodePTuple(lPType.shape.pType, leftShape)
        val rShapeTuple = new CodePTuple(rPType.shape.pType, rightShape)

        val (leftShapeArraySetup, leftShapeArray) = (0 until lPType.nDims).map(i => coerce[Long](lShapeTuple(i))).toArray.cacheEntries(mb, LongInfo)
        val (rightShapeArraySetup, rightShapeArray) = (0 until rPType.nDims).map(i => coerce[Long](rShapeTuple(i))).toArray.cacheEntries(mb, LongInfo)

        val (unifyShapeSetup, unifiedShapeArray) = NDArrayEmitter.matmulShape(leftShapeArray, rightShapeArray)

        val leftBroadcastMask = if (lPType.nDims > 2) NDArrayEmitter.broadcastMask(leftShapeArray) else Array[Code[Long]]()
        val rightBroadcastMask = if (rPType.nDims > 2) NDArrayEmitter.broadcastMask(rightShapeArray) else Array[Code[Long]]()

        val missingSetup = Code(
          lT.setup,
          rT.setup
        )

        val shapeSetup = Code(
          leftND := lT.value[Long],
          rightND := rT.value[Long],
          leftShapeArraySetup,
          rightShapeArraySetup,
          unifyShapeSetup
        )

        val outputPType = PNDArray(lPType.elementType, TNDArray.matMulNDims(lPType.nDims, rPType.nDims), true)

        val numericElementType = coerce[PNumeric](lPType.elementType)

        val eVti = typeToTypeInfo(numericElementType.virtualType)

        val isMissing = lT.m || rT.m

        if ((lPType.elementType.isInstanceOf[PFloat64] || lPType.elementType.isInstanceOf[PFloat32]) && lPType.nDims == 2 && rPType.nDims == 2) {
          val leftDataAddress = lPType.data.load(leftND)
          val rightDataAddress = rPType.data.load(rightND)

          val leftColumnMajorAddress = mb.newLocal[Long]
          val rightColumnMajorAddress = mb.newLocal[Long]
          val answerColumnMajorAddress = mb.newLocal[Long]
          val answerRowMajorPArrayAddress = mb.newField[Long]
          val M = leftShapeArray(lPType.nDims - 2)
          val N = rightShapeArray(rPType.nDims - 1)
          val K = leftShapeArray(lPType.nDims - 1)

          val LDA = M
          val LDB = K
          val LDC = M
          val elementByteSize = lPType.elementType.byteSize

          val multiplyViaDGEMM = Code(
            shapeSetup,
            leftColumnMajorAddress := Code.invokeStatic[Memory, Long, Long]("malloc", M * K * elementByteSize),
            rightColumnMajorAddress := Code.invokeStatic[Memory, Long, Long]("malloc", K * N * elementByteSize),
            answerColumnMajorAddress := Code.invokeStatic[Memory, Long, Long]("malloc", M * N * elementByteSize),

            Code.invokeScalaObject[Long, Long, Long, Long, Long, Unit](LinalgCodeUtils.getClass,
              method="copyRowMajorToColumnMajor", lPType.data.pType.firstElementOffset(leftDataAddress), leftColumnMajorAddress, M, K, lPType.elementType.byteSize),
            Code.invokeScalaObject[Long, Long, Long, Long, Long, Unit](LinalgCodeUtils.getClass,
              method="copyRowMajorToColumnMajor", rPType.data.pType.firstElementOffset(rightDataAddress), rightColumnMajorAddress, K, N, rPType.elementType.byteSize),
            lPType.elementType match {
              case PFloat32(_) =>
                Code.invokeScalaObject[String, String, Int, Int, Int, Float, Long, Int, Long, Int, Float, Long, Int, Unit](BLAS.getClass, method="sgemm",
                  "N",
                  "N",
                  M.toI,
                  N.toI,
                  K.toI,
                  1.0f,
                  leftColumnMajorAddress,
                  LDA.toI,
                  rightColumnMajorAddress,
                  LDB.toI,
                  0.0f,
                  answerColumnMajorAddress,
                  LDC.toI
                )
              case PFloat64(_) =>
                Code.invokeScalaObject[String, String, Int, Int, Int, Double, Long, Int, Long, Int, Double, Long, Int, Unit](BLAS.getClass, method="dgemm",
                  "N",
                  "N",
                  M.toI,
                  N.toI,
                  K.toI,
                  1.0,
                  leftColumnMajorAddress,
                  LDA.toI,
                  rightColumnMajorAddress,
                  LDB.toI,
                  0.0,
                  answerColumnMajorAddress,
                  LDC.toI
                )
            },
            answerRowMajorPArrayAddress := outputPType.data.pType.allocate(region, (M * N).toI),
            outputPType.data.pType.stagedInitialize(answerRowMajorPArrayAddress, (M * N).toI),
            Code.invokeScalaObject[Long, Long, Long, Long, Long, Unit](LinalgCodeUtils.getClass,
              method="copyColumnMajorToRowMajor", answerColumnMajorAddress, outputPType.data.pType.firstElementOffset(answerRowMajorPArrayAddress, (M * N).toI), M, N, lPType.elementType.byteSize),
            Code.invokeStatic[Memory, Long, Unit]("free", leftColumnMajorAddress.load()),
            Code.invokeStatic[Memory, Long, Unit]("free", rightColumnMajorAddress.load()),
            Code.invokeStatic[Memory, Long, Unit]("free", answerColumnMajorAddress.load()),
            outputPType.construct(0, 0, outputPType.makeShapeBuilder(Array(M, N)), outputPType.makeDefaultStridesBuilder(Array(M, N), mb), answerRowMajorPArrayAddress, mb)
          )

          EmitTriplet(missingSetup, isMissing, multiplyViaDGEMM)
        } else {
          val emitter = new NDArrayEmitter(mb, outputPType.nDims, unifiedShapeArray, lPType.shape.pType, lPType.elementType, shapeSetup, missingSetup, isMissing) {
            override def outputElement(idxVars: Array[Code[Long]]): Code[_] = {
              val element = coerce[Any](mb.newField("matmul_element")(eVti))
              val k = mb.newField[Long]

              val (lIndices: Array[Code[Long]], rIndices: Array[Code[Long]]) = (lPType.nDims, rPType.nDims, idxVars.toSeq) match {
                case (1, 1, Seq()) => (Array[Code[Long]](k), Array[Code[Long]](k))
                case (1, _, stack :+ m) =>
                  val rStackVars = NDArrayEmitter.zeroBroadcastedDims(stack.toArray, rightBroadcastMask)
                  (Array(k.load()), rStackVars :+ k.load() :+ m)
                case (_, 1, stack :+ n) =>
                  val lStackVars = NDArrayEmitter.zeroBroadcastedDims(stack.toArray, leftBroadcastMask)
                  (lStackVars :+ n :+ k.load(), Array(k.load()))
                case (_, _, stack :+ n :+ m) => {
                  val lStackVars = NDArrayEmitter.zeroBroadcastedDims(stack.toArray, leftBroadcastMask)
                  val rStackVars = NDArrayEmitter.zeroBroadcastedDims(stack.toArray, rightBroadcastMask)
                  (lStackVars :+ n :+ k.load(), rStackVars :+ k.load() :+  m)
                }
              }

              val lElem = lPType.loadElementToIRIntermediate(lIndices, leftND, mb)
              val rElem = rPType.loadElementToIRIntermediate(rIndices, rightND, mb)
              val kLen = mb.newField[Long]

              val innerMethod = mb.fb.newMethod(eVti)

              val loopCode = Code(
                k := 0L,
                kLen := leftShapeArray(lPType.nDims - 1),
                element := numericElementType.zero,
                Code.whileLoop(k < kLen,
                  element := numericElementType.add(numericElementType.multiply(lElem, rElem), element),
                  k := k + 1L
                ),
                element
              )
              innerMethod.emit(loopCode)
              innerMethod.invoke()
            }
          }
          emitter.emit(outputPType)
        }

      case x@NDArrayQR(nd, mode) =>
        // See here to understand different modes: https://docs.scipy.org/doc/numpy/reference/generated/numpy.linalg.qr.html
        val ndt = emit(nd)
        val ndAddress = mb.newField[Long]
        val ndPType = nd.pType.asInstanceOf[PNDArray]

        val shapeAddress = ndPType.shape.load(ndAddress)
        val shapeTuple = new CodePTuple(ndPType.shape.pType, shapeAddress)
        val shapeArray = (0 until ndPType.shape.pType.nFields).map(shapeTuple[Long](_)).toArray

        val LWORKAddress = mb.newLocal[Long]

        val M = shapeArray(0)
        val N = shapeArray(1)
        val K = (M < N).mux(M, N)
        val LDA = M // Possible stride tricks could change this in the future.
        val LWORK = Region.loadDouble(LWORKAddress).toI

        val dataAddress = ndPType.data.load(ndAddress)

        val tauPType = PArray(PFloat64Required, true)
        val tauAddress = mb.newField[Long]
        val workAddress = mb.newField[Long]
        val aAddressDGEQRF = mb.newField[Long] // Should be column major
        val rDataAddress = mb.newField[Long]
        val aNumElements = mb.newField[Long]

        val infoDGEQRFResult = mb.newLocal[Int]
        val infoDGEQRFErrorTest = (extraErrorMsg: String) => (infoDGEQRFResult cne  0)
          .orEmpty(Code._fatal(const(s"LAPACK error DGEQRF. $extraErrorMsg Error code = ").concat(infoDGEQRFResult.toS)))

        val computeHAndTau = Code(
          ndAddress := ndt.value[Long],
          aNumElements := ndPType.numElements(shapeArray, mb),

          // Make some space for the column major form (which means copying the input)
          aAddressDGEQRF := ndPType.data.pType.allocate(region, aNumElements.toI),
          ndPType.data.pType.stagedInitialize(aAddressDGEQRF, aNumElements.toI),
          ndPType.copyRowMajorToColumnMajor(dataAddress, aAddressDGEQRF, M, N, mb),

          tauAddress := tauPType.allocate(region, K.toI),
          tauPType.stagedInitialize(tauAddress, K.toI),

          LWORKAddress := region.allocate(8L, 8L),

          infoDGEQRFResult := Code.invokeScalaObject[Int, Int, Long, Int, Long, Long, Int, Int](LAPACK.getClass, "dgeqrf",
            M.toI,
            N.toI,
            ndPType.data.pType.elementOffset(aAddressDGEQRF, aNumElements.toI, 0),
            LDA.toI,
            tauPType.elementOffset(tauAddress, K.toI, 0),
            LWORKAddress,
            -1
          ),
          infoDGEQRFErrorTest("Failed size query."),

          workAddress := Code.invokeStatic[Memory, Long, Long]("malloc", LWORK.toL * 8L),

          infoDGEQRFResult := Code.invokeScalaObject[Int, Int, Long, Int, Long, Long, Int, Int](LAPACK.getClass, "dgeqrf",
            M.toI,
            N.toI,
            ndPType.data.pType.elementOffset(aAddressDGEQRF, aNumElements.toI, 0),
            LDA.toI,
            tauPType.elementOffset(tauAddress, K.toI, 0),
            workAddress,
            LWORK
          ),
          Code.invokeStatic[Memory, Long, Unit]("free", workAddress.load()),
          infoDGEQRFErrorTest("Failed to compute H and Tau.")
        )

        val result = if (mode == "raw") {
          val rawPType = x.pType.asInstanceOf[PTuple]
          val rawOutputSrvb = new StagedRegionValueBuilder(mb, x.pType, region)
          val hPType = rawPType.types(0).asInstanceOf[PNDArray]
          val tauPType = rawPType.types(1).asInstanceOf[PNDArray]

          val hShapeArray = Array(N, M)
          val hShapeBuilder = hPType.makeShapeBuilder(hShapeArray)
          val hStridesBuilder = hPType.makeDefaultStridesBuilder(hShapeArray, mb)

          val tauShapeBuilder = tauPType.makeShapeBuilder(Array(K))
          val tauStridesBuilder = tauPType.makeDefaultStridesBuilder(Array(K), mb)

          val h = hPType.construct(0, 0, hShapeBuilder, hStridesBuilder, aAddressDGEQRF, mb)
          val tau = tauPType.construct(0, 0, tauShapeBuilder, tauStridesBuilder, tauAddress, mb)

          val constructHAndTauTuple = Code(
            rawOutputSrvb.start(),
            rawOutputSrvb.addIRIntermediate(hPType)(h),
            rawOutputSrvb.advance(),
            rawOutputSrvb.addIRIntermediate(tauPType)(tau),
            rawOutputSrvb.advance(),
            rawOutputSrvb.end()
          )

          Code(
            computeHAndTau,
            constructHAndTauTuple
          )
        }
        else {
          val currRow = mb.newField[Int]
          val currCol = mb.newField[Int]

          val (rPType, rShapeArray) = if (mode == "r") {
            (x.pType.asInstanceOf[PNDArray], Array(K, N))
          } else if (mode == "complete") {
            (x.pType.asInstanceOf[PTuple].types(1).asInstanceOf[PNDArray], Array(M, N))
          } else if (mode == "reduced") {
            (x.pType.asInstanceOf[PTuple].types(1).asInstanceOf[PNDArray], Array(K, N))
          } else {
            throw new AssertionError(s"Unsupported QR mode $mode")
          }

          val rShapeBuilder = rPType.makeShapeBuilder(rShapeArray)
          val rStridesBuilder = rPType.makeDefaultStridesBuilder(rShapeArray, mb)

          //This block assumes rDataAddress is a row major ndarray.
          val zeroOutLowerTriangle =
            Code.forLoop(currRow := 0, currRow < M.toI, currRow := currRow + 1,
              Code.forLoop(currCol := 0, currCol < N.toI, currCol := currCol + 1,
                (currRow > currCol).orEmpty(
                  Region.storeDouble(
                    ndPType.data.pType.elementOffset(rDataAddress, aNumElements.toI, currRow * N.toI + currCol),
                    0.0)
                )
              )
            )

          val computeR = Code(
            // Note: this always makes room for the (M, N) R, and in cases where we need only the (K, N) R the smaller shape
            // results in these elements being ignored. When everything is column major all the time should be easy to fix.
            rDataAddress := rPType.data.pType.allocate(region, aNumElements.toI),
            rPType.data.pType.stagedInitialize(rDataAddress, aNumElements.toI),
            rPType.copyColumnMajorToRowMajor(aAddressDGEQRF,
              rDataAddress, M, N, mb),
            zeroOutLowerTriangle,
            rPType.construct(0, 0, rShapeBuilder, rStridesBuilder, rDataAddress, mb)
          )

          if (mode == "r") {
            Code(
              computeHAndTau,
              computeR
            )
          }
          else {
            val crPType = x.pType.asInstanceOf[PTuple]
            val crOutputSrvb = new StagedRegionValueBuilder(mb, crPType, region)

            val qPType = crPType.types(0).asInstanceOf[PNDArray]
            val qShapeArray = if (mode == "complete") Array(M, M) else Array(M, K)
            val qShapeBuilder = qPType.makeShapeBuilder(qShapeArray)
            val qStridesBuilder = qPType.makeDefaultStridesBuilder(qShapeArray, mb)

            val rNDArrayAddress = mb.newField[Long]
            val qDataAddress = mb.newField[Long]

            val infoDORGQRResult = mb.newField[Int]
            val infoDORQRErrorTest = (extraErrorMsg: String) => (infoDORGQRResult cne 0)
              .orEmpty(Code._fatal(const(s"LAPACK error DORGQR. $extraErrorMsg Error code = ").concat(infoDORGQRResult.toS)))

            val qCondition = const(mode == "complete") && (M > N)
            val numColsToUse = qCondition.mux(M, K)
            val aAddressDORGQR = mb.newField[Long]

            val qNumElements = M * numColsToUse

            val computeCompleteOrReduced = Code(
              qCondition.mux(
                Code(
                  aAddressDORGQR := ndPType.data.pType.allocate(region, qNumElements.toI),
                  qPType.data.pType.stagedInitialize(aAddressDORGQR, qNumElements.toI),
                  Region.copyFrom(ndPType.data.pType.firstElementOffset(aAddressDGEQRF, aNumElements.toI),
                    qPType.data.pType.firstElementOffset(aAddressDORGQR, qNumElements.toI), aNumElements * 8L)
                ),
                aAddressDORGQR := aAddressDGEQRF
              ),

              // Query optimal size for work array
              infoDORGQRResult := Code.invokeScalaObject[Int, Int, Int, Long, Int, Long, Long, Int, Int](LAPACK.getClass, "dorgqr",
                M.toI,
                numColsToUse.toI,
                K.toI,
                ndPType.data.pType.firstElementOffset(aAddressDORGQR, aNumElements.toI),
                LDA.toI,
                tauPType.firstElementOffset(tauAddress, K.toI),
                LWORKAddress,
                -1
              ),
              infoDORQRErrorTest("Failed size query."),

              workAddress := Code.invokeStatic[Memory, Long, Long]("malloc", LWORK.toL * 8L),

              infoDORGQRResult := Code.invokeScalaObject[Int, Int, Int, Long, Int, Long, Long, Int, Int](LAPACK.getClass, "dorgqr",
                M.toI,
                numColsToUse.toI,
                K.toI,
                ndPType.data.pType.elementOffset(aAddressDORGQR, (M * numColsToUse).toI, 0),
                LDA.toI,
                tauPType.elementOffset(tauAddress, K.toI, 0),
                workAddress,
                LWORK
              ),
              Code.invokeStatic[Memory, Long, Unit]("free", workAddress.load()),
              infoDORQRErrorTest("Failed to compute Q."),

              qDataAddress := qPType.data.pType.allocate(region, qNumElements.toI),
              qPType.data.pType.stagedInitialize(qDataAddress, qNumElements.toI),
              qPType.copyColumnMajorToRowMajor(aAddressDORGQR, qDataAddress, M, numColsToUse, mb),

              crOutputSrvb.start(),
              crOutputSrvb.addIRIntermediate(qPType)(qPType.construct(0, 0, qShapeBuilder, qStridesBuilder, qDataAddress, mb)),
              crOutputSrvb.advance(),
              crOutputSrvb.addIRIntermediate(rPType)(rNDArrayAddress),
              crOutputSrvb.advance(),
              crOutputSrvb.end()
            )

            Code(
              computeHAndTau,
              rNDArrayAddress := computeR,
              computeCompleteOrReduced
            )
          }
        }
        EmitTriplet(ndt.setup, ndt.m, result)


      case x@CollectDistributedArray(contexts, globals, cname, gname, body) =>
        val ctxType = coerce[PArray](contexts.pType).elementType
        val gType = globals.pType
        val bType = body.pType

        val ctxTypeTuple = PTuple(ctxType)
        val gTypeTuple = PTuple(gType)
        val bTypeTuple = PTuple(bType)

        val spec = BufferSpec.defaultUncompressed
        val parentFB = mb.fb

        val cCodec = TypedCodecSpec(ctxTypeTuple, spec)
        val gCodec = TypedCodecSpec(gTypeTuple, spec)
        val bCodec = TypedCodecSpec(bTypeTuple, spec)

        val functionID: String = {
          val bodyFB = EmitFunctionBuilder[Region, Array[Byte], Array[Byte], Array[Byte]]("collect_distributed_array")
          val bodyMB = bodyFB.newMethod(Array[TypeInfo[_]](typeInfo[Region], typeToTypeInfo(ctxType), typeInfo[Boolean], typeToTypeInfo(gType), typeInfo[Boolean]), typeInfo[Long])

          val (cRetPtype, cDec) = cCodec.buildEmitDecoderF[Long](ctxTypeTuple.virtualType, bodyFB)
          val (gRetPtype, gDec) = gCodec.buildEmitDecoderF[Long](gTypeTuple.virtualType, bodyFB)
          val bEnc = bCodec.buildEmitEncoderF[Long](bTypeTuple, bodyFB)
          val bOB = bodyFB.newField[OutputBuffer]

          assert(cRetPtype == ctxTypeTuple)
          assert(gRetPtype == gTypeTuple)

          val env = Env[(TypeInfo[_], Code[Boolean], Code[_])](
            (cname, (typeToTypeInfo(ctxType), bodyMB.getArg[Boolean](3).load(), bodyMB.getArg(2)(typeToTypeInfo(ctxType)).load())),
            (gname, (typeToTypeInfo(gType), bodyMB.getArg[Boolean](5).load(), bodyMB.getArg(4)(typeToTypeInfo(gType)).load())))

          // FIXME fix number of aggs here
          val t = new Emit(ctx, bodyMB).emit(MakeTuple.ordered(FastSeq(body)), env, EmitRegion.default(bodyMB), None)
          bodyMB.emit(Code(t.setup, t.m.mux(Code._fatal("return cannot be missing"), t.v)))

          val ctxIS = Code.newInstance[ByteArrayInputStream, Array[Byte]](bodyFB.getArg[Array[Byte]](2))
          val gIS = Code.newInstance[ByteArrayInputStream, Array[Byte]](bodyFB.getArg[Array[Byte]](3))

          val ctxOff = bodyFB.newLocal[Long]
          val gOff = bodyFB.newLocal[Long]
          val bOff = bodyFB.newLocal[Long]
          val bOS = bodyFB.newLocal[ByteArrayOutputStream]

          bodyFB.emit(Code(
            ctxOff := cDec(bodyFB.getArg[Region](1), cCodec.buildCodeInputBuffer(ctxIS)),
            gOff := gDec(bodyFB.getArg[Region](1), gCodec.buildCodeInputBuffer(gIS)),
            bOff := bodyMB.invoke[Long](bodyFB.getArg[Region](1),
              Region.loadIRIntermediate(ctxType)(ctxTypeTuple.fieldOffset(ctxOff, 0)),
              ctxTypeTuple.isFieldMissing(ctxOff, 0),
              Region.loadIRIntermediate(gType)(gTypeTuple.fieldOffset(gOff, 0)),
              gTypeTuple.isFieldMissing(gOff, 0)),
            bOS := Code.newInstance[ByteArrayOutputStream](),
            bOB := bCodec.buildCodeOutputBuffer(bOS),
            bEnc(bodyFB.getArg[Region](1), bOff, bOB),
            bOB.invoke[Unit]("flush"),
            bOB.invoke[Unit]("close"),
            bOS.invoke[Array[Byte]]("toByteArray")))

          val fID = genUID()
          parentFB.addModule(fID, bodyFB.resultWithIndex())
          fID
        }

        val spark = parentFB.backend()
        val contextAE = emitArrayIterator(contexts)
        val globalsT = emit(globals)

        val cEnc = cCodec.buildEmitEncoderF[Long](ctxTypeTuple, parentFB)
        val gEnc = gCodec.buildEmitEncoderF[Long](gTypeTuple, parentFB)
        val (bRetPType, bDec) = bCodec.buildEmitDecoderF[Long](bTypeTuple.virtualType, parentFB)

        assert(bRetPType == bTypeTuple)

        val baos = mb.newField[ByteArrayOutputStream]
        val buf = mb.newField[OutputBuffer]
        val ctxab = mb.newField[ByteArrayArrayBuilder]
        val encRes = mb.newField[Array[Array[Byte]]]

        val contextT = {
          val sctxb = new StagedRegionValueBuilder(mb, ctxTypeTuple)
          contextAE.arrayEmitter { (m: Code[Boolean], v: Code[_]) =>
            Code(
              baos.invoke[Unit]("reset"),
              sctxb.start(),
              m.mux(
                sctxb.setMissing(),
                sctxb.addIRIntermediate(ctxType)(v)),
              cEnc(region, sctxb.offset, buf),
              buf.invoke[Unit]("flush"),
              ctxab.invoke[Array[Byte], Unit]("add", baos.invoke[Array[Byte]]("toByteArray")))
          }
        }

        val addGlobals = {
          val sgb = new StagedRegionValueBuilder(mb, gTypeTuple)
          Code(
            globalsT.setup,
            sgb.start(),
            globalsT.m.mux(
              sgb.setMissing(),
              sgb.addIRIntermediate(gType)(globalsT.v)),
            gEnc(region, sgb.offset, buf),
            buf.invoke[Unit]("flush"))
        }

        val decodeResult = {
          val sab = new StagedRegionValueBuilder(mb, x.pType)
          val bais = Code.newInstance[ByteArrayInputStream, Array[Byte]](encRes(sab.arrayIdx))
          val eltTupled = mb.newField[Long]
          Code(
            sab.start(encRes.length()),
            Code.whileLoop(sab.arrayIdx < encRes.length(),
              eltTupled := bDec(region, bCodec.buildCodeInputBuffer(bais)),
              bTypeTuple.isFieldMissing(eltTupled, 0).mux(
                sab.setMissing(),
                sab.addIRIntermediate(bType)(Region.loadIRIntermediate(bType)(bTypeTuple.fieldOffset(eltTupled, 0)))),
              sab.advance()),
            sab.end())
        }

        EmitTriplet(
          contextT.setup,
          contextT.m.getOrElse(false),
          Code(
            baos := Code.newInstance[ByteArrayOutputStream](),
            buf := cCodec.buildCodeOutputBuffer(baos), // TODO: take a closer look at whether we need two codec buffers?
            ctxab := Code.newInstance[ByteArrayArrayBuilder, Int](16),
            contextAE.calcLength,
            contextT.addElements,
            baos.invoke[Unit]("reset"),
            addGlobals,
            encRes := spark.invoke[String, Array[Array[Byte]], Array[Byte], Array[Array[Byte]]](
              "collectDArray", functionID,
              ctxab.invoke[Array[Array[Byte]]]("result"),
              baos.invoke[Array[Byte]]("toByteArray")),
            decodeResult))
      case x@TailLoop(name, args, body) =>
        val loopRefs = args.map { case (name, ir) =>
          val ti = typeToTypeInfo(ir.typ)
          ti -> LoopRef(mb.newField[Boolean], mb.newField()(ti), mb.newLocal[Boolean], mb.newLocal()(ti))
        }

        val storeInitArgs = args.zip(loopRefs).map { case ((_, ir), (_, loopref)) =>
          val t = emit(ir)
          Code(t.setup, loopref.m := t.m, (!loopref.m).orEmpty(loopref.v := t.value))
        }

        val label = new CodeLabel
        val m = mb.newField[Boolean]
        val v = mb.newField()(typeToTypeInfo(x.typ))

        val argEnv = env
          .bind(args.zip(loopRefs).map { case ((name, _), (ti, ref)) => (name, (ti, ref.m.load(), ref.v.load())) } : _*)
          .bind(name, (typeToTypeInfo(x.typ), const(false), label.goto))

        val newLoopEnv = loopEnv.getOrElse(Env.empty)
        val bodyT = emit(body, argEnv, loopEnv = Some(newLoopEnv.bind(name, loopRefs.map(_._2).toArray)))
        val bodyF = Code(
          bodyT.setup,
          m := bodyT.m,
          (!m).orEmpty(v := bodyT.value))

        EmitTriplet(Code(storeInitArgs, label, bodyF), m, v)

      case Recur(name, args, _) =>
        val (_, _, jump) = env.lookup(name)
        val refs = loopEnv.get.lookup(name)
        val storeTempArgs = Array.tabulate(refs.length) { i =>
          val t = emit(args(i), env.delete(name), loopEnv = None)
          Code(t.setup, refs(i).tempM := t.m, refs(i).tempV.storeAny(refs(i).tempM.mux(defaultValue(args(i).typ), t.v)))
        }
        val moveArgs = refs.map( ref =>  Code(ref.m := ref.tempM, ref.v.storeAny(ref.tempV)) )
        EmitTriplet(Code(Code(storeTempArgs ++ moveArgs: _*), coerce[Unit](jump)), const(false), Code._empty)
    }
  }

  private def capturedReferences(ir: IR): (IR, (Emit.E, DependentEmitFunction[_]) => Emit.E) = {
    var ids = Set[String]()

    def getReferenced: IR => IR = {
      case node@Ref(id, typ) =>
        ids += id
        node
      case node => MapIR(getReferenced)(node)
    }

    (getReferenced(ir), { (env: Emit.E, f: DependentEmitFunction[_]) =>
      Env[(TypeInfo[_], Code[Boolean], Code[_])](ids.toFastSeq.flatMap { id: String =>
         env.lookupOption(id).map { e =>
           val (ti, m, v) = e
           (id, (ti, f.addField[Boolean](m).load(), f.addField(v)(ti.asInstanceOf[TypeInfo[Any]]).load()))
        }
      }: _*)
    })
  }

  private def makeDependentSortingFunction[T: TypeInfo](
    ir: IR, env: Emit.E): DependentEmitFunction[AsmFunction2[T, T, Boolean]] = {
    val (newIR, getEnv) = capturedReferences(ir)
    val f = mb.fb.newDependentFunction[T, T, Boolean]
    val fregion = f.addField[Region](region)
    val newEnv = getEnv(env, f)

    val sort = f.newMethod[Region, T, Boolean, T, Boolean, Boolean]
    val EmitTriplet(setup, m, v) = new Emit(ctx, sort).emit(newIR, newEnv, EmitRegion.default(sort), None)

    sort.emit(Code(setup, m.mux(Code._fatal("Result of sorting function cannot be missing."), v)))
    f.apply_method.emit(Code(sort.invoke(fregion, f.getArg[T](1), false, f.getArg[T](2), false)))
    f
  }

  private def emitArrayIterator(ir: IR, env: E, er: EmitRegion, container: Option[AggContainer]): ArrayIteratorTriplet =
    EmitStream(this, Streamify(ir), env, er, container)
      .toArrayIterator(mb)

  private def present(x: Code[_]): EmitTriplet =
    EmitTriplet(Code._empty, const(false), x)

  private def void(x: Code[Unit]*): EmitTriplet = EmitTriplet(coerce[Unit](Code(x: _*)), false, Code._empty)

  private def strict(value: Code[_], args: EmitTriplet*): EmitTriplet = {
    EmitTriplet(
      coerce[Unit](Code(args.map(_.setup): _*)),
      if (args.isEmpty) false else args.map(_.m).reduce(_ || _),
      value)
  }

  private[ir] def normalArgument(idx: Int, pType: PType): EmitTriplet = {
    val i = 2 + idx * 2
    EmitTriplet(Code._empty,
      mb.getArg[Boolean](i + 1),
      mb.getArg(i)(typeToTypeInfo(pType)))
  }

  def deforestNDArray(er: EmitRegion, x: IR, env: Emit.E): NDArrayEmitter = {
    def deforest(nd: IR): NDArrayEmitter = deforestNDArray(er, nd, env)

    val xType = coerce[PNDArray](x.pType)
    val nDims = xType.nDims

    x match {
      case NDArrayMap(child, elemName, body) =>
        val childP = child.pType.asInstanceOf[PNDArray]
        val elemPType = childP.elementType
        val vti = typeToTypeInfo(elemPType.virtualType)
        val elemRef = coerce[Any](mb.newField(elemName)(vti))
        val bodyEnv = env.bind(name=elemName, v=(vti, false, elemRef.load()))
        val bodyt = this.emit(body, bodyEnv, er, None)

        val childEmitter = deforest(child)
        val setup = Code(childEmitter.setupShape)

        new NDArrayEmitter(mb, childEmitter.nDims, childEmitter.outputShape,
          childP.shape.pType, body.pType, setup, childEmitter.setupMissing, childEmitter.missing) {
          override def outputElement(idxVars: Array[Code[Long]]): Code[_] = {
            Code(
              elemRef := childEmitter.outputElement(idxVars),
              bodyt.setup,
              bodyt.m.orEmpty(Code._fatal("NDArray map body cannot be missing")),
              bodyt.v
            )
          }
        }
      case NDArrayMap2(lChild, rChild, lName, rName, body) =>
        val lP = coerce[PNDArray](lChild.pType)
        val rP = coerce[PNDArray](rChild.pType)

        val lVti = typeToTypeInfo(lP.elementType.virtualType)
        val rVti = typeToTypeInfo(rP.elementType.virtualType)

        val lElemRef = coerce[Any](mb.newField(lName)(lVti))
        val rElemRef = coerce[Any](mb.newField(rName)(rVti))

        val bodyEnv = env.bind(name=lName, v=(lVti, false, lElemRef.load()))
                         .bind(name=rName, v=(rVti, false, rElemRef.load()))
        val bodyt = this.emit(body, bodyEnv, er, None)

        val leftChildEmitter = deforest(lChild)
        val rightChildEmitter = deforest(rChild)

        val shapeArray = NDArrayEmitter.unifyShapes2(leftChildEmitter.outputShape, rightChildEmitter.outputShape)

        val setupMissing = Code(leftChildEmitter.setupMissing, rightChildEmitter.setupMissing)
        val setupShape = Code(leftChildEmitter.setupShape, rightChildEmitter.setupShape)

        new NDArrayEmitter(mb, lP.shape.pType.size, shapeArray, lP.shape.pType, body.pType, setupShape, setupMissing, leftChildEmitter.missing || rightChildEmitter.missing) {
          override def outputElement(idxVars: Array[Code[Long]]): Code[_] = {

            val lIdxVars2 = NDArrayEmitter.zeroBroadcastedDims2(mb, idxVars, nDims, leftChildEmitter.outputShape)
            val rIdxVars2 = NDArrayEmitter.zeroBroadcastedDims2(mb, idxVars, nDims, rightChildEmitter.outputShape)

            Code(
              lElemRef := leftChildEmitter.outputElement(lIdxVars2),
              rElemRef := rightChildEmitter.outputElement(rIdxVars2),
              bodyt.setup,
              bodyt.m.orEmpty(Code._fatal("NDArray map body cannot be missing")),
              bodyt.v
            )
          }
        }

      case x@NDArrayReindex(child, indexExpr) =>
        val childEmitter = deforest(child)
        val childPType = child.pType.asInstanceOf[PNDArray]

        val outputPType = x.pType
        val outputShapePType = outputPType.shape.pType


        val shapeSeq = indexExpr.map {childIndex =>
          if (childIndex < childPType.nDims) {
            childEmitter.outputShape(childIndex)
          }
          else {
            const(1L)
          }
        }.toArray

        new NDArrayEmitter(mb, indexExpr.length, shapeSeq, outputShapePType, outputPType.elementType, childEmitter.setupShape, childEmitter.setupMissing, childEmitter.missing) {
          override def outputElement(idxVars: Array[Code[Long]]): Code[_] = {
            val concreteIdxsForChild = Array.tabulate(childEmitter.nDims) { childDim =>
              val parentDim = indexExpr.indexOf(childDim)
              idxVars(parentDim)
            }
            childEmitter.outputElement(concreteIdxsForChild)
          }
        }

      case x@NDArrayReshape(childND, shape) =>

        // Need to take this shape, which may have a -1 in it, and turn it into a compatible shape if possible.
        def compatibleShape(numElements: Code[Long], requestedShape: Array[Code[Long]]): (Code[Unit], Array[Code[Long]]) = {
          val hasNegativeOne = mb.newLocal[Boolean]
          val runningProduct = mb.newLocal[Long]
          val quotient = mb.newLocal[Long]
          val tempShapeElement = mb.newLocal[Long]

          val newShapeVars = (0 until requestedShape.length).map(_ => mb.newField[Long]).toArray

          val setupShape = coerce[Unit](Code(
            hasNegativeOne := false,
            runningProduct := 1L,

            Code.foreach(requestedShape) { requestedShapeElement => Code(
              tempShapeElement := requestedShapeElement,
              (tempShapeElement <= 0L).mux(
                (tempShapeElement ceq -1L).mux(
                  hasNegativeOne.mux(
                    Code._fatal("Can't infer shape, more than one -1"),
                    hasNegativeOne := true
                  ),
                  Code._fatal("Can't reshape, new shape must contain only positive numbers or -1")),
                runningProduct := runningProduct * tempShapeElement
              )
            )},
            hasNegativeOne.mux(
              (numElements % runningProduct) > 0L,
              numElements cne runningProduct
            ).orEmpty(Code._fatal("Can't reshape since requested shape is incompatible with number of elements")),
            quotient := numElements / runningProduct,
            Code(newShapeVars.zip(requestedShape).map { case (variable, shapeElement) =>
              variable := (shapeElement ceq -1L).mux(quotient, shapeElement)}:_*)
          ))

          (setupShape, newShapeVars.map(_.load()))
        }

        val childEmitter = deforest(childND)

        val requestedShapet = emit(shape, env, resultRegion, None)
        val requestedShapeAddress = mb.newField[Long]
        val requestedShapePType = coerce[PTuple](shape.pType)
        val requestedShapeTuple = new CodePTuple(requestedShapePType, requestedShapeAddress)
        val requestedShapeArray = (0 until requestedShapePType.size).map(i => requestedShapeTuple[Long](i)).toArray

        val (childShapeCachingCode, childShapeCached) = childEmitter.outputShape.cacheEntries(mb, LongInfo)

        val numElements = mb.newField[Long]

        val (reshapeSetup, reshapedShapeArray) = compatibleShape(numElements, requestedShapeArray)

        val setupMissing = Code(
          childEmitter.setupMissing,
          requestedShapet.setup
        )

        val setupShape = Code(
          childEmitter.setupShape,
          childShapeCachingCode,
          requestedShapeAddress := requestedShapet.value[Long],
          numElements := coerce[PNDArray](childND.pType).numElements(childShapeCached, mb),
          reshapeSetup
        )

        new NDArrayEmitter(mb, reshapedShapeArray.length, reshapedShapeArray, requestedShapePType.setRequired(true).asInstanceOf[PTuple],
          childEmitter.outputElementPType, setupShape, setupMissing, childEmitter.missing || requestedShapet.m) {
          override def outputElement(idxVars: Array[Code[Long]]): Code[_] = {
            val storeElementIndex = mb.newField[Long]

            val (newIdxVarsSetup, newIdxVars) = x.pType.unlinearizeIndexRowMajor(storeElementIndex, childShapeCached, mb)

            assert(newIdxVars.length == childEmitter.nDims)

            Code(
              storeElementIndex := x.pType.linearizeIndicesRowMajor(idxVars, reshapedShapeArray, mb),
              newIdxVarsSetup,
              childEmitter.outputElement(newIdxVars)
            )
          }
        }

      case x@NDArrayConcat(nds, axis) =>
        val inputType = coerce[PArray](nds.pType2)
        val inputNDType = coerce[PNDArray](inputType.elementType)

        val ndType = coerce[PNDArray](x.pType2)
        val codeNDs = emit(nds, env, er, None)

        val inputArray = mb.newField[Long]
        val n = mb.newField[Int]
        val i = mb.newField[Int]

        val loadAndValidateArray = Code(
          inputArray := codeNDs.value[Long],
          n := inputType.loadLength(inputArray),
          (n < 1).orEmpty(Code._fatal("NDArrayConcat: can't concatenate 0 NDArrays")))

        val (missingSetup: Code[Unit @unchecked], missing: Code[Boolean @unchecked], setupShape: Code[Unit @unchecked]) = (inputType.required, inputNDType.required) match {
          case (true, true) => (Code._empty, const(false), Code(
            codeNDs.setup,
            codeNDs.m.orEmpty(Code._fatal("NDArrayConcat: required NDArray can't be missing")),
            loadAndValidateArray))
          case (false, true) => (codeNDs.setup, codeNDs.m, loadAndValidateArray)
          case _ =>
            val m = mb.newField[Boolean]
            val setup = Code(
              codeNDs.setup,
              m := codeNDs.m,
              loadAndValidateArray,
              i := 0,
              Code.whileLoop(i < n,
                m := m | inputType.isElementMissing(inputArray, i),
                i := i + 1))
            (setup, m.load(), Code._empty)
        }

        val localDim = mb.newField[Long]
        val outputShape = Array.tabulate(ndType.nDims) { idx =>
          Code(
            localDim := inputNDType.dimensionLength(inputType.loadElement(inputArray, 0), idx),
            i := 1,
            Code.whileLoop(i < n,
              {
                if (idx == axis)
                  localDim := localDim + inputNDType.dimensionLength(inputType.loadElement(inputArray, i), idx)
                else
                  inputNDType.dimensionLength(inputType.loadElement(inputArray, i), idx).cne(localDim)
                    .orEmpty(Code._fatal(
                      const(s"NDArrayConcat: mismatched dimensions of input NDArrays along axis $i: expected ")
                        .concat(localDim.toS).concat(", got ")
                        .concat(inputNDType.dimensionLength(inputType.loadElement(inputArray, i), idx).toS)))
              },
              i := i + 1),
            localDim)
        }

        new NDArrayEmitter(mb, x.typ.nDims,
          outputShape,
          ndType.shape.pType,
          ndType.elementType,
          setupShape,
          missingSetup,
          missing) {
          private val concatAxisIdx = mb.newLocal[Long]

          override def outputElement(idxVars: Array[Code[Long]]): Code[_] = {
            val setupTransformedIdx = Code(
              i := 0,
              concatAxisIdx := idxVars(axis),
              Code.whileLoop(concatAxisIdx >= inputNDType.dimensionLength(inputType.loadElement(inputArray, i), axis),
                concatAxisIdx := concatAxisIdx - inputNDType.dimensionLength(inputType.loadElement(inputArray, i), axis),
                i := i + 1
              ),
              (i > n).orEmpty(Code._fatal("NDArrayConcat: trying to access element greater than length of concatenation axis")))

            val transformedIdxs = Array.tabulate(x.typ.nDims) { idx =>
              if (idx == axis) concatAxisIdx.load() else idxVars(idx)
            }
            Code(
              setupTransformedIdx,
              Region.loadIRIntermediate(ndType.elementType)(
                inputNDType.getElementAddress(transformedIdxs, inputType.loadElement(inputArray, i), mb)))
          }
        }


      case x@NDArraySlice(child, slicesIR) =>
        val childEmitter = deforest(child)

        val slicest = emit(slicesIR, env, resultRegion, None)
        val slicesValueAddress = mb.newField[Long]
        val slices = new CodePTuple(coerce[PTuple](slicesIR.pType), slicesValueAddress)

        val slicers = slices.withTypes.collect {
          case (t: PTuple, slice) => new CodePTuple(t, slice)
        }

        val missingSliceElements = slicers.map(_.missingnessPattern.reduce(_ || _)).fold(const(false))(_ || _)
        val anyMissingness = missingSliceElements || slices.missingnessPattern.fold(const(false))(_ || _)

        val codeSlices = slicers.map(_.values[Long, Long, Long])

        val outputShape = codeSlices.map { case (start, stop, step) =>
          (step >= 0L && start <= stop).mux(
            const(1L) + ((stop - start) - 1L) / step,
            (step < 0L && start >= stop).mux(
              (((stop - start) + 1L) / step) + 1L,
              0L)
          )
        }.toArray

        val setupMissing = Code(
          slicest.setup,
          slicesValueAddress := slicest.value[Long],
          childEmitter.setupMissing
        )

        val missing = childEmitter.missing || anyMissingness

        new NDArrayEmitter(mb, x.pType.nDims, outputShape, x.pType.shape.pType, x.pType.elementType, childEmitter.setupShape, setupMissing, missing) {
          override def outputElement(idxVars: Array[Code[Long]]): Code[_] = {
            val oldIdxVarsIter = idxVars.iterator

            val sliceIdxVars2 = slices.withTypes.map {
              case (_: PInt64, indexer) =>
                coerce[Long](indexer)
              case (t: PTuple, slicer) =>
                val (start, _, step) = new CodePTuple(t, slicer).values[Long, Long, Long]
                start + oldIdxVarsIter.next() * step
            }

            childEmitter.outputElement(sliceIdxVars2.toArray)
          }
        }

      case _ =>
        val ndt = emit(x, env, er, None)
        val ndAddress = mb.newField[Long]
        val setup = Code(
          ndAddress := ndt.value[Long]
        )
        val xP = x.pType.asInstanceOf[PNDArray]

        val shapeAddress = xP.shape.load(ndAddress)
        val shapeTuple = new CodePTuple(xP.shape.pType, shapeAddress)

        val shapeArray = (0 until xP.shape.pType.nFields).map(i => shapeTuple.apply[Long](i)).toArray

        new NDArrayEmitter(mb, nDims, shapeArray,
          xP.shape.pType, xP.elementType, setup, ndt.setup, ndt.m) {
          override def outputElement(idxVars: Array[Code[Long]]): Code[_] = {
            val elementLocation = xP.getElementAddress(idxVars, ndAddress, mb)
            Region.loadIRIntermediate(outputElementPType)(elementLocation)
          }
        }
    }
  }
}

object NDArrayEmitter {

  def zeroBroadcastedDims2(mb: MethodBuilder, loopVars: Array[Code[Long]], nDims: Int, shapeArray: Array[Code[Long]]): Array[Code[Long]] = {
    val broadcasted = 0L
    val notBroadcasted = 1L
    Array.tabulate(nDims)(dim => (shapeArray(dim) > 1L).mux(notBroadcasted, broadcasted) * loopVars(dim))
  }

  def broadcastMask(shapeArray: Array[Code[Long]]): Array[Code[Long]] = {
    val broadcasted = 0L
    val notBroadcasted = 1L
    shapeArray.map(shapeElement => (shapeElement > 1L).mux(notBroadcasted, broadcasted))
  }

  def zeroBroadcastedDims(indices: Array[Code[Long]], broadcastMask: Array[Code[Long]]): Array[Code[Long]] = {
    indices.zip(broadcastMask).map { case (index, flag) => index * flag }
  }

  def unifyShapes2(leftShape: Array[Code[Long]], rightShape: Array[Code[Long]]): Array[Code[Long]] = {
    leftShape.zip(rightShape).map{case (left, right) =>
      val notSameAndNotBroadcastable = !((left ceq right) || (left ceq 1L) || (right ceq 1L))
      coerce[Long](Code(
        notSameAndNotBroadcastable.mux(
          Code._fatal("Incompatible NDArray shapes"),
          (left > right).mux(left, right)
        )
      ))
    }
  }

  def matmulShape(leftShape: Array[Code[Long]], rightShape: Array[Code[Long]]): (Code[Unit], Array[Code[Long]]) = {
    val ((lK, rK), shape) = (leftShape.toSeq, rightShape.toSeq) match {
      case (Seq(l), Seq(r)) =>
        ((l, r), Array[Code[Long]]())
      case (Seq(l), rs :+ r2 :+ r1) =>
        ((l, r2), (rs :+ r1).toArray)
      case (ls :+ l2 :+ l1, Seq(r)) =>
        ((l1, r), (ls :+ l2).toArray)
      case (
        ls :+ l2 :+ l1,
        rs :+ r2 :+ r1
        ) => ((l1, r2), unifyShapes2(ls.toArray, rs.toArray) :+ l2 :+ r1)
      case (l, r) =>
        fatal(s"Matrix multiply compiler bug: $l $r")
    }

    val dimCheck = (lK cne rK).orEmpty(
      Code._fatal(const("Matrix dimensions incompatible: ").concat(lK.toS).concat(" ").concat(rK.toS)))

    (dimCheck, shape)
  }
}

abstract class NDArrayEmitter(
   val mb: MethodBuilder,
   val nDims: Int,
   val outputShape: Array[Code[Long]],
   val outputShapePType: PTuple,
   val outputElementPType: PType,
   val setupShape: Code[_],
   val setupMissing: Code[Unit] = Code._empty[Unit],
   val missing: Code[Boolean] = false) {

  private val outputShapeVariables = (0 until nDims).map(_ => mb.newField[Long]).toArray

  def outputElement(idxVars: Array[Code[Long]]): Code[_]

  def emit(targetType: PNDArray): EmitTriplet = {
    val dataSrvb = new StagedRegionValueBuilder(mb, targetType.data.pType)

    val dataAddress: Code[Long] =
      Code(
        dataSrvb.start(targetType.numElements(outputShapeVariables.map(_.load()), mb).toI),
        emitLoops(dataSrvb),
        dataSrvb.end()
      )

    def shapeBuilder(srvb: StagedRegionValueBuilder): Code[Unit] = {
      coerce[Unit](Code(
        srvb.start(),
        Code.foreach(outputShapeVariables){ shapeElement =>
          Code(
            srvb.addLong(shapeElement),
            srvb.advance()
          )
        }
      ))
    }

    val fullSetup = Code(
      setupMissing,
      missing.mux(
        Code._empty,
        Code(
          setupShape,
          Code.foreach(0 until nDims)(index => outputShapeVariables(index) := outputShape(index))
        )
      )
    )

    EmitTriplet(fullSetup, missing, targetType.construct(0, 0, shapeBuilder, targetType.makeDefaultStridesBuilder(outputShapeVariables.map(_.load()), mb), dataAddress, mb))
  }

  private def emitLoops(srvb: StagedRegionValueBuilder): Code[_] = {
    val idxVars = Array.tabulate(nDims) {_ => mb.newField[Long]}
    val loadedIdxVars = idxVars.map(_.load())
    val storeElement = mb.newLocal(typeToTypeInfo(outputElementPType.virtualType)).asInstanceOf[LocalRef[Double]]
    val body =
      Code(
        storeElement := outputElement(loadedIdxVars).asInstanceOf[Code[Double]],
        srvb.addIRIntermediate(outputElementPType)(storeElement),
        srvb.advance()
      )
    val loops = idxVars.zipWithIndex.foldRight(body) { case((dimVar, dimIdx), innerLoops) =>
      Code(
        dimVar := 0L,
        Code.whileLoop(dimVar < outputShapeVariables(dimIdx),
          innerLoops,
          dimVar := dimVar + 1L
        )
      )
    }
    val eVti = typeToTypeInfo(TVoid)
    val innerMethod = mb.fb.newMethod(eVti)
    innerMethod.emit(loops)
    innerMethod.invoke()
  }
}
