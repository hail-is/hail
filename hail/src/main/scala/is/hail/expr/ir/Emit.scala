package is.hail.expr.ir

import java.io.{ByteArrayInputStream, ByteArrayOutputStream}

import is.hail.annotations._
import is.hail.asm4s.joinpoint.Ctrl
import is.hail.asm4s._
import is.hail.backend.HailTaskContext
import is.hail.expr.ir.functions.StringFunctions
import is.hail.expr.types.physical._
import is.hail.expr.types.virtual._
import is.hail.io.{BufferSpec, InputBuffer, OutputBuffer, TypedCodecSpec}
import is.hail.linalg.{BLAS, LAPACK, LinalgCodeUtils}
import is.hail.lir
import is.hail.utils._

import scala.collection.mutable
import scala.language.{existentials, postfixOps}

object SetupBuilder {
  def apply(mb: EmitMethodBuilder[_]): SetupBuilder = new SetupBuilder(mb, Code._empty)

  def apply(mb: EmitMethodBuilder[_], setup: Code[Unit]): SetupBuilder = new SetupBuilder(mb, setup)

  def map[T, U](mb: EmitMethodBuilder[_])(is: IndexedSeq[T])(f: (SetupBuilder, T) => U): (Code[Unit], IndexedSeq[U]) = {
    val sb = SetupBuilder(mb)
    val rs = sb.map(is)(f)
    (sb.setup, rs)
  }

  def map[T, U](mb: EmitMethodBuilder[_], setup: Code[Unit])(is: IndexedSeq[T])(f: (SetupBuilder, T) => U): (Code[Unit], IndexedSeq[U]) = {
    val sb = SetupBuilder(mb, setup)
    val rs = sb.map(is)(f)
    (sb.setup, rs)
  }
}

class SetupBuilder(mb: EmitMethodBuilder[_], var setup: Code[Unit]) {
  def append(c: Code[Unit]): Unit = {
    setup = Code(setup, c)
  }

  def +=(c: Code[Unit]): Unit = append(c)

  def memoize[T](e: Code[T], name: String)(implicit tti: TypeInfo[T]): Value[T] = {
    val l = mb.newLocal[T](name)
    append(l := e)
    l
  }

  def memoizeField[T](e: Code[T], name: String)(implicit tti: TypeInfo[T]): Value[T] = {
    val l = mb.genFieldThisRef[T](name)
    append(l := e)
    l
  }

  def map[T, U](is: IndexedSeq[T])(f: (SetupBuilder, T) => U): IndexedSeq[U] = is.map(f(this, _))

  def result(): Code[Unit] = {
    val r = setup
    setup = null
    r
  }
}


object Emit {
  type E = Env[EmitValue]

  def apply[C](ctx: ExecuteContext, ir: IR, fb: EmitFunctionBuilder[C], aggs: Option[Array[AggStatePhysicalSignature]] = None) {
    val triplet = emit[C](ctx, ir, fb, Env.empty, aggs)
    typeToTypeInfo(ir.typ) match {
      case ti: TypeInfo[t] =>
        fb.emit(Code(triplet.setup, triplet.m.mux(
          Code._throw[RuntimeException, t](Code.newInstance[RuntimeException, String]("cannot return empty"))(ti),
          coerce[t](triplet.v))))
    }
  }

  private def emit[C](
    ctx: ExecuteContext,
    ir: IR,
    fb: EmitFunctionBuilder[C],
    env: E,
    aggs: Option[Array[AggStatePhysicalSignature]]): EmitCode = {
    TypeCheck(ir)
    val container = aggs.map { a =>
      val c = fb.addAggStates(a)
      AggContainer(a, c)
    }

    new Emit[C](ctx, fb.ecb)
      .emit(ir, fb.apply_method, env, EmitRegion.default(fb.apply_method), container = container)
  }
}

object AggContainer {
  def fromVars(aggs: Array[AggStatePhysicalSignature], cb: EmitClassBuilder[_], region: Settable[Region], off: Settable[Long]): (AggContainer, Code[Unit], Code[Unit]) = {
    val states = agg.StateTuple(aggs.map(a => agg.Extract.getAgg(a, a.default).createState(cb)).toArray)
    val aggState = new agg.TupleAggregatorState(cb, states, region, off)

    val setup = Code(
      region := Region.stagedCreate(Region.REGULAR),
      region.load().setNumParents(aggs.length),
      off := region.load().allocate(aggState.storageType.alignment, aggState.storageType.byteSize),
      states.createStates(cb))

    val cleanup = Code(
      region.load().invalidate(),
      region := Code._null)

    (AggContainer(aggs, aggState), setup, cleanup)
  }

  def fromClassBuilder[C](aggs: Array[AggStatePhysicalSignature], cb: EmitClassBuilder[C], varPrefix: String): (AggContainer, Code[Unit], Code[Unit]) =
    fromVars(aggs, cb, cb.genFieldThisRef[Region](s"${varPrefix}_top_region"), cb.genFieldThisRef[Long](s"${varPrefix}_off"))
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
  def default(mb: EmitMethodBuilder[_]): EmitRegion = EmitRegion(mb, mb.getArg[Region](1))
}

case class EmitRegion(mb: EmitMethodBuilder[_], region: Value[Region]) {
  def baseRegion: Value[Region] = mb.getArg[Region](1)
}

abstract class EmitValue {
  def pt: PType

  def get: EmitCode
}

object IEmitCode {
  def apply(cb: EmitCodeBuilder, m: Code[Boolean], pc: PCode): IEmitCode = {
    val Lmissing = CodeLabel()
    val Lpresent = CodeLabel()
    cb.ifx(m, { cb.goto(Lmissing) }, { cb.goto(Lpresent) })
    IEmitCode(Lmissing, Lpresent, pc)
  }
}

case class IEmitCode(Lmissing: CodeLabel, Lpresent: CodeLabel, pc: PCode) {
  def map(f: (PCode) => PCode): IEmitCode = {
    IEmitCode(Lmissing, Lpresent, f(pc))
  }

  def flatMap(cb: EmitCodeBuilder)(f: (PCode) => IEmitCode): IEmitCode = {
    cb.define(Lpresent)
    val ec2 = f(pc)
    cb.define(ec2.Lmissing)
    cb.goto(Lmissing)
    IEmitCode(Lmissing, ec2.Lpresent, ec2.pc)
  }
}

object EmitCode {
  def apply(setup: Code[Unit], ec: EmitCode): EmitCode =
    new EmitCode(Code(setup, ec.setup), ec.m, ec.pv)

  def present(pt: PType, v: Code[_]): EmitCode = EmitCode(Code._empty, false, PCode(pt, v))

  def missing(pt: PType): EmitCode = EmitCode(Code._empty, true, pt.defaultValue)

  def fromI(mb: EmitMethodBuilder[_])(f: (EmitCodeBuilder) => IEmitCode): EmitCode = {
    val cb = EmitCodeBuilder(mb)
    val iec = f(cb)
    val setup = cb.result()
    val newEC = EmitCode(Code._empty,
      new CCode(setup.start, iec.Lmissing.start, iec.Lpresent.start),
      iec.pc)
    iec.Lmissing.clear()
    iec.Lpresent.clear()
    setup.clear()
    newEC
  }
}

case class EmitCode(setup: Code[Unit], m: Code[Boolean], pv: PCode) {
  def pt: PType = pv.pt

  def v: Code[_] = pv.code

  def value[T]: Code[T] = coerce[T](v)

  def map(f: PCode => PCode): EmitCode = new EmitCode(setup, m, pv = f(pv))

  def toI(cb: EmitCodeBuilder): IEmitCode = {
    val Lmissing = CodeLabel()
    val Lpresent = CodeLabel()
    cb += setup
    cb.ifx(m, { cb.goto(Lmissing) }, { cb.goto(Lpresent) })
    IEmitCode(Lmissing, Lpresent, pv)
  }

  def castTo(mb: EmitMethodBuilder[_], region: Value[Region], destType: PType): EmitCode = {
    EmitCode(
      setup,
      m,
      pv.castTo(mb, region, destType))
  }
}

abstract class EmitSettable extends EmitValue {
  def store(ec: EmitCode): Code[Unit]

  def load(): EmitCode = get

  def :=(ec: EmitCode): Code[Unit] = store(ec)
}

abstract class PresentEmitSettable extends EmitValue {
  def store(pc: PCode): Code[Unit]

  def load(): EmitCode = get

  def :=(pc: PCode): Code[Unit] = store(pc)
}

class RichIndexedSeqEmitSettable(is: IndexedSeq[EmitSettable]) {
  def :=(ix: IndexedSeq[EmitCode]): Code[Unit] = Code((is, ix).zipped.map(_ := _))

  def load(): IndexedSeq[EmitCode] = is.map(_.load())
}

object LoopRef {
  def apply(mb: EmitMethodBuilder[_], L: CodeLabel, args: IndexedSeq[(String, PType)]): LoopRef = {
    val (loopArgs, tmpLoopArgs) = args.zipWithIndex.map { case ((name, pt), i) =>
      (mb.newEmitField(s"$name$i", pt), mb.newEmitField(s"tmp$name$i", pt))
    }.unzip
    LoopRef(L, loopArgs, tmpLoopArgs)
  }
}

case class LoopRef(
  L: CodeLabel,
  loopArgs: IndexedSeq[EmitSettable],
  tmpLoopArgs: IndexedSeq[EmitSettable])

abstract class EstimableEmitter[C] {
  def emit(mb: EmitMethodBuilder[C]): Code[Unit]

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

  def wrapToMethod[C](items: Seq[EstimableEmitter[C]], mb: EmitMethodBuilder[C]): Code[Unit] = {
    if (items.isEmpty)
      return Code._empty

    val sizes = items.map(_.estimatedSize)
    if (sizes.sum < 100)
      return Code(items.map(_.emit(mb)))

    val chunkBounds = getChunkBounds(sizes)
    assert(chunkBounds(0) == 0 && chunkBounds.last == sizes.length)

    val chunks = chunkBounds.zip(chunkBounds.tail).map { case (start, end) =>
      assert(start < end)
      val newMB = mb.genEmitMethod("wrapped", mb.parameterTypeInfo, typeInfo[Unit])
      val c = items.slice(start, end)
      newMB.emit(Code(c.map(_.emit(newMB))))
      new EstimableEmitter[C] {
        def emit(mb: EmitMethodBuilder[C]): Code[Unit] = {
          val args = mb.parameterTypeInfo.toFastIndexedSeq.zipWithIndex.map { case (ti, i) => mb.getArg(i + 1)(ti).load() }
          coerce[Unit](newMB.invoke(args: _*))
        }

        def estimatedSize: Int = 5
      }
    }
    wrapToMethod(chunks, mb)
  }
}

private class Emit[C](
  val ctx: ExecuteContext,
  val cb: EmitClassBuilder[C]) { emitSelf =>

  val methods: mutable.Map[String, Seq[(Seq[PType], PType, EmitMethodBuilder[C])]] = mutable.Map().withDefaultValue(FastSeq())

  import Emit.E

  private def wrapToMethod(irs: Seq[IR], mb: EmitMethodBuilder[C], env: E, container: Option[AggContainer])(useValues: (EmitMethodBuilder[C], PType, EmitCode) => Code[Unit]): Code[Unit] = {
    val opSize: Int = 20
    val items = irs.map { ir =>
      new EstimableEmitter[C] {
        def estimatedSize: Int = ir.size * opSize

        def emit(mb: EmitMethodBuilder[C]): Code[Unit] =
          // wrapped methods can't contain uses of Recur
          useValues(mb, ir.pType, emitSelf.emit(ir, mb, env, EmitRegion.default(mb), container, None))
      }
    }

    EmitUtils.wrapToMethod(items, mb)
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
  private[ir] def emit(ir: IR, mb: EmitMethodBuilder[C], env: E, er: EmitRegion, container: Option[AggContainer]): EmitCode =
    emit(ir, mb, env, er, container, None)

  private def emit(ir: IR, mb: EmitMethodBuilder[C], env: E, er: EmitRegion, container: Option[AggContainer], loopEnv: Option[Env[LoopRef]]): EmitCode = {
    def emit(ir: IR, mb: EmitMethodBuilder[C] = mb, env: E = env, er: EmitRegion = er, container: Option[AggContainer] = container, loopEnv: Option[Env[LoopRef]] = loopEnv): EmitCode =
      this.emit(ir, mb, env, er, container, loopEnv)

    def wrapToMethod(irs: Seq[IR], mb: EmitMethodBuilder[C] = mb, env: E = env, container: Option[AggContainer] = container)(useValues: (EmitMethodBuilder[C], PType, EmitCode) => Code[Unit]): Code[Unit] =
      this.wrapToMethod(irs, mb, env, container)(useValues)

    def emitStream(ir: IR, mb: EmitMethodBuilder[C] = mb): COption[EmitStream.SizedStream] =
      EmitStream(this, mb, ir, env, er, container)

    def emitDeforestedNDArray(ir: IR, mb: EmitMethodBuilder[C] = mb, er: EmitRegion = er): EmitCode =
      deforestNDArray(mb, er, ir, env).emit(mb, coerce[PNDArray](ir.pType))

    def emitNDArrayStandardStrides(ir: IR): EmitCode =
      // Currently relying on the fact that emitDeforestedNDArray always emits standard striding.
      emitDeforestedNDArray(ir)

    val region = er.region

    val pt = ir.pType

    (ir: @unchecked) match {
      case I32(x) =>
        present(pt, const(x))
      case I64(x) =>
        present(pt, const(x))
      case F32(x) =>
        present(pt, const(x))
      case F64(x) =>
        present(pt, const(x))
      case s@Str(x) =>
        present(pt, mb.addLiteral(x, coerce[PString](s.pType)))
      case x@Literal(t, v) =>
        present(pt, mb.addLiteral(v, x.pType))
      case True() =>
        present(pt, const(true))
      case False() =>
        present(pt, const(false))
      case Void() =>
        EmitCode(Code._empty, const(false), PCode(pt, Code._empty))

      case Cast(v, typ) =>
        val codeV = emit(v)
        val cast = Casts.get(v.typ, typ)
        EmitCode(codeV.setup, codeV.m, PCode(pt, cast(codeV.v)))

      case CastRename(v, _typ) =>
        emit(v)

      case NA(typ) =>
        EmitCode(Code._empty, const(true), pt.defaultValue)
      case IsNA(v) =>
        val codeV = emit(v)
        EmitCode(codeV.setup, const(false), PCode(pt, codeV.m))

      case Coalesce(values) =>
        val mout = mb.newLocal[Boolean]()
        val out = mb.newPLocal(pt)

        def f(i: Int): Code[Unit] = {
          if (i < values.length) {
            val ec = emit(values(i))
            Code(ec.setup,
              ec.m.mux(
                f(i + 1),
                Code(mout := false, out := pt.copyFromPValue(mb, region, ec.pv))))
          } else
            mout := true
        }

        EmitCode(
          setup = f(0),
          m = mout,
          pv = out.get)

      case If(cond, cnsq, altr) =>
        assert(cnsq.typ == altr.typ)

        if (cnsq.typ == TVoid) {
          val codeCond = emit(cond)
          val codeCnsq = emit(cnsq)
          val codeAltr = emit(altr)
          EmitCode(
            Code(
              codeCond.setup,
              codeCond.m.mux(
                Code._empty,
                codeCond.value[Boolean].mux(
                  codeCnsq.setup,
                  codeAltr.setup))),
            false,
            PCode._empty)
        } else {
          val codeCond = emit(cond)
          val out = mb.newPLocal(pt)
          val mout = mb.newLocal[Boolean]()
          val codeCnsq = emit(cnsq)
          val codeAltr = emit(altr)

          val setup = Code(
            codeCond.setup,
            codeCond.m.mux(
              mout := true,
              coerce[Boolean](codeCond.v).mux(
                Code(codeCnsq.setup,
                  mout := codeCnsq.m,
                  mout.mux(
                    Code._empty,
                    out := ir.pType.copyFromPValue(mb, er.region, codeCnsq.pv))),
                Code(codeAltr.setup,
                  mout := codeAltr.m,
                  mout.mux(
                    Code._empty,
                    out := ir.pType.copyFromPValue(mb, er.region, codeAltr.pv))))))

          EmitCode(setup, mout, out.load())
        }

      case Let(name, value, body) =>
        val x = mb.newEmitField(name, value.pType)
        val storeV = wrapToMethod(FastIndexedSeq(value)) { (_, _, ecV) =>
          x := ecV
        }
        val bodyenv = env.bind(name, x)
        val codeBody = emit(body, env = bodyenv)
        val setup = Code(
          storeV,
          codeBody.setup)

        EmitCode(setup, codeBody.m, codeBody.pv)
      case Ref(name, _) =>
        val ev = env.lookup(name)
        if (ev.pt != pt)
          throw new RuntimeException(s"PValue type did not match inferred ptype:\n name: $name\n  pv: ${ ev.pt }\n  ir: $pt")
        ev.get

      case ApplyBinaryPrimOp(op, l, r) =>
        val codeL = emit(l)
        val codeR = emit(r)
        strict(pt, BinaryOp.emit(op, l.typ, r.typ, codeL.v, codeR.v), codeL, codeR)
      case ApplyUnaryPrimOp(op, x) =>
        val v = emit(x)
        strict(pt, UnaryOp.emit(op, x.typ, v.v), v)
      case ApplyComparisonOp(op, l, r) =>
        val f = op.codeOrdering(mb, l.pType, r.pType)
        val codeL = emit(l)
        val codeR = emit(r)
        if (op.strict) {
          strict(pt, f((false, codeL.v), (false, codeR.v)),
            codeL, codeR)
        } else {
          val lm = mb.newLocal[Boolean]()
          val rm = mb.newLocal[Boolean]()
          present(pt, Code(
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

        val addElts = { (newMB: EmitMethodBuilder[C], pt: PType, v: EmitCode) =>
          Code(
            v.setup,
            v.m.mux(srvb.setMissing(), addElement(pType.elementType.copyFromTypeAndStackValue(newMB, er.region, pt, v.v))),
            srvb.advance())
        }
        present(pt, Code(srvb.start(args.size, init = true), wrapToMethod(args)(addElts), srvb.offset))
      case x@ArrayRef(a, i, s) =>
        val errorTransformer: Code[String] => Code[String] = s match {
          case Str("") =>
            val prettied = Pretty.short(x)
            (c: Code[String]) =>
              c.concat("\n----------\nIR:\n").concat(prettied)
          case Str(s) => (c: Code[String]) => c.concat("\n----------\nPython traceback:\n").concat(s)
          case s =>
            val codeS = emit(s)
            (c: Code[String]) =>
              Code.memoize(c, "array_ref_c") { c =>
                Code(codeS.setup,
                  codeS.m.mux(c,
                    c.concat("\n----------\nPython traceback:\n")
                      .concat(s.pType.asInstanceOf[PString].loadString(coerce[Long](codeS.v)))))
              }
        }

        EmitCode.fromI(mb) { cb =>
          emit(a).toI(cb).flatMap(cb) { (ac) =>
            emit(i).toI(cb).flatMap(cb) { (ic) =>
              val av = ac.asIndexable.memoize(cb, "aref_a")
              val iv = cb.memoize(ic.tcode[Int], "aref_i")

              cb.ifx(iv < 0 || iv >= av.loadLength(), {
                cb._fatal(errorTransformer(
                  const("array index out of bounds: index=")
                    .concat(iv.toS)
                    .concat(", length=")
                    .concat(av.loadLength().toS)))
              })
              av.loadElement(cb, iv)
            }
          }
        }

      case ArrayLen(a) =>
        EmitCode.fromI(mb) { cb =>
          emit(a).toI(cb).map { (ac) =>
            PCode(PInt32Required, ac.asIndexable.loadLength())
          }
        }

      case x@(_: ArraySort | _: ToSet | _: ToDict) =>
        val atyp = coerce[PIterable](x.pType)
        val eltType = atyp.elementType
        val eltVType = eltType.virtualType

        val vab = new StagedArrayBuilder(atyp.elementType, mb, 0)
        val sorter = new ArraySorter(er, vab)

        val (array, compare, distinct, leftRightComparatorNames: Array[String]) = (x: @unchecked) match {
          case ArraySort(a, l, r, comp) => (a, comp, Code._empty, Array(l, r))
          case ToSet(a) =>
            val discardNext = mb.genEmitMethod("discardNext", Array[TypeInfo[_]](typeInfo[Region], sorter.ti, typeInfo[Boolean], sorter.ti, typeInfo[Boolean]), typeInfo[Boolean])
            val cmp2 = ApplyComparisonOp(EQWithNA(eltVType), In(0, eltType), In(1, eltType))
            InferPType(cmp2, Env.empty)
            val EmitCode(s, m, pv) = emit(cmp2, discardNext, Env.empty, er, container)
            discardNext.emit(Code(s, m || pv.tcode[Boolean]))
            val compare = ApplyComparisonOp(Compare(eltVType), In(0, eltType), In(1, eltType)) < 0
            InferPType(compare, Env.empty)
            (a, compare, sorter.distinctFromSorted(discardNext.invoke(_, _, _, _, _)), Array.empty[String])
          case ToDict(a) =>
            val elementType = a.pType.asInstanceOf[PStream].elementType
            val (k0, k1, keyType) = elementType match {
              case t: PStruct => (GetField(In(0, elementType), "key"), GetField(In(1, elementType), "key"), t.fieldType("key"))
              case t: PTuple => (GetTupleElement(In(0, elementType), 0), GetTupleElement(In(1, elementType), 0), t.types(0))
            }
            val discardNext = mb.genEmitMethod("discardNext", Array[TypeInfo[_]](typeInfo[Region], sorter.ti, typeInfo[Boolean], sorter.ti, typeInfo[Boolean]), typeInfo[Boolean])
            val cmp2 = ApplyComparisonOp(EQWithNA(keyType.virtualType), k0, k1).deepCopy()
            InferPType(cmp2, Env.empty)
            val EmitCode(s, m, pv) = emit(cmp2, discardNext, Env.empty, er, container)
            discardNext.emit(Code(s, m || pv.tcode[Boolean]))
            val compare = (ApplyComparisonOp(Compare(keyType.virtualType), k0, k1) < 0).deepCopy()
            InferPType(compare, Env.empty)
            (a, compare, Code(sorter.pruneMissing, sorter.distinctFromSorted(discardNext.invoke(_, _, _, _, _))), Array.empty[String])
        }

        val sort = vab.ti match {
          case BooleanInfo => sorter.sort(makeDependentSortingFunction[Boolean](region, eltType, compare, env, leftRightComparatorNames))
          case IntInfo => sorter.sort(makeDependentSortingFunction[Int](region, eltType, compare, env, leftRightComparatorNames))
          case LongInfo => sorter.sort(makeDependentSortingFunction[Long](region, eltType, compare, env, leftRightComparatorNames))
          case FloatInfo => sorter.sort(makeDependentSortingFunction[Float](region, eltType, compare, env, leftRightComparatorNames))
          case DoubleInfo => sorter.sort(makeDependentSortingFunction[Double](region, eltType, compare, env, leftRightComparatorNames))
        }

        val optStream = emitStream(array)
        val result = optStream.map { stream =>
          PCode(pt, Code(
            EmitStream.write(mb, stream, vab),
            sort,
            distinct,
            sorter.toRegion()))
        }

        COption.toEmitCode(result, atyp, mb)

      case CastToArray(a) =>
        val et = emit(a)
        EmitCode(et.setup, et.m, PCode(pt, et.v))

      case ToArray(a) =>
        EmitStream.toArray(mb, coerce[PArray](pt), emitStream(a))

      case x@LowerBoundOnOrderedCollection(orderedCollection, elem, onKey) =>
        val typ: PContainer = coerce[PIterable](orderedCollection.pType).asPContainer
        val a = emit(orderedCollection)
        val e = emit(elem)
        val bs = new BinarySearch[C](mb, typ, elem.pType, keyOnly = onKey)

        val localA = mb.newLocal[Long]()
        val localElementMB = mb.newLocal[Boolean]()
        val localElementValue = mb.newLocal()(typeToTypeInfo(elem.pType))
        EmitCode(
          Code(a.setup, e.setup),
          a.m,
          PCode(pt, Code(
            localA := a.value[Long],
            localElementMB := e.m,
            localElementMB.mux(Code._empty, localElementValue.storeAny(e.v)),
            bs.getClosestIndex(localA, localElementMB, localElementValue))))

      case GroupByKey(collection) =>
        // sort collection by group
        val atyp = coerce[PStream](collection.pType)
        val etyp = coerce[PBaseStruct](atyp.elementType)
        val ktyp = etyp.types(0)
        val vtyp = etyp.types(1)
        val eltOut = coerce[PDict](ir.pType).elementType

        val eab = new StagedArrayBuilder(etyp, mb, 16)
        val sorter = new ArraySorter(er, eab)

        val (k1, k2) = etyp match {
          case t: PStruct => GetField(In(0, t), "key") -> GetField(In(1, t), "key")
          case t: PTuple =>
            assert(t.fields(0).index == 0)
            GetTupleElement(In(0, t), 0) -> GetTupleElement(In(1, t), 0)
        }

        val compare = ApplyComparisonOp(Compare(etyp.types(0).virtualType), k1, k2) < 0
        InferPType(compare, Env.empty)
        val leftRightComparatorNames = Array.empty[String]
        val sortF = eab.ti match {
          case BooleanInfo => makeDependentSortingFunction[Boolean](region, etyp, compare, env, leftRightComparatorNames)
          case IntInfo => makeDependentSortingFunction[Int](region, etyp, compare, env, leftRightComparatorNames)
          case LongInfo => makeDependentSortingFunction[Long](region, etyp, compare, env, leftRightComparatorNames)
          case FloatInfo => makeDependentSortingFunction[Float](region, etyp, compare, env, leftRightComparatorNames)
          case DoubleInfo => makeDependentSortingFunction[Double](region, etyp, compare, env, leftRightComparatorNames)
        }

        val nab = new StagedArrayBuilder(PInt32(), mb, 0)
        val i = mb.newLocal[Int]()

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

        val compare2 = ApplyComparisonOp(EQWithNA(ktyp.virtualType), lastKey, currKey)
        InferPType(compare2, Env("i-1" -> etyp, "i" -> etyp))
        val isSame = emit(
          compare2,
          env = Env(
            ("i-1", eab.applyEV(mb, i-1)),
            ("i", eab.applyEV(mb, i))))

        val optStream = emitStream(collection)
        val result = optStream.map { stream =>
          PCode(pt, Code(
            EmitStream.write(mb, stream, eab),
            sorter.sort(sortF),
            sorter.pruneMissing,
            eab.size.ceq(0).mux(
              Code(srvb.start(0), srvb.offset),
              Code(
                nab.clear,
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
                      structbuilder.addArray(coerce[PArray](eltOut.types(1)), { arraybuilder =>
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
                  srvb.advance()),
                srvb.offset))))
        }

        COption.toEmitCode(result, pt, mb)

      case ArrayZeros(length) =>
        val lengthTriplet = emit(length)
        val outputPType = coerce[PArray](ir.pType)
        val elementSize = outputPType.elementByteSize
        val numElements = mb.genFieldThisRef[Int]()
        val arrayAddress = mb.genFieldThisRef[Long]()
        val result = Code(
          numElements := lengthTriplet.value[Int],
          arrayAddress := outputPType.allocate(region, numElements),
          outputPType.stagedInitialize(arrayAddress, numElements),
          Region.setMemory(outputPType.firstElementOffset(arrayAddress), numElements.toL * elementSize, 0.toByte),
          arrayAddress
        )
        EmitCode(lengthTriplet.setup, lengthTriplet.m, PCode(pt, result))

      case x@StreamFold(a, zero, accumName, valueName, body) =>
        val eltType = coerce[PStream](a.pType).elementType
        val accType = ir.pType

        val streamOpt = emitStream(a)
        val resOpt: COption[PCode] = streamOpt.flatMapCPS { (stream, _ctx, ret) =>
          implicit val c = _ctx

          val xAcc = mb.newEmitField(accumName, accType)
          val tmpAcc = mb.newEmitField(accumName, accType)

          def foldBody(elt: EmitCode): Code[Unit] = {
            val xElt = mb.newEmitField(valueName, eltType)
            val bodyenv = env.bind(accumName -> xAcc, valueName -> xElt)

            val codeB = emit(body, env = bodyenv)
            Code(xElt := elt,
              tmpAcc := codeB.map(v => accType.copyFromPValue(mb, region, PCode(body.pType, codeB.v))),
              xAcc := tmpAcc)
          }

          val codeZ = emit(zero).map(accType.copyFromPValue(mb, region, _))
          def retTT(): Code[Ctrl] =
            ret(COption.fromEmitCode(xAcc.get))

          stream.stream
                .fold(mb, xAcc := codeZ, foldBody, retTT())
        }

        COption.toEmitCode(resOpt, accType, mb)

      case x@StreamFold2(a, acc, valueName, seq, res) =>
        val eltType = coerce[PStream](a.pType).elementType

        val xElt = mb.newEmitField(valueName, eltType)
        val names = acc.map(_._1)
        val accVars = (names, x.accPTypes).zipped.map(mb.newEmitField)
        val tmpAccVars = (names, x.accPTypes).zipped.map(mb.newEmitField)

        val resEnv = env.bind(names.zip(accVars): _*)
        val seqEnv = resEnv.bind(valueName, xElt)

        val codeR = emit(res, env = resEnv)
        val typedCodeSeq = seq.map(ir => emit(ir, env = seqEnv))

        val streamOpt = emitStream(a)
         val resOpt = streamOpt.flatMapCPS[PCode] { (stream, _ctx, ret) =>
          implicit val c = _ctx

          def foldBody(elt: EmitCode): Code[Unit] =
            Code(xElt := elt,
              Code(tmpAccVars.zip(typedCodeSeq).map { case (v, x) =>
                v := x.castTo(mb, region, v.pt)
              }),
              accVars := tmpAccVars.load())

          def computeRes(): Code[Ctrl] =
              ret(COption.fromEmitCode(codeR))

          stream.stream
            .fold(mb, Code(accVars.zip(acc).map { case (v, (name, x)) =>
              v := emit(x).castTo(mb, region, v.pt)
            }),
              foldBody, computeRes())
        }

        COption.toEmitCode(resOpt, res.pType, mb)

      case StreamFor(a, valueName, body) =>
        val eltType = a.pType.asInstanceOf[PStream].elementType
        val streamOpt = emitStream(a)
        def forBody(elt: EmitCode): Code[Unit] = {
          val xElt = mb.newEmitField(valueName, eltType)
          val bodyenv = env.bind(
            (valueName, xElt))
          val codeB = emit(body, env = bodyenv)

          Code(xElt := elt, codeB.setup)
        }

        EmitCode(
          streamOpt.cases(mb)(
            Code._empty,
            stream =>
              stream.stream.forEach(mb)(forBody)),
          const(false),
          PCode._empty)

      case x@RunAgg(body, result, _) =>
        val aggs = x.physicalSignatures
        val (newContainer, aggSetup, aggCleanup) = AggContainer.fromClassBuilder(aggs, mb.ecb, "run_agg")
        val codeBody = emit(body, env = env, container = Some(newContainer))
        val codeRes = emit(result, env = env, container = Some(newContainer))
        val resm = mb.genFieldThisRef[Boolean]()
        val resv = mb.genFieldThisRef("run_agg_result")(typeToTypeInfo(result.pType))

        val aggregation = Code(
          aggSetup,
          codeBody.setup,
          codeRes.setup,
          resm := codeRes.m,
          resv.storeAny(resm.mux(defaultValue(result.pType), codeRes.v)),
          aggCleanup)

        EmitCode(aggregation, resm, PCode(pt, resv))

      case x@InitOp(i, args, _, op) =>
        val AggContainer(aggs, sc) = container.get
        val statePTypes = aggs(i).lookup(op).physicalInitOpArgs
        val rvAgg = agg.Extract.getAgg(aggs(i), op)

        val argVars = args.zip(statePTypes).map { case (a, t) =>
          emit(a, container = container.flatMap(_.nested(i, init = true)))
            .map(t.copyFromPValue(mb, region, _))
        }.toArray
        void(
          sc.newState(i),
          rvAgg.initOp(sc.states(i), argVars))

      case x@SeqOp(i, args, _, op) =>
        val AggContainer(aggs, sc) = container.get
        val aggSig = aggs(i)
        val statePTypes = aggSig.lookup(op).physicalSeqOpArgs
        val rvAgg = agg.Extract.getAgg(aggSig, op)

        val argVars = args.zip(statePTypes).map { case (a, t) =>
          emit(a, container = container.flatMap(_.nested(i, init = false)))
            .map(t.copyFromPValue(mb, region, _))
        }.toArray
        void(rvAgg.seqOp(sc.states(i), argVars))

      case x@CombOp(i1, i2, _) =>
        val AggContainer(aggs, sc) = container.get
        val aggSig = aggs(i1)
        assert(agg.Extract.compatible(aggs(i2), aggSig), s"${ aggs(i2) } vs $aggSig")
        val rvAgg = agg.Extract.getAgg(aggSig, aggSig.default)

        void(rvAgg.combOp(sc.states(i1), sc.states(i2)))

      case x@ResultOp(start, _) =>
        val newRegion = mb.genFieldThisRef[Region]()
        val AggContainer(aggs, sc) = container.get
        val srvb = new StagedRegionValueBuilder(EmitRegion(mb, newRegion), x.pType)
        val addFields = mb.wrapVoids(Array.tabulate(aggs.length) { j =>
          val idx = start + j
          val rvAgg = agg.Extract.getAgg(aggs(j), aggs(j).default)
          Code(
            rvAgg.result(sc.states(idx), srvb),
            srvb.advance())
        }, "aggs_result")

        present(pt, Code(
          newRegion := region,
          srvb.start(),
          addFields,
          sc.store,
          srvb.offset))

      case x@CombOpValue(i, value, sig) =>
        throw new NotImplementedError("CombOpValue emitter cannot be implemented until physical type passed across serialization boundary. See PR #8142")

      case x@AggStateValue(i, _) =>
        throw new NotImplementedError("AggStateValue emitter cannot be implemented until physical type passed across serialization boundary. See PR #8142")

      case x@SerializeAggs(start, sIdx, spec, sigs) =>
        val AggContainer(_, sc) = container.get
        val ob = mb.genFieldThisRef[OutputBuffer]()
        val baos = mb.genFieldThisRef[ByteArrayOutputStream]()

        val serialize = Array.range(start, start + sigs.length)
          .map { idx => sc.states(idx).serialize(spec)(ob) }

        void(
          baos := Code.newInstance[ByteArrayOutputStream](),
          ob := spec.buildCodeOutputBuffer(baos),
          mb.wrapVoids(serialize, "serialize_aggs"),
          ob.invoke[Unit]("flush"),
          ob.invoke[Unit]("close"),
          mb.setSerializedAgg(sIdx, baos.invoke[Array[Byte]]("toByteArray")),
          sc.store)

      case DeserializeAggs(start, sIdx, spec, sigs) =>
        val AggContainer(_, sc) = container.get
        val ib = mb.genFieldThisRef[InputBuffer]()

        val ns = sigs.length
        val deserializers = sc.states.states
          .slice(start, start + ns)
          .map(sc => sc.deserialize(BufferSpec.defaultUncompressed))

        val init = Code(Array.range(start, start + ns)
          .map(i => sc.newState(i)))

        val unserialize = Array.tabulate(ns) { j =>
          deserializers(j)(ib)
        }

        void(
          init,
          ib := spec.buildCodeInputBuffer(
            Code.newInstance[ByteArrayInputStream, Array[Byte]](
              mb.getSerializedAgg(sIdx))),
          mb.wrapVoids(unserialize, "deserialize_aggs"))

      case Begin(xs) =>
        EmitCode(
          wrapToMethod(xs) { case (_, t, code) =>
            code.setup
          },
          false,
          PCode._empty)

      case x@MakeStruct(fields) =>
        val srvb = new StagedRegionValueBuilder(mb, x.pType)
        val addFields = { (newMB: EmitMethodBuilder[C], t: PType, v: EmitCode) =>
          Code(
            v.setup,
            v.m.mux(srvb.setMissing(), srvb.addIRIntermediate(t)(v.v)),
            srvb.advance())
        }
        present(pt, Code(srvb.start(init = true), wrapToMethod(fields.map(_._2))(addFields), srvb.offset))

      case x@SelectFields(oldStruct, fields) =>
        val old = emit(oldStruct)
        val oldt = coerce[PStruct](oldStruct.pType)
        val oldv = mb.genFieldThisRef[Long]()
        val srvb = new StagedRegionValueBuilder(mb, x.pType)

        val addFields = fields.map { name =>
          new EstimableEmitter[C] {
            def estimatedSize: Int = 20

            def emit(mb: EmitMethodBuilder[C]): Code[Unit] = {
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

        EmitCode(
          old.setup,
          old.m,
          PCode(pt, Code(
            oldv := old.value[Long],
            srvb.start(),
            EmitUtils.wrapToMethod(addFields, mb),
            srvb.offset)))

      case x@InsertFields(old, fields, fieldOrder) =>
        if (fields.isEmpty)
          emit(old)
        else
          old.pType match {
            case oldtype: PStruct =>
              val codeOld = emit(old)
              val xo = mb.genFieldThisRef[Long]()
              val updateMap = Map(fields: _*)
              val srvb = new StagedRegionValueBuilder(mb, x.pType)

              val addFields = { (newMB: EmitMethodBuilder[C], t: PType, v: EmitCode) =>
                Code(
                  v.setup,
                  v.m.mux(srvb.setMissing(), srvb.addIRIntermediate(t)(v.v)),
                  srvb.advance())
              }

              val opSize: Int = 20
              val items = x.pType.fields.map { f =>
                updateMap.get(f.name) match {
                  case Some(vir) =>
                    new EstimableEmitter[C] {
                      def estimatedSize: Int = vir.size * opSize

                      def emit(mb: EmitMethodBuilder[C]): Code[Unit] =
                        addFields(mb, vir.pType, emitSelf.emit(vir, mb, env, EmitRegion.default(mb), container))
                    }
                  case None =>
                    val oldField = oldtype.field(f.name)
                    new EstimableEmitter[C] {
                      def estimatedSize: Int = 20

                      def emit(mb: EmitMethodBuilder[C]): Code[Unit] =
                        Code(
                          oldtype.isFieldMissing(xo, oldField.index).mux(
                            srvb.setMissing(),
                            srvb.addIRIntermediate(f.typ)(Region.loadIRIntermediate(oldField.typ)(oldtype.fieldOffset(xo, oldField.index)))),
                          srvb.advance())
                    }
                }
              }

              EmitCode(
                codeOld.setup,
                codeOld.m,
                PCode(pt, Code(
                  srvb.start(init = true),
                  xo := coerce[Long](codeOld.v),
                  EmitUtils.wrapToMethod(items, mb),
                  srvb.offset)))
            case _ =>
              val newIR = MakeStruct(fields)
              emit(newIR)
          }

      case GetField(o, name) =>
        EmitCode.fromI(mb) { cb =>
          emit(o).toI(cb).flatMap(cb) { oc =>
            val ov = oc.asBaseStruct.memoize(cb, "get_tup_elem_o")
            ov.loadField(cb, name)
          }
        }

      case x@MakeTuple(fields) =>
        val srvb = new StagedRegionValueBuilder(mb, x.pType)
        val addFields = { (newMB: EmitMethodBuilder[_], t: PType, v: EmitCode) =>
          Code(
            v.setup,
            v.m.mux(srvb.setMissing(), srvb.addIRIntermediate(t)(v.v)),
            srvb.advance())
        }
        present(pt, Code(srvb.start(init = true), wrapToMethod(fields.map(_._2))(addFields), srvb.offset))

      case GetTupleElement(o, i) =>
        EmitCode.fromI(mb) { cb =>
          emit(o).toI(cb).flatMap(cb) { oc =>
            val ov = oc.asBaseStruct.memoize(cb, "get_tup_elem_o")
            ov.loadField(cb, i)
          }
        }

      case In(i, typ) =>
        normalArgument(mb, i, typ)
      case Die(m, typ) =>
        val cm = emit(m)
        EmitCode(
          Code(
            cm.setup,
            Code._throw[HailException, Unit](Code.newInstance[HailException, String](
              cm.m.mux[String](
                "<exception message missing>",
                coerce[String](StringFunctions.wrapArg(er, m.pType)(cm.v)))))),
          false,
          pt.defaultValue)

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
              val methodbuilder = impl.getAsMethod(mb.ecb, pt, argPTypes: _*)
              methods.update(fn, methods(fn) :+ ((argPTypes, pt, methodbuilder)))
              methodbuilder
          }
        val codeArgs = args.map(emit(_))
        val vars = args.map { a => coerce[Any](mb.newLocal()(typeToTypeInfo(a.typ))) }
        val ins = vars.zip(codeArgs.map(_.v)).map { case (l, i) => l := i }
        val value = Code(Code(ins), meth.invoke[Any]((coerce[Any](mb.getArg[Region](1).get) +: vars.map(_.get)): _*))
        strict(pt, value, codeArgs: _*)
      case x@ApplySeeded(fn, args, seed, rt) =>
        val codeArgs = args.map(a => (a.pType, emit(a)))
        val impl = x.implementation
        val unified = impl.unify(args.map(_.typ) :+ rt)
        assert(unified)
        impl.setSeed(seed)
        impl.apply(er, pt, codeArgs: _*)
      case x@ApplySpecial(_, args, rt) =>
        val codeArgs = args.map(a => (a.pType, emit(a)))
        val impl = x.implementation
        impl.argTypes.foreach(_.clear())
        val unified = impl.unify(args.map(_.typ) :+ rt)
        assert(unified)
        impl.apply(er, pt, codeArgs: _*)
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
        val shapeAddress = mb.genFieldThisRef[Long]()

        val shapeTuple = new CodePTuple(shapePType, shapeAddress)

        val shapeVariables = (0 until nDims).map(_ => mb.newLocal[Long]()).toArray

        def shapeBuilder(srvb: StagedRegionValueBuilder): Code[Unit] = {
          Code(
            srvb.start(),
            Code.foreach(0 until nDims) { index =>
              Code(
                srvb.addLong(shapeVariables(index)),
                srvb.advance())
            })
        }

        val setup = Code(
          shapet.setup,
          datat.setup,
          rowMajort.setup)
        val result = Code(
          shapeAddress := shapet.value[Long],
          Code.foreach(0 until nDims) { index =>
            shapeTuple.isMissing(index).mux[Unit](
              Code._fatal[Unit](s"shape missing at index $index"),
              shapeVariables(index) := shapeTuple(index))
          },
          xP.construct(0, 0, shapeBuilder, xP.makeDefaultStridesBuilder(shapeVariables.map(_.load()), mb), requiredData, mb))
        EmitCode(setup, datat.m || shapet.m, PCode(pt, result))
      case NDArrayShape(ndIR) =>
        val ndt = emit(ndIR)
        val ndP = ndIR.pType.asInstanceOf[PNDArray]

        EmitCode(ndt.setup, ndt.m, PCode(pt, ndP.shape.load(ndt.value[Long])))
      case NDArrayRef(nd, idxs) =>
        val ndt = emit(nd)
        val idxst = idxs.map(emit(_))
        val childPType = coerce[PNDArray](nd.pType)
        val ndAddress = mb.genFieldThisRef[Long]()
        val overallMissing = mb.genFieldThisRef[Boolean]()

        val idxFields = idxst.map(_ => mb.genFieldThisRef[Long]())
        val idxFieldsBinding = Code(
          idxFields.zip(idxst).map { case (field, idxTriplet) =>
            field := idxTriplet.value[Long]
          })

        val setup = coerce[Unit](Code(
          ndt.setup,
          overallMissing := ndt.m,
          Code(idxst.map(_.setup)),
          Code.foreach(idxst.map(_.m)){ idxMissingness =>
            overallMissing := overallMissing || idxMissingness
          }))

        val value = Code(
          ndAddress := ndt.value[Long],
          idxFieldsBinding,
          childPType.outOfBounds(idxFields, ndAddress, mb).orEmpty(Code._fatal[Unit]("Index out of bounds")),
          childPType.loadElementToIRIntermediate(idxFields, ndAddress, mb))

        EmitCode(setup, overallMissing, PCode(pt, value))
      case x@NDArrayReindex(child, indexMap) =>
        val childt = emit(child)
        val childPType = coerce[PNDArray](child.pType)

        val setup = childt.setup
        val value =
          Code.memoize(childt.value[Long], "ndarr_reindex_child_addr") { childAddress =>
            Code.memoize(
              childPType.shape.load(childAddress), "ndarr_reindex_child_shape_addr",
              childPType.strides.load(childAddress), "ndarr_reindex_child_strides_addr") { (childShapeAddr, childStridesAddr) =>
              val childShape = new CodePTuple(childPType.shape.pType, childShapeAddr)
              val childStrides = new CodePTuple(childPType.strides.pType, childStridesAddr)

              x.pType.construct(
                childPType.flags.load(childAddress),
                childPType.offset.load(childAddress),
                { srvb =>
                  Code(
                    srvb.start(),
                    Code.foreach(indexMap) { childIndex =>
                      Code(
                        srvb.addLong(if (childIndex < childPType.nDims) childShape(childIndex) else 1L),
                        srvb.advance())
                    })
                },
                { srvb =>
                  Code(
                    srvb.start(),
                    Code.foreach(indexMap) { index =>
                      Code(
                        srvb.addLong(if (index < childPType.nDims) childStrides(index) else 0L),
                        srvb.advance())
                    })
                },
                childPType.data.load(childAddress),
                mb)
            }
          }
        EmitCode(setup, childt.m, PCode(pt, value))
      case x: NDArrayMap  =>  emitDeforestedNDArray(x)
      case x: NDArrayMap2 =>  emitDeforestedNDArray(x)
      case x: NDArrayReshape => emitDeforestedNDArray(x)
      case x: NDArrayConcat => emitDeforestedNDArray(x)
      case x: NDArraySlice => emitDeforestedNDArray(x)
      case x: NDArrayFilter => emitDeforestedNDArray(x)

      case NDArrayMatMul(lChild, rChild) =>
        val lT = emitNDArrayStandardStrides(lChild)
        val rT = emitNDArrayStandardStrides(rChild)

        val lPType = coerce[PNDArray](lChild.pType)
        val rPType = coerce[PNDArray](rChild.pType)

        val leftND = mb.genFieldThisRef[Long]()
        val rightND = mb.genFieldThisRef[Long]()

        val leftShape: Value[Long] = new Value[Long] {
          def get: Code[Long] = lPType.shape.load(leftND)
        }
        val rightShape: Value[Long] = new Value[Long] {
          def get: Code[Long] = rPType.shape.load(rightND)
        }

        val lShapeTuple = new CodePTuple(lPType.shape.pType, leftShape)
        val rShapeTuple = new CodePTuple(rPType.shape.pType, rightShape)

        val (leftShapeArraySetup, leftShapeArray) = (0 until lPType.nDims).map(i => coerce[Long](lShapeTuple(i))).cacheEntries(mb, LongInfo)
        val (rightShapeArraySetup, rightShapeArray) = (0 until rPType.nDims).map(i => coerce[Long](rShapeTuple(i))).cacheEntries(mb, LongInfo)

        val (unifyShapeSetup, unifiedShapeArray) = NDArrayEmitter.matmulShape(mb, leftShapeArray, rightShapeArray)

        val leftBroadcastMask = if (lPType.nDims > 2) NDArrayEmitter.broadcastMask(leftShapeArray) else IndexedSeq[Value[Long]]()
        val rightBroadcastMask = if (rPType.nDims > 2) NDArrayEmitter.broadcastMask(rightShapeArray) else IndexedSeq[Value[Long]]()

        val missingSetup = Code(
          lT.setup,
          rT.setup)

        val shapeSetup = Code(
            leftND := lT.value[Long],
            rightND := rT.value[Long],
            leftShapeArraySetup,
            rightShapeArraySetup,
            unifyShapeSetup)

        val outputPType = PNDArray(lPType.elementType, TNDArray.matMulNDims(lPType.nDims, rPType.nDims), true)

        val numericElementType = coerce[PNumeric](lPType.elementType)

        val eVti = typeToTypeInfo(numericElementType.virtualType)

        val isMissing = lT.m || rT.m

        if ((lPType.elementType.isInstanceOf[PFloat64] || lPType.elementType.isInstanceOf[PFloat32]) && lPType.nDims == 2 && rPType.nDims == 2) {
          val leftDataAddress = lPType.data.load(leftND)
          val rightDataAddress = rPType.data.load(rightND)

          val leftColumnMajorAddress = mb.newLocal[Long]()
          val rightColumnMajorAddress = mb.newLocal[Long]()
          val answerColumnMajorAddress = mb.newLocal[Long]()
          val answerRowMajorPArrayAddress = mb.genFieldThisRef[Long]()
          val M = leftShapeArray(lPType.nDims - 2)
          val N = rightShapeArray(rPType.nDims - 1)
          val K = leftShapeArray(lPType.nDims - 1)

          val LDA = M
          val LDB = K
          val LDC = M
          val elementByteSize = lPType.elementType.byteSize

          val multiplyViaDGEMM = Code(Code(FastIndexedSeq(
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
              method="copyColumnMajorToRowMajor", answerColumnMajorAddress, outputPType.data.pType.firstElementOffset(answerRowMajorPArrayAddress, (M * N).toI), M, N, const(lPType.elementType.byteSize)),
            Code.invokeStatic[Memory, Long, Unit]("free", leftColumnMajorAddress.load()),
            Code.invokeStatic[Memory, Long, Unit]("free", rightColumnMajorAddress.load()),
            Code.invokeStatic[Memory, Long, Unit]("free", answerColumnMajorAddress.load()))),
            outputPType.construct(0, 0, outputPType.makeShapeBuilder(IndexedSeq(M, N)), outputPType.makeDefaultStridesBuilder(IndexedSeq(M, N), mb), answerRowMajorPArrayAddress, mb)
          )

          EmitCode(missingSetup, isMissing, PCode(pt, multiplyViaDGEMM))
        } else {
          val emitter = new NDArrayEmitter(outputPType.nDims, unifiedShapeArray, lPType.shape.pType, lPType.elementType, shapeSetup, missingSetup, isMissing) {
            override def outputElement(elemMB: EmitMethodBuilder[_], idxVars: IndexedSeq[Value[Long]]): Code[_] = {
              val element = coerce[Any](elemMB.genFieldThisRef("matmul_element")(eVti))
              val k = elemMB.genFieldThisRef[Long]()

              val innerMethod = elemMB.genEmitMethod("ndaMatMulLoop", FastIndexedSeq.empty[TypeInfo[_]], eVti)

              val (lIndices: IndexedSeq[Value[Long]], rIndices: IndexedSeq[Value[Long]]) = (lPType.nDims, rPType.nDims, idxVars) match {
                case (1, 1, Seq()) => (IndexedSeq(k), IndexedSeq(k))
                case (1, _, stack :+ m) =>
                  val rStackVars = NDArrayEmitter.zeroBroadcastedDims(stack, rightBroadcastMask)
                  (IndexedSeq(k), rStackVars :+ k :+ m)
                case (_, 1, stack :+ n) =>
                  val lStackVars = NDArrayEmitter.zeroBroadcastedDims(stack, leftBroadcastMask)
                  (lStackVars :+ n :+ k, FastIndexedSeq(k))
                case (_, _, stack :+ n :+ m) =>
                  val lStackVars = NDArrayEmitter.zeroBroadcastedDims(stack, leftBroadcastMask)
                  val rStackVars = NDArrayEmitter.zeroBroadcastedDims(stack, rightBroadcastMask)
                  (lStackVars :+ n :+ k, rStackVars :+ k :+  m)
              }

              val lElem = lPType.loadElementToIRIntermediate(lIndices, leftND, innerMethod)
              val rElem = rPType.loadElementToIRIntermediate(rIndices, rightND, innerMethod)
              val kLen = elemMB.genFieldThisRef[Long]()

              val loopCode = Code(
                k := 0L,
                kLen := leftShapeArray(lPType.nDims - 1),
                element := numericElementType.zero,
                Code.whileLoop(k < kLen,
                  element := numericElementType.add(numericElementType.multiply(lElem, rElem), element),
                  k := k + 1L),
                element)
              innerMethod.emit(loopCode)
              innerMethod.invoke()
            }
          }
          emitter.emit(mb, outputPType)
        }

      case x@NDArrayQR(nd, mode) =>
        // See here to understand different modes: https://docs.scipy.org/doc/numpy/reference/generated/numpy.linalg.qr.html
        val ndt = emit(nd)
        val ndAddress = mb.genFieldThisRef[Long]()
        val ndPType = nd.pType.asInstanceOf[PNDArray]

        val shapeAddress: Value[Long] = new Value[Long] {
          def get: Code[Long] = ndPType.shape.load(ndAddress)
        }
        val shapeTuple = new CodePTuple(ndPType.shape.pType, shapeAddress)
        val shapeArray = (0 until ndPType.shape.pType.nFields).map(shapeTuple[Long](_))

        val LWORKAddress = mb.newLocal[Long]()

        val M = shapeArray(0)
        val N = shapeArray(1)
        val K = new Value[Long] {
          def get: Code[Long] = (M < N).mux(M, N)
        }
        val LDA = M // Possible stride tricks could change this in the future.

        def LWORK = Region.loadDouble(LWORKAddress).toI

        val dataAddress = ndPType.data.load(ndAddress)

        val tauPType = PArray(PFloat64Required, true)
        val tauAddress = mb.genFieldThisRef[Long]()
        val workAddress = mb.genFieldThisRef[Long]()
        val aAddressDGEQRF = mb.genFieldThisRef[Long]() // Should be column major
        val rDataAddress = mb.genFieldThisRef[Long]()
        val aNumElements = mb.genFieldThisRef[Long]()

        val infoDGEQRFResult = mb.newLocal[Int]()
        val infoDGEQRFErrorTest = (extraErrorMsg: String) => (infoDGEQRFResult cne  0)
          .orEmpty(Code._fatal[Unit](const(s"LAPACK error DGEQRF. $extraErrorMsg Error code = ").concat(infoDGEQRFResult.toS)))

        val computeHAndTau = Code(FastIndexedSeq(
          ndAddress := ndt.value[Long],
          aNumElements := ndPType.numElements(shapeArray.map(_.get), mb),

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
        ))

        val result = if (mode == "raw") {
          val rawPType = x.pType.asInstanceOf[PTuple]
          val rawOutputSrvb = new StagedRegionValueBuilder(mb, x.pType, region)
          val hPType = rawPType.types(0).asInstanceOf[PNDArray]
          val tauPType = rawPType.types(1).asInstanceOf[PNDArray]

          val hShapeArray = FastIndexedSeq[Value[Long]](N, M)
          val hShapeBuilder = hPType.makeShapeBuilder(hShapeArray.map(_.get))
          val hStridesBuilder = hPType.makeDefaultStridesBuilder(hShapeArray.map(_.get), mb)

          val tauShapeBuilder = tauPType.makeShapeBuilder(FastIndexedSeq(K.get))
          val tauStridesBuilder = tauPType.makeDefaultStridesBuilder(FastIndexedSeq(K.get), mb)

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
          val currRow = mb.genFieldThisRef[Int]()
          val currCol = mb.genFieldThisRef[Int]()

          val (rPType, rShapeArray) = if (mode == "r") {
            (x.pType.asInstanceOf[PNDArray], FastIndexedSeq[Value[Long]](K, N))
          } else if (mode == "complete") {
            (x.pType.asInstanceOf[PTuple].types(1).asInstanceOf[PNDArray], FastIndexedSeq[Value[Long]](M, N))
          } else if (mode == "reduced") {
            (x.pType.asInstanceOf[PTuple].types(1).asInstanceOf[PNDArray], FastIndexedSeq[Value[Long]](K, N))
          } else {
            throw new AssertionError(s"Unsupported QR mode $mode")
          }

          val rShapeBuilder = rPType.makeShapeBuilder(rShapeArray.map(_.get))
          val rStridesBuilder = rPType.makeDefaultStridesBuilder(rShapeArray.map(_.get), mb)

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
            val qShapeBuilder = qPType.makeShapeBuilder(qShapeArray.map(_.get))
            val qStridesBuilder = qPType.makeDefaultStridesBuilder(qShapeArray.map(_.get), mb)

            val rNDArrayAddress = mb.genFieldThisRef[Long]()
            val qDataAddress = mb.genFieldThisRef[Long]()

            val infoDORGQRResult = mb.genFieldThisRef[Int]()
            val infoDORQRErrorTest = (extraErrorMsg: String) => (infoDORGQRResult cne 0)
              .orEmpty(Code._fatal[Unit](const(s"LAPACK error DORGQR. $extraErrorMsg Error code = ").concat(infoDORGQRResult.toS)))

            val qCondition = mb.genFieldThisRef[Boolean]()
            val numColsToUse = mb.genFieldThisRef[Long]()
            val aAddressDORGQR = mb.genFieldThisRef[Long]()

            // val qNumElements = M * numColsToUse
            val qNumElements = mb.genFieldThisRef[Long]()

            val computeCompleteOrReduced = Code(Code(FastIndexedSeq(
              qCondition := const(mode == "complete") && (M > N),
              numColsToUse := qCondition.mux(M, K),
              qNumElements := M * numColsToUse,
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
              crOutputSrvb.advance())),
              crOutputSrvb.end()
            )

            Code(
              computeHAndTau,
              rNDArrayAddress := computeR,
              computeCompleteOrReduced
            )
          }
        }
        EmitCode(ndt.setup, ndt.m, PCode(pt, result))

      case x@CollectDistributedArray(contexts, globals, cname, gname, body) =>
        val ctxType = coerce[PArray](contexts.pType).elementType
        val gType = globals.pType
        val bType = body.pType

        val ctxTypeTuple = PTuple(ctxType)
        val gTypeTuple = PTuple(gType)
        val bTypeTuple = PTuple(bType)

        val spec = BufferSpec.defaultUncompressed
        val parentCB = mb.ecb

        val cCodec = TypedCodecSpec(ctxTypeTuple, spec)
        val gCodec = TypedCodecSpec(gTypeTuple, spec)
        val bCodec = TypedCodecSpec(bTypeTuple, spec)

        val functionID: String = {
          val bodyFB = EmitFunctionBuilder[Region, Array[Byte], Array[Byte], Array[Byte]]("collect_distributed_array")
          val bodyMB = bodyFB.genEmitMethod("cdaBody", Array[TypeInfo[_]](typeInfo[Region], typeToTypeInfo(ctxType), typeInfo[Boolean], typeToTypeInfo(gType), typeInfo[Boolean]), typeInfo[Long])

          val (cRetPtype, cDec) = cCodec.buildEmitDecoderF[Long](ctxTypeTuple.virtualType, bodyFB.ecb)
          val (gRetPtype, gDec) = gCodec.buildEmitDecoderF[Long](gTypeTuple.virtualType, bodyFB.ecb)
          val bEnc = bCodec.buildEmitEncoderF[Long](bTypeTuple, bodyFB.ecb)
          val bOB = bodyFB.genFieldThisRef[OutputBuffer]()

          assert(cRetPtype == ctxTypeTuple)
          assert(gRetPtype == gTypeTuple)

          val env = Env[EmitValue](
            (cname, bodyMB.getArgEV(0, ctxType)),
            (gname, bodyMB.getArgEV(1, gType)))

          // FIXME fix number of aggs here
          val m = MakeTuple.ordered(FastSeq(body))
          m._pType = PCanonicalTuple(true, body.pType)
          val t = new Emit(ctx, bodyFB.ecb).emit(m, bodyMB, env, EmitRegion.default(bodyMB), None)
          bodyMB.emit(Code(t.setup, t.m.mux(Code._fatal[Long]("return cannot be missing"), t.v)))

          val ctxIS = Code.newInstance[ByteArrayInputStream, Array[Byte]](bodyFB.getArg[Array[Byte]](2))
          val gIS = Code.newInstance[ByteArrayInputStream, Array[Byte]](bodyFB.getArg[Array[Byte]](3))

          val ctxOff = bodyFB.newLocal[Long]()
          val gOff = bodyFB.newLocal[Long]()
          val bOff = bodyFB.newLocal[Long]()
          val bOS = bodyFB.newLocal[ByteArrayOutputStream]()

          bodyFB.emit(Code(
            ctxOff := Code.memoize(cCodec.buildCodeInputBuffer(ctxIS), "cda_ctx_ib") { ib =>
              cDec(bodyFB.getArg[Region](1), ib)
            },
            gOff := Code.memoize(gCodec.buildCodeInputBuffer(gIS), "cda_g_ib") { ib =>
              gDec(bodyFB.getArg[Region](1), ib)
            },
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
          parentCB.addModule(fID, bodyFB.resultWithIndex())
          fID
        }

        val spark = parentCB.backend()

        val optCtxStream = emitStream(contexts)
        val globalsT = emit(globals)

        val cEnc = cCodec.buildEmitEncoderF[Long](ctxTypeTuple, parentCB)
        val gEnc = gCodec.buildEmitEncoderF[Long](gTypeTuple, parentCB)
        val (bRetPType, bDec) = bCodec.buildEmitDecoderF[Long](bTypeTuple.virtualType, parentCB)

        assert(bRetPType == bTypeTuple)

        val baos = mb.genFieldThisRef[ByteArrayOutputStream]()
        val buf = mb.genFieldThisRef[OutputBuffer]()
        val ctxab = mb.genFieldThisRef[ByteArrayArrayBuilder]()
        val encRes = mb.genFieldThisRef[Array[Array[Byte]]]()

        def etToTuple(et: EmitCode, t: PType): Code[Long] = {
          val srvb = new StagedRegionValueBuilder(mb, PTuple(t))
          Code(
            srvb.start(),
            et.setup,
            et.m.mux(
              srvb.setMissing(),
              srvb.addIRIntermediate(t)(et.v)),
            srvb.offset)
        }

        def addContexts(ctxStream: EmitStream.SizedStream): Code[Unit] =
          Code(
            ctxStream.length match {
              case None => ctxab.invoke[Int, Unit]("ensureCapacity", 16)
              case Some((setupLen, len)) => Code(setupLen, ctxab.invoke[Int, Unit]("ensureCapacity", len))
            },
            ctxStream.stream.map(etToTuple(_, ctxType)).forEach(mb) { offset =>
              Code(
                baos.invoke[Unit]("reset"),
                Code.memoize(offset, "cda_add_contexts_addr") { offset =>
                  cEnc(region, offset, buf)
                },
                buf.invoke[Unit]("flush"),
                ctxab.invoke[Array[Byte], Unit]("add", baos.invoke[Array[Byte]]("toByteArray")))
            })

        val addGlobals = Code(
          Code.memoize(etToTuple(globalsT, gType), "cda_g") { g =>
            gEnc(region, g, buf)
          },
          buf.invoke[Unit]("flush"))

        val decodeResult = {
          val sab = new StagedRegionValueBuilder(mb, x.pType)
          val bais = Code.newInstance[ByteArrayInputStream, Array[Byte]](encRes(sab.arrayIdx))
          val eltTupled = mb.genFieldThisRef[Long]()
          Code(
            sab.start(encRes.length()),
            Code.whileLoop(sab.arrayIdx < encRes.length(),
              eltTupled := Code.memoize(bCodec.buildCodeInputBuffer(bais), "decode_ib") { ib =>
                bDec(region, ib)
              },
              bTypeTuple.isFieldMissing(eltTupled, 0).mux(
                sab.setMissing(),
                sab.addIRIntermediate(bType)(Region.loadIRIntermediate(bType)(bTypeTuple.fieldOffset(eltTupled, 0)))),
              sab.advance()),
            sab.end())
        }

        val optRes = optCtxStream.map { ctxStream => PCode(pt, Code(
          baos := Code.newInstance[ByteArrayOutputStream](),
          buf := cCodec.buildCodeOutputBuffer(baos), // TODO: take a closer look at whether we need two codec buffers?
          ctxab := Code.newInstance[ByteArrayArrayBuilder, Int](16),
          addContexts(ctxStream),
          baos.invoke[Unit]("reset"),
          addGlobals,
          encRes := spark.invoke[String, Array[Array[Byte]], Array[Byte], Array[Array[Byte]]](
            "collectDArray", functionID,
            ctxab.invoke[Array[Array[Byte]]]("result"),
            baos.invoke[Array[Byte]]("toByteArray")),
          decodeResult))
        }

        COption.toEmitCode(optRes, x.pType, mb)

      case x@TailLoop(name, args, body) =>
        val label = CodeLabel()
        val loopRef = LoopRef(mb, label, args.map { case (name, x) => (name, x.pType) })

        val m = mb.genFieldThisRef[Boolean]()
        val v = mb.newPField(x.pType)

        val argEnv = env
          .bind((args.map(_._1), loopRef.loopArgs).zipped.toArray: _*)

        val newLoopEnv = loopEnv.getOrElse(Env.empty)
        val bodyT = emit(body, env = argEnv, loopEnv = Some(newLoopEnv.bind(name, loopRef)))
        val bodyF = Code(
          bodyT.setup,
          m := bodyT.m,
          (!m).orEmpty(v := bodyT.pv))

        EmitCode(Code(loopRef.loopArgs := args.map { case (_, x) => emit(x) }, label, bodyF), m, v.load())

      case Recur(name, args, _) =>
        val loopRef = loopEnv.get.lookup(name)

        EmitCode(
          Code(
            loopRef.tmpLoopArgs := args.map { arg =>
              emit(arg, loopEnv = None)
            },
            loopRef.loopArgs := loopRef.tmpLoopArgs.load(),
            loopRef.L.goto),
          // dead code
          const(true), pt.defaultValue)
      case x@ReadValue(path, spec, requestedType) =>
        val p = emit(path)
        val pathString = coerce[PString](path.pType).loadString(p.value[Long])
        val rowBuf = spec.buildCodeInputBuffer(mb.getUnsafeReader(pathString, true))
        val (pt, dec) = spec.buildEmitDecoderF(requestedType, mb.ecb, typeToTypeInfo(x.pType))
        EmitCode(p.setup, p.m, PCode(pt,
          Code.memoize(rowBuf, "read_ib") { ib =>
            dec(er.region, ib)
          }))
      case x@WriteValue(value, pathPrefix, spec) =>
        val v = emit(value)
        val p = emit(pathPrefix)
        val m = mb.newLocal[Boolean]()
        val pv = mb.newLocal[String]()
        val rb = mb.newLocal[OutputBuffer]()

        val taskCtx = Code.invokeScalaObject[HailTaskContext](HailTaskContext.getClass, "get")
        val vti = typeToTypeInfo(value.pType)

        EmitCode(
          Code(
            p.setup, v.setup,
            m := p.m || v.m,
            m.mux(
              Code(pv := Code._null[String], rb := Code._null[OutputBuffer]),
              Code(
                pv := coerce[PString](pathPrefix.pType).loadString(p.value[Long]),
                Code.memoize(taskCtx, "write_val_task_ctx") { taskCtx =>
                  (!taskCtx.isNull).orEmpty(
                    pv := pv.concat("-").concat(taskCtx.invoke[String]("partSuffix")))
                },
                rb := spec.buildCodeOutputBuffer(mb.getUnsafeWriter(pv)),
                vti match {
                  case vti: TypeInfo[t] =>
                    val enc = spec.buildEmitEncoderF(value.pType, mb.ecb, vti)
                    Code.memoize(v.value[t], "write_value") { v =>
                        enc(er.region, v, rb)
                    }(vti)
                },
                rb.invoke[Unit]("close")
              ))
          ), m,
          PCode(x.pType, coerce[PString](x.pType).allocateAndStoreString(mb, er.region, pv)))
    }
  }

  private def capturedReferences(ir: IR): (IR, (Emit.E, DependentEmitFunction[_]) => Emit.E) = {
    var ids = Set[String]()

    VisitIR(ir) {
      case Ref(id, _) =>
        ids += id
      case _ =>
    }

    (ir, { (env: Emit.E, f: DependentEmitFunction[_]) =>
      Env[EmitValue](ids.toFastSeq.flatMap { id =>
         env.lookupOption(id).map { e =>
           (id, f.newDepEmitField(e.get))
        }
      }: _*)
    })
  }

  private def makeDependentSortingFunction[T: TypeInfo](
    region: Code[Region],
    elemPType: PType, ir: IR, env: Emit.E, leftRightComparatorNames: Array[String]): DependentEmitFunction[AsmFunction2[T, T, Boolean]] = {
    val (newIR, getEnv) = capturedReferences(ir)
    val f = cb.genDependentFunction[T, T, Boolean](baseName = "sort_compare")
    val fregion = f.newDepField[Region](region)
    var newEnv = getEnv(env, f)

    val sort = f.genEmitMethod[Region, T, Boolean, T, Boolean, Boolean]("sort")

    if (leftRightComparatorNames.nonEmpty) {
      assert(leftRightComparatorNames.length == 2)
      newEnv = newEnv.bindIterable(
        IndexedSeq(
          (leftRightComparatorNames(0), sort.getArgEV(0, elemPType)),
          (leftRightComparatorNames(1), sort.getArgEV(1, elemPType))))
    }

    val EmitCode(setup, m, v) = new Emit(ctx, f.ecb).emit(newIR, sort, newEnv, EmitRegion.default(sort), None)

    sort.emit(Code(setup, m.mux(Code._fatal[Boolean]("Result of sorting function cannot be missing."), v.code)))
    f.apply_method.emit(sort.invoke(fregion, f.getArg[T](1), false, f.getArg[T](2), false))
    f
  }

  private def present(pt: PType, c: Code[_]): EmitCode =
    EmitCode(Code._empty, false, PCode(pt, c))

  private def void(x: Code[Unit]*): EmitCode = EmitCode(Code(x), false, PCode._empty)

  private def strict(pt: PType, value: Code[_], args: EmitCode*): EmitCode = {
    EmitCode(
      Code(args.map(_.setup)),
      if (args.isEmpty) false else args.map(_.m).reduce(_ || _),
      PCode(pt, value))
  }

  private[ir] def normalArgument(mb: EmitMethodBuilder[_], idx: Int, pType: PType): EmitCode = {
    val i = 2 + idx * 2
    EmitCode(Code._empty,
      mb.getArg[Boolean](i + 1),
      PCode(pType, mb.getArg(i)(typeToTypeInfo(pType)).load()))
  }

  def deforestNDArray(mb: EmitMethodBuilder[C], er: EmitRegion, x: IR, env: Emit.E): NDArrayEmitter = {
    def deforest(nd: IR): NDArrayEmitter = deforestNDArray(mb, er, nd, env)

    val xType = coerce[PNDArray](x.pType)
    val nDims = xType.nDims

    x match {
      case NDArrayMap(child, elemName, body) =>
        val childP = child.pType.asInstanceOf[PNDArray]
        val elemPType = childP.elementType
        val elemRef = mb.newPresentEmitField(elemPType) // FIXME elemName
        val bodyEnv = env.bind(elemName, elemRef)
        val bodyt = this.emit(body, mb, bodyEnv, er, None)

        val childEmitter = deforest(child)
        val setup = childEmitter.setupShape

        new NDArrayEmitter(childEmitter.nDims, childEmitter.outputShape,
          childP.shape.pType, body.pType, setup, childEmitter.setupMissing, childEmitter.missing) {
          override def outputElement(elemMB: EmitMethodBuilder[_], idxVars: IndexedSeq[Value[Long]]): Code[_] = {
            Code(
              elemRef := PCode(elemPType, childEmitter.outputElement(elemMB, idxVars)),
              bodyt.setup,
              bodyt.m.orEmpty(Code._fatal[Unit]("NDArray map body cannot be missing")),
              bodyt.v
            )
          }
        }
      case NDArrayMap2(lChild, rChild, lName, rName, body) =>
        val lP = coerce[PNDArray](lChild.pType)
        val rP = coerce[PNDArray](rChild.pType)

        val lElemRef = mb.newPresentEmitField(lName, lP.elementType)
        val rElemRef = mb.newPresentEmitField(rName, rP.elementType)

        val bodyEnv = env.bind(lName, lElemRef)
          .bind(rName, rElemRef)
        val bodyt = this.emit(body, mb, bodyEnv, er, None)

        val leftChildEmitter = deforest(lChild)
        val rightChildEmitter = deforest(rChild)

        val (newSetupShape, shapeArray) = NDArrayEmitter.unifyShapes2(mb, leftChildEmitter.outputShape, rightChildEmitter.outputShape)

        val setupMissing = Code(leftChildEmitter.setupMissing, rightChildEmitter.setupMissing)
        val setupShape = Code(leftChildEmitter.setupShape, rightChildEmitter.setupShape, newSetupShape)

        new NDArrayEmitter(lP.shape.pType.size, shapeArray, lP.shape.pType, body.pType, setupShape, setupMissing, leftChildEmitter.missing || rightChildEmitter.missing) {
          override def outputElement(elemMB: EmitMethodBuilder[_], idxVars: IndexedSeq[Value[Long]]): Code[_] = {

            val lIdxVars2 = NDArrayEmitter.zeroBroadcastedDims2(elemMB, idxVars, nDims, leftChildEmitter.outputShape)
            val rIdxVars2 = NDArrayEmitter.zeroBroadcastedDims2(elemMB, idxVars, nDims, rightChildEmitter.outputShape)

            Code(
              lElemRef := PCode(lP, leftChildEmitter.outputElement(elemMB, lIdxVars2)),
              rElemRef := PCode(rP, rightChildEmitter.outputElement(elemMB, rIdxVars2)),
              bodyt.setup,
              bodyt.m.orEmpty(Code._fatal[Unit]("NDArray map body cannot be missing")),
              bodyt.v
            )
          }
        }

      case x@NDArrayReindex(child, indexExpr) =>
        val childEmitter = deforest(child)
        val childPType = child.pType.asInstanceOf[PNDArray]

        val outputPType = x.pType
        val outputShapePType = outputPType.shape.pType

        val shapeSeq = indexExpr.map { childIndex =>
          if (childIndex < childPType.nDims)
            childEmitter.outputShape(childIndex)
          else
            const(1L)
        }

        new NDArrayEmitter(indexExpr.length, shapeSeq, outputShapePType, outputPType.elementType, childEmitter.setupShape, childEmitter.setupMissing, childEmitter.missing) {
          override def outputElement(elemMB: EmitMethodBuilder[_], idxVars: IndexedSeq[Value[Long]]): Code[_] = {
            val concreteIdxsForChild = Array.tabulate(childEmitter.nDims) { childDim =>
              val parentDim = indexExpr.indexOf(childDim)
              idxVars(parentDim)
            }
            childEmitter.outputElement(elemMB, concreteIdxsForChild)
          }
        }

      case x@NDArrayReshape(childND, shape) =>

        // Need to take this shape, which may have a -1 in it, and turn it into a compatible shape if possible.
        def compatibleShape(numElements: Value[Long], requestedShape: IndexedSeq[Value[Long]]): (Code[Unit], IndexedSeq[Value[Long]]) = {
          val hasNegativeOne = mb.newLocal[Boolean]()
          val runningProduct = mb.newLocal[Long]()
          val quotient = mb.newLocal[Long]()
          val tempShapeElement = mb.newLocal[Long]()

          val newShapeVars = requestedShape.indices.map(_ => mb.genFieldThisRef[Long]())

          val setupShape = coerce[Unit](Code(
            hasNegativeOne := false,
            runningProduct := 1L,

            Code.foreach(requestedShape) { requestedShapeElement => Code(
              tempShapeElement := requestedShapeElement,
              (tempShapeElement <= 0L).mux(
                (tempShapeElement ceq -1L).mux(
                  hasNegativeOne.mux(
                    Code._fatal[Unit]("Can't infer shape, more than one -1"),
                    hasNegativeOne := true
                  ),
                  Code._fatal[Unit]("Can't reshape, new shape must contain only positive numbers or -1")),
                runningProduct := runningProduct * tempShapeElement
              )
            ) },
            hasNegativeOne.mux(
              (numElements % runningProduct) > 0L,
              numElements cne runningProduct
            ).orEmpty(Code._fatal[Unit]("Can't reshape since requested shape is incompatible with number of elements")),
            quotient := numElements / runningProduct,
            Code(newShapeVars.zip(requestedShape).map { case (variable, shapeElement) =>
              variable := (shapeElement ceq -1L).mux(quotient, shapeElement)
            })
          ))

          (setupShape, newShapeVars)
        }

        val childEmitter = deforest(childND)

        val requestedShapet = emit(shape, mb, env, er, None)
        val requestedShapeAddress = mb.genFieldThisRef[Long]()
        val requestedShapePType = coerce[PTuple](shape.pType)
        val requestedShapeTuple = new CodePTuple(requestedShapePType, requestedShapeAddress)
        val requestedShapeArray = (0 until requestedShapePType.size).map(i => requestedShapeTuple[Long](i)).toArray

        val (childShapeCachingCode, childShapeCached) = childEmitter.outputShape.cacheEntries(mb, LongInfo)

        val numElements = mb.genFieldThisRef[Long]()

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

        new NDArrayEmitter(reshapedShapeArray.length, reshapedShapeArray, requestedShapePType.setRequired(true).asInstanceOf[PTuple],
          childEmitter.outputElementPType, setupShape, setupMissing, childEmitter.missing || requestedShapet.m) {
          override def outputElement(elemMB: EmitMethodBuilder[_], idxVars: IndexedSeq[Value[Long]]): Code[_] = {
            val storeElementIndex = elemMB.genFieldThisRef[Long]()

            val (newIdxVarsSetup, newIdxVars) = x.pType.unlinearizeIndexRowMajor(storeElementIndex, childShapeCached, elemMB)

            assert(newIdxVars.length == childEmitter.nDims)

            Code(
              storeElementIndex := x.pType.linearizeIndicesRowMajor(idxVars, reshapedShapeArray, elemMB),
              newIdxVarsSetup,
              childEmitter.outputElement(elemMB, newIdxVars)
            )
          }
        }

      case x@NDArrayConcat(nds, axis) =>
        val inputType = coerce[PArray](nds.pType)
        val inputNDType = coerce[PNDArray](inputType.elementType)

        val ndType = coerce[PNDArray](x.pType)
        val codeNDs = emit(nds, mb, env, er, None)

        val inputArray = mb.genFieldThisRef[Long]()
        val n = mb.genFieldThisRef[Int]()
        val i = mb.genFieldThisRef[Int]()

        val loadAndValidateArray = Code(
          inputArray := codeNDs.value[Long],
          n := inputType.loadLength(inputArray),
          (n < 1).orEmpty(Code._fatal[Unit]("NDArrayConcat: can't concatenate 0 NDArrays")))

        val (missingSetup: Code[Unit @unchecked], missing: Code[Boolean @unchecked], setupShape: Code[Unit @unchecked]) = (inputType.required, inputNDType.required) match {
          case (true, true) => (Code._empty, false: Code[Boolean], Code(
            codeNDs.setup,
            codeNDs.m.orEmpty(Code._fatal[Unit]("NDArrayConcat: required NDArray can't be missing")),
            loadAndValidateArray))
          case (false, true) => (codeNDs.setup, codeNDs.m, loadAndValidateArray)
          case _ =>
            val m = mb.genFieldThisRef[Boolean]()
            val setup = Code(
              codeNDs.setup,
              m := codeNDs.m,
              (!m).orEmpty(
                Code(
                  loadAndValidateArray,
                  i := 0,
                  Code.whileLoop(i < n,
                    m := m | inputType.isElementMissing(inputArray, i),
                    i := i + 1))
              ))
            (setup, m.load(), Code._empty)
        }

        val sb = SetupBuilder(mb, setupShape)
        val outputShape = sb.map(0 until ndType.nDims) { (sb, idx) =>
          val localDim = mb.genFieldThisRef[Long]()

          sb += Code(
            localDim := inputNDType.dimensionLength(inputType.loadElement(inputArray, 0), idx),
            i := 1,
            Code.whileLoop(i < n,
              {
                if (idx == axis)
                  localDim := localDim + inputNDType.dimensionLength(inputType.loadElement(inputArray, i), idx)
                else
                  inputNDType.dimensionLength(inputType.loadElement(inputArray, i), idx).cne(localDim)
                    .orEmpty(Code._fatal[Unit](
                      const(s"NDArrayConcat: mismatched dimensions of input NDArrays along axis $i: expected ")
                        .concat(localDim.toS).concat(", got ")
                        .concat(inputNDType.dimensionLength(inputType.loadElement(inputArray, i), idx).toS)))
              },
              i := i + 1))

          localDim
        }

        val setupShape2 = sb.result()

        new NDArrayEmitter(x.typ.nDims,
          outputShape,
          ndType.shape.pType,
          ndType.elementType,
          setupShape2,
          missingSetup,
          missing) {
          override def outputElement(elemMB: EmitMethodBuilder[_], idxVars: IndexedSeq[Value[Long]]): Code[_] = {
            val concatAxisIdx = elemMB.newLocal[Long]()

            val setupTransformedIdx = Code(
              i := 0,
              concatAxisIdx := idxVars(axis),
              Code.whileLoop(concatAxisIdx >= inputNDType.dimensionLength(inputType.loadElement(inputArray, i), axis),
                concatAxisIdx := concatAxisIdx - inputNDType.dimensionLength(inputType.loadElement(inputArray, i), axis),
                i := i + 1),
              (i > n).orEmpty(Code._fatal[Unit]("NDArrayConcat: trying to access element greater than length of concatenation axis")))

            val transformedIdxs = Array.tabulate(x.typ.nDims) { idx =>
              if (idx == axis) concatAxisIdx else idxVars(idx)
            }.toFastIndexedSeq
            Code(
              setupTransformedIdx,
              inputNDType.loadElementToIRIntermediate(transformedIdxs, new Value[Long] {
                def get: Code[Long] = inputType.loadElement(inputArray, i)
              }, elemMB))
          }
        }


      case x@NDArraySlice(child, slicesIR) =>
        val childEmitter = deforest(child)

        val slicest = emit(slicesIR, mb, env, er, None)
        val slicesValueAddress = mb.genFieldThisRef[Long]("ndarr_slicev")
        val slicesm = mb.genFieldThisRef[Boolean]("ndarr_slicem")
        val slices = new CodePTuple(coerce[PTuple](slicesIR.pType), slicesValueAddress)

        val slicers = slices.withTypes.collect {
          case (t: PTuple, slice) => new CodePTuple(t, coerce[Long](slice))
        }

        val missingSliceElements = slicers.map(_.missingnessPattern.reduce(_ || _)).fold(false: Code[Boolean])(_ || _)
        val anyMissingness = missingSliceElements || slices.missingnessPattern.fold(false: Code[Boolean])(_ || _)

        val codeSlices = slicers.map(_.values[Long, Long, Long])

        val sb = SetupBuilder(mb, childEmitter.setupShape)
        val outputShape = codeSlices.zipWithIndex.map { case ((start, stop, step), i) =>
          sb.memoizeField(
            (step >= 0L && start <= stop).mux(
              const(1L) + ((stop - start) - 1L) / step,
              (step < 0L && start >= stop).mux(
                (((stop - start) + 1L) / step) + 1L,
                0L)),
            s"nda_slice_shape$i")
        }

        val setupShape = sb.result()

        val setupMissing = Code(childEmitter.setupMissing,
          slicesm := slicest.m,
          slicesValueAddress := slicesm.mux(0L, slicest.value[Long]),
          slicest.setup)

        val missing = childEmitter.missing || anyMissingness

        new NDArrayEmitter(x.pType.nDims, outputShape, x.pType.shape.pType, x.pType.elementType, setupShape, setupMissing, missing) {
          override def outputElement(elemMB: EmitMethodBuilder[_], idxVars: IndexedSeq[Value[Long]]): Code[_] = {
            val oldIdxVarsIter = idxVars.iterator

            val sliceIdxVars2: IndexedSeq[Value[Long]] = slices.withTypes.map {
              case (_: PInt64, indexer) =>
                coerce[Long](indexer)
              case (t: PTuple, slicer) =>
                val (start, _, step) = new CodePTuple(t, coerce[Long](slicer)).values[Long, Long, Long]
                new Value[Long] {
                  def get: Code[Long] = start + oldIdxVarsIter.next() * step
                }
            }

            childEmitter.outputElement(elemMB, sliceIdxVars2)
          }
        }

      case x@NDArrayFilter(child, filters) =>
        val childEmitter = deforest(child)

        val sb = SetupBuilder(mb, childEmitter.setupShape)

        val (vars, outputShape) = filters.zipWithIndex.map { case (f, i) =>
          val codeF = emit(f, mb, env, er, None)
          val m = mb.genFieldThisRef[Boolean](s"m_filter$i")
          val v = mb.genFieldThisRef[Long](s"v_filter$i")

          val shapeVar = sb.memoizeField(Code(
              codeF.setup,
              m := codeF.m,
              m.mux(
                Code(v := 0L, childEmitter.outputShape(i)),
                Code(v := codeF.value[Long], coerce[PArray](f.pType).loadLength(v).toL))),
            s"nda_filter_shape$i")

          ((m, v), shapeVar)
        }.unzip

        val setupShape = sb.result()

        new NDArrayEmitter(x.pType.nDims, outputShape, x.pType.shape.pType, x.pType.elementType, setupShape, childEmitter.setupMissing, childEmitter.missing) {
          override def outputElement(elemMB: EmitMethodBuilder[_], idxVars: IndexedSeq[Value[Long]]): Code[_] = {
            val newIdxVars: IndexedSeq[Settable[Long]] = Array.tabulate(x.pType.nDims) { _ => mb.genFieldThisRef[Long]() }

            Code(
              Code(
                Array.tabulate(x.pType.nDims) { i =>
                  val (m, v) = vars(i)
                  val typ = coerce[PArray](filters(i).pType)
                  newIdxVars(i) := m.mux(
                    idxVars(i),
                    typ.isElementMissing(v, idxVars(i).toI).mux(
                      Code._fatal[Long](s"NDArrayFilter: can't filter on missing index (axis=$i)"),
                      Region.loadLong(typ.loadElement(v.load(), idxVars(i).toI))))
                }),
              childEmitter.outputElement(elemMB, newIdxVars))
          }
        }

      case _ =>
        val ndt = emit(x, mb, env, er, None)
        val ndAddress = mb.genFieldThisRef[Long]()
        val setup = (ndAddress := ndt.value[Long])
        val xP = x.pType.asInstanceOf[PNDArray]

        val shapeAddress = new Value[Long] {
          def get: Code[Long] = xP.shape.load(ndAddress)
        }
        val shapeTuple = new CodePTuple(xP.shape.pType, shapeAddress)

        val shapeArray = (0 until xP.shape.pType.nFields).map(i => shapeTuple.apply[Long](i))

        new NDArrayEmitter(nDims, shapeArray,
          xP.shape.pType, xP.elementType, setup, ndt.setup, ndt.m) {
          override def outputElement(elemMB: EmitMethodBuilder[_], idxVars: IndexedSeq[Value[Long]]): Code[_] =
            xP.loadElementToIRIntermediate(idxVars, ndAddress, elemMB)
        }
    }
  }
}

object NDArrayEmitter {

  def zeroBroadcastedDims2(mb: EmitMethodBuilder[_], loopVars: IndexedSeq[Value[Long]], nDims: Int, shapeArray: IndexedSeq[Value[Long]]): IndexedSeq[Value[Long]] = {
    val broadcasted = 0L
    val notBroadcasted = 1L
    Array.tabulate(nDims)(dim => new Value[Long] {
      def get: Code[Long] = (shapeArray(dim) > 1L).mux(notBroadcasted, broadcasted) * loopVars(dim)
    })
  }

  def broadcastMask(shapeArray: IndexedSeq[Code[Long]]): IndexedSeq[Value[Long]] = {
    val broadcasted = 0L
    val notBroadcasted = 1L
    shapeArray.map(shapeElement => new Value[Long] {
      def get: Code[Long] = (shapeElement > 1L).mux(notBroadcasted, broadcasted)
    })
  }

  def zeroBroadcastedDims(indices: IndexedSeq[Code[Long]], broadcastMask: IndexedSeq[Code[Long]]): IndexedSeq[Value[Long]] = {
    indices.zip(broadcastMask).map { case (index, flag) => new Value[Long] {
      def get: Code[Long] = index * flag
    }}
  }

  def unifyShapes2(mb: EmitMethodBuilder[_], leftShape: IndexedSeq[Value[Long]], rightShape: IndexedSeq[Value[Long]]): (Code[Unit], IndexedSeq[Value[Long]]) = {
    val sb = SetupBuilder(mb)

    val shape = leftShape.zip(rightShape).zipWithIndex.map { case ((left, right), i) =>
      val notSameAndNotBroadcastable = !((left ceq right) || (left ceq 1L) || (right ceq 1L))
      sb.memoizeField(
        notSameAndNotBroadcastable.mux(
          Code._fatal[Long]("Incompatible NDArray shapes"),
          (left > right).mux(left, right)),
        s"unify_shapes2_shape$i")
    }

    (sb.result(), shape)
  }

  def matmulShape(mb: EmitMethodBuilder[_], leftShape: IndexedSeq[Value[Long]], rightShape: IndexedSeq[Value[Long]]): (Code[Unit], IndexedSeq[Value[Long]]) = {
    val sb = SetupBuilder(mb)

    assert(leftShape.nonEmpty)
    assert(rightShape.nonEmpty)

    var lK: Value[Long] = null
    var rK: Value[Long] = null
    var shape: IndexedSeq[Value[Long]] = null
    var setup: Code[Unit] = Code._empty

    if (leftShape.length == 1) {
      lK = leftShape.head
      if (rightShape.length == 1) {
        rK = rightShape.head
        shape = FastIndexedSeq()
      } else {
        rK = rightShape(rightShape.length - 2)
        shape = rightShape.slice(0, rightShape.length - 2) :+ rightShape.last
      }
    } else {
      lK = leftShape.last
      if (rightShape.length == 1) {
        rK = rightShape.head
        shape = leftShape.slice(0, leftShape.length - 1)
      } else {
        rK = rightShape(rightShape.length - 2)
        val (unifiedSetup, unifiedShape) = unifyShapes2(mb,
          leftShape.slice(0, leftShape.length - 2),
          rightShape.slice(0, rightShape.length - 2))
        setup = Code(setup, unifiedSetup)
        shape = unifiedShape :+ leftShape(leftShape.length - 2) :+ rightShape.last
      }
    }

    setup = Code(setup,
      (lK cne rK).orEmpty(
        Code._fatal[Unit](const("Matrix dimensions incompatible: ").concat(lK.toS).concat(" ").concat(rK.toS))))

    (setup, shape)
  }
}

abstract class NDArrayEmitter(
   val nDims: Int,
   val outputShape: IndexedSeq[Value[Long]],
   val outputShapePType: PTuple,
   val outputElementPType: PType,
   val setupShape: Code[Unit],
   val setupMissing: Code[Unit] = Code._empty,
   val missing: Code[Boolean] = false) {

  def outputElement(elemMB: EmitMethodBuilder[_], idxVars: IndexedSeq[Value[Long]]): Code[_]

  def emit(mb: EmitMethodBuilder[_], targetType: PNDArray): EmitCode = {
    val outputShapeVariables = (0 until nDims).map(_ => mb.genFieldThisRef[Long]())

    val dataSrvb = new StagedRegionValueBuilder(mb, targetType.data.pType)

    val dataAddress: Code[Long] =
      Code(
        dataSrvb.start(targetType.numElements(outputShapeVariables.map(_.load()), mb).toI),
        emitLoops(mb, outputShapeVariables, dataSrvb),
        dataSrvb.end())

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

    val m = mb.genFieldThisRef[Boolean]()

    val fullSetup = Code(
      setupMissing,
      m := missing,
      m.mux(
        Code._empty,
        Code(
          setupShape,
          Code.foreach(0 until nDims)(index => outputShapeVariables(index) := outputShape(index)))))

    EmitCode(fullSetup, m,
      PCode(targetType, targetType.construct(0, 0, shapeBuilder, targetType.makeDefaultStridesBuilder(outputShapeVariables.map(_.load()), mb), dataAddress, mb)))
  }

  private def emitLoops(mb: EmitMethodBuilder[_], outputShapeVariables: IndexedSeq[Value[Long]], srvb: StagedRegionValueBuilder): Code[Unit] = {
    val innerMethod = mb.genEmitMethod[Unit]("ndaLoop")

    val idxVars = Array.tabulate(nDims) { _ => mb.genFieldThisRef[Long]() }.toFastIndexedSeq
    val storeElement = innerMethod.newLocal("nda_elem_out")(typeToTypeInfo(outputElementPType.virtualType))

    val body =
      Code(
        storeElement.storeAny(outputElement(innerMethod, idxVars)),
        srvb.addIRIntermediate(outputElementPType)(storeElement),
        srvb.advance()
      )
    val loops = idxVars.zipWithIndex.foldRight(body) { case ((dimVar, dimIdx), innerLoops) =>
      Code(
        dimVar := 0L,
        Code.whileLoop(dimVar < outputShapeVariables(dimIdx),
          innerLoops,
          dimVar := dimVar + 1L
        )
      )
    }
    innerMethod.emit(loops)
    innerMethod.invoke[Unit]()
  }
}
