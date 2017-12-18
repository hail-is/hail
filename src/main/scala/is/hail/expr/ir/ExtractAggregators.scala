package is.hail.expr.ir

import is.hail.annotations._
import is.hail.annotations.aggregators._
import is.hail.asm4s._
import is.hail.expr.{RegionValueAggregator, TAggregable, TArray, TStruct, Type}
import is.hail.utils._

import scala.language.{existentials, postfixOps}

object ExtractAggregators {

  private case class IRAgg(in: In, agg: ApplyAggOp) { }

  def apply(ir: IR, tAggIn: TAggregable, aggFb: FunctionBuilder[_]): (IR, TStruct, Array[RegionValueAggregator]) = {
    val (ir2, aggs) = extract(ir, tAggIn)
    val rvas = emitAgg(aggs.map(_.agg), tAggIn, aggFb)

    val fields = aggs.map(_.agg.typ).zipWithIndex.map { case (t, i) => i.toString -> t }
    val resultStruct = TStruct(fields: _*)
    // mutate the type of the input IR node now that we know what the combined
    // struct's type is
    aggs.foreach(_.in.typ = resultStruct)

    (ir2, resultStruct, rvas)
  }

  private def extract(ir: IR, tAggIn: TAggregable): (IR, Array[IRAgg]) = {
    val ab = new ArrayBuilder[IRAgg]()
    val ir2 = extract(ir, ab, tAggIn)
    (ir2, ab.result())
  }

  private def extract(ir: IR, ab: ArrayBuilder[IRAgg], tAggIn: TAggregable): IR = {
    def extract(ir: IR): IR = this.extract(ir, ab, tAggIn)
    ir match {
      case Ref(name, typ) =>
        assert(typ.isRealizable)
        ir
      case _: AggIn | _: AggMap | _: AggFilter | _: AggFlatMap =>
        throw new RuntimeException(s"Aggregable manipulations must appear inside the lexical scope of an Aggregation: $ir")
      case x: ApplyAggOp =>
        val in = In(0, null)

        ab += IRAgg(in, x)

        GetField(in, (ab.length - 1).toString(), x.typ)
      case _ => Recur(extract)(ir)
    }
  }

  private def loadRegion(fb: FunctionBuilder[_]): Code[Region] =
    fb.getArg[Region](1)

  private def loadAggArray(fb: FunctionBuilder[_]): Code[Array[RegionValueAggregator]] =
    fb.getArg[Array[RegionValueAggregator]](2)

  private def loadElement(t: Type, fb: FunctionBuilder[_]): Code[_] =
    fb.getArg(3)(typeToTypeInfo(t))

  private def isElementMissing(fb: FunctionBuilder[_]): Code[Boolean] =
    fb.getArg[Boolean](4)

 private val nonScopeArguments = 5 // this, Region, Array[Aggregator], ElementType, Boolean

  private def emitAgg(irs: Array[ApplyAggOp], tAggIn: TAggregable, fb: FunctionBuilder[_]): Array[RegionValueAggregator] = {
    val scopeBindings = tAggIn.bindings.zipWithIndex
      .map { case ((n, t), i) => n -> ((
        typeToTypeInfo(t),
        fb.getArg[Boolean](nonScopeArguments + i*2 + 1).load(),
        fb.getArg(nonScopeArguments + i*2)(typeToTypeInfo(t)).load())) }

    irs.zipWithIndex.map {
      case (ApplyAggNullaryOp(a, op, typ), i) =>
        val agg = AggOp.getNullary(op, a.typ.asInstanceOf[TAggregable].elementType)
        fb.emit(emitAgg2(a, tAggIn, agg.seqOp(loadAggArray(fb)(i), _, _),
          fb, new StagedBitSet(fb), Env.empty.bind(scopeBindings: _*)))
        agg.aggregator
      case (ApplyAggTernaryOp(a, op, arg1, arg2, arg3, typ), i) =>
        val agg = AggOp.getTernary(op, arg1.typ, arg2.typ, arg3.typ, a.typ.asInstanceOf[TAggregable].elementType)
        fb.emit(emitAgg2(a, tAggIn, agg.seqOp(loadAggArray(fb)(i), _, _),
          fb, new StagedBitSet(fb), Env.empty.bind(scopeBindings: _*)))

        val constfb = FunctionBuilder.functionBuilder[Region, RegionValueAggregator]
        val (doarg1, varg1, marg1) = Emit.toCode(arg1, constfb)
        val (doarg2, varg2, marg2) = Emit.toCode(arg2, constfb)
        val (doarg3, varg3, marg3) = Emit.toCode(arg3, constfb)
        constfb.emit(Code(
          doarg1,
          doarg2,
          doarg3,
          agg.aggregator.asInstanceOf[(Code[Any], Code[Boolean], Code[Any], Code[Boolean], Code[Any], Code[Boolean]) => Code[RegionValueAggregator]](
            varg1, coerce[Boolean](marg1), varg2, coerce[Boolean](marg2), varg3, coerce[Boolean](marg3))))
        constfb.result()()(Region())
    }
  }

  private def emitAgg2(
    ir: IR,
    tAggIn: TAggregable,
    continuation: (Code[_], Code[Boolean]) => Code[Unit],
    fb: FunctionBuilder[_],
    mb: StagedBitSet,
    env: Emit.E): Code[Unit] = {

    def emitAgg2(ir: IR)(continuation: (Code[_], Code[Boolean]) => Code[Unit]): Code[Unit] =
      this.emitAgg2(ir, tAggIn, continuation, fb, mb, env)
    val region = loadRegion(fb)
    ir match {
      case AggIn(typ) =>
        assert(tAggIn == typ)
        continuation(loadElement(tAggIn.elementType, fb), isElementMissing(fb))
      case AggMap(a, name, body, typ) =>
        val tA = a.typ.asInstanceOf[TAggregable]
        val tElement = tA.elementType
        val elementTi = typeToTypeInfo(tElement)
        val x = fb.newLocal()(elementTi).asInstanceOf[LocalRef[Any]]
        val mx = mb.newBit
        val (dobody, mbody, vbody) = Emit.toCode(body, fb, env = env.bind(name, (elementTi, mx.load(), x.load())), mb)
        emitAgg2(a) { (v, mv) =>
          Code(
            mx := mv,
            x := mv.mux(defaultValue(tElement), v),
            dobody,
            continuation(vbody, mbody)) }
      case AggFilter(a, name, body, typ) =>
        val tA = a.typ.asInstanceOf[TAggregable]
        val tElement = tA.elementType
        val elementTi = typeToTypeInfo(tElement)
        val x = fb.newLocal()(elementTi).asInstanceOf[LocalRef[Any]]
        val mx = mb.newBit
        val (dobody, mbody, vbody) = Emit.toCode(body, fb, env = env.bind(name, (elementTi, mx.load(), x.load())), mb)
        emitAgg2(a) { (v, mv) =>
          Code(
            mx := mv,
            x := mv.mux(defaultValue(tElement), v),
            dobody,
            // missing is false
            (!mbody && coerce[Boolean](vbody)).mux(continuation(x, mx), Code._empty)) }
      case AggFlatMap(a, name, body, typ) =>
        val tA = a.typ.asInstanceOf[TAggregable]
        val tElement = tA.elementType
        val elementTi = typeToTypeInfo(tElement)
        val tArray = body.typ.asInstanceOf[TArray]
        val x = fb.newLocal()(elementTi).asInstanceOf[LocalRef[Any]]
        val arr = fb.newLocal[Long]
        val len = fb.newLocal[Int]
        val i = fb.newLocal[Int]
        val mx = mb.newBit
        val (dobody, mbody, vbody) = Emit.toCode(body, fb, env = env.bind(name, (elementTi, mx.load(), x.load())), mb)
        emitAgg2(a) { (v, mv) =>
          Code(
            mx := mv,
            x := mv.mux(defaultValue(tElement), v),
            dobody,
            mbody.mux(
              Code._empty,
              Code(
                arr := coerce[Long](vbody),
                i := 0,
                len := tArray.loadLength(region, arr),
                Code.whileLoop(i < len,
                  continuation(
                    region.loadIRIntermediate(tArray.elementType)(tArray.loadElement(region, arr, i)),
                    tArray.isElementMissing(region, arr, i)),
                  i ++)))) }
      case _: ApplyAggNullaryOp | _: ApplyAggTernaryOp =>
        throw new RuntimeException(s"No nested aggregations allowed: $ir")
      case In(_, _) | InMissingness(_) =>
        throw new RuntimeException(s"No inputs may be referenced inside an aggregator: $ir")
      case _ =>
        throw new RuntimeException(s"Expected an aggregator, but found: $ir")
    }
  }
}
