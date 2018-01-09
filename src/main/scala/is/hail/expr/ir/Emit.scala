package is.hail.expr.ir

import is.hail.annotations._
import is.hail.asm4s._
import is.hail.expr.types._
import is.hail.utils._
import org.objectweb.asm.tree._

import scala.language.existentials

object Emit {
  type E = Env[(TypeInfo[_], Code[Boolean], Code[_])]

  private[ir] def toCode(ir: IR, fb: FunctionBuilder[_]): (Code[Unit], Code[Boolean], Code[_]) = {
    toCode(ir, fb, Env.empty)
  }

  private[ir] def toCode(ir: IR, fb: FunctionBuilder[_], env: E): (Code[Unit], Code[Boolean], Code[_]) = {
    emit(ir, fb, env, new StagedBitSet(fb))
  }

  private[ir] def toCode(ir: IR, fb: FunctionBuilder[_], env: E, mb: StagedBitSet): (Code[Unit], Code[Boolean], Code[_]) = {
    emit(ir, fb, env, mb)
  }

  def apply(ir: IR, fb: FunctionBuilder[_]) {
    val (dov, mv, vv) = toCode(ir, fb)
    typeToTypeInfo(ir.typ) match { case ti: TypeInfo[t] =>
      fb.emit(Code(dov, mv.mux(
        Code._throw(Code.newInstance[RuntimeException, String]("cannot return empty")),
        coerce[t](vv))))
    }
  }

  private def present(x: Code[_]): (Code[Unit], Code[Boolean], Code[_]) =
    (Code._empty, const(false), x)

  // the return value is interpreted as: (precompute, missingness, value)
  // rules:
  //  1. evaluate each returned Code[_] at most once
  //  2. evaluate precompute *on all static code-paths* leading to missingness or value
  //  3. gaurd the the evaluation of value by missingness
  //
  // JVM gotcha:
  //  a variable must be initialized on all static code-paths to its use (ergo defaultValue)
  private def emit(ir: IR, fb: FunctionBuilder[_], env: E, mb: StagedBitSet): (Code[Unit], Code[Boolean], Code[_]) = {
    val region = fb.getArg[Region](1).load()
    def emit(ir: IR, fb: FunctionBuilder[_] = fb, env: E = env, mb: StagedBitSet = mb): (Code[Unit], Code[Boolean], Code[_]) =
      Emit.emit(ir, fb, env, mb)
    ir match {
      case I32(x) =>
        present(const(x))
      case I64(x) =>
        present(const(x))
      case F32(x) =>
        present(const(x))
      case F64(x) =>
        present(const(x))
      case True() =>
        present(const(true))
      case False() =>
        present(const(false))

      case Cast(v, typ) =>
        val (dov, mv, vv) = emit(v)
        val cast = Casts.get(v.typ, typ)
        (dov, mv, cast(vv))

      case NA(typ) =>
        (Code._empty, const(true), defaultValue(typ))
      case IsNA(v) =>
        val (dov, mv, _) = emit(v)
        (dov, const(false), mv)
      case MapNA(name, value, body, typ) =>
        val vti = typeToTypeInfo(value.typ)
        val bti = typeToTypeInfo(typ)
        val mx = mb.newBit()
        val x = coerce[Any](fb.newLocal(name)(vti))
        val mout = mb.newBit()
        val out = coerce[Any](fb.newLocal(name)(bti))
        val (dovalue, mvalue, vvalue) = emit(value)
        val bodyenv = env.bind(name -> (vti, mx, x))
        val (dobody, mbody, vbody) = emit(body, env = bodyenv)
        val setup = Code(
          dovalue,
          mx := mvalue,
          mx.mux(
            Code(mout := true, out := defaultValue(typ)),
            Code(x := vvalue, dobody, mout := mbody, out := vbody)))

        (setup, mout, out)

      case If(cond, cnsq, altr, typ) =>
        val (docond, mcond, vcond) = emit(cond)
        val xvcond = mb.newBit()
        val out = coerce[Any](fb.newLocal()(typeToTypeInfo(typ)))
        val mout = mb.newBit()
        val (docnsq, mcnsq, vcnsq) = emit(cnsq)
        val (doaltr, maltr, valtr) = emit(altr)
        val setup = Code(
          docond,
          mcond.mux(
            Code(mout := true, out := defaultValue(typ)),
            Code(
              xvcond := coerce[Boolean](vcond),
              coerce[Boolean](xvcond).mux(
                Code(docnsq, mout := mcnsq, out := vcnsq),
                Code(doaltr, mout := maltr, out := valtr)))))

        (setup, mout, out)

      case Let(name, value, body, typ) =>
        val vti = typeToTypeInfo(value.typ)
        val mx = mb.newBit()
        val x = coerce[Any](fb.newLocal(name)(vti))
        val (dovalue, mvalue, vvalue) = emit(value)
        val bodyenv = env.bind(name -> (vti, mx, x))
        val (dobody, mbody, vbody) = emit(body, env = bodyenv)
        val setup = Code(
          dovalue,
          mx := mvalue,
          x := vvalue,
          dobody)

        (setup, mbody, vbody)
      case Ref(name, typ) =>
        val ti = typeToTypeInfo(typ)
        val (t, m, v) = env.lookup(name)
        assert(t == ti, s"$name type annotation, $typ, $t doesn't match typeinfo: $ti")
        (Code._empty, m, v)

      case ApplyBinaryPrimOp(op, l, r, typ) =>
        val (dol, ml, vl) = emit(l)
        val (dor, mr, vr) = emit(r)
        (Code(dol, dor),
          ml || mr,
          BinaryOp.emit(op, l.typ, r.typ, vl, vr))
      case ApplyUnaryPrimOp(op, x, typ) =>
        val (dox, mx, vx) = emit(x)
        (dox, mx, UnaryOp.emit(op, x.typ, vx))

      case MakeArray(args, typ) =>
        val srvb = new StagedRegionValueBuilder(fb, typ)
        val addElement = srvb.addIRIntermediate(typ.elementType)
        val mvargs = args.map(emit(_))
        present(Code(
          srvb.start(args.length, init = true),
          Code(mvargs.map { case (dov, m, v) =>
            Code(dov, m.mux(srvb.setMissing(), addElement(v)), srvb.advance())
          }: _*),
          srvb.offset))
      case x@MakeArrayN(len, elementType) =>
        val srvb = new StagedRegionValueBuilder(fb, x.typ)
        val (dolen, mlen, vlen) = emit(len)
        (dolen,
          mlen,
          Code(srvb.start(coerce[Int](vlen), init = true),
            srvb.offset))
      case ArrayRef(a, i, typ) =>
        val ti = typeToTypeInfo(typ)
        val tarray = TArray(typ)
        val ati = coerce[Long](typeToTypeInfo(tarray))
        val (doa, ma, va) = emit(a)
        val (doi, mi, vi) = emit(i)
        val xma = mb.newBit()
        val xa = fb.newLocal()(ati)
        val xi = fb.newLocal[Int]
        val xmi = mb.newBit()
        val xmv = mb.newBit()
        val setup = Code(
          doa,
          xma := ma,
          xa := coerce[Long](xma.mux(defaultValue(tarray), va)),
          doi,
          xmi := mi,
          xi := coerce[Int](xmi.mux(defaultValue(TInt32()), vi)),
          xmv := xma || xmi || !tarray.isElementDefined(region, xa, xi))

        (setup, xmv, region.loadIRIntermediate(typ)(tarray.loadElement(region, xa, xi)))
      case ArrayMissingnessRef(a, i) =>
        val tarray = coerce[TArray](a.typ)
        val ati = coerce[Long](typeToTypeInfo(tarray))
        val (doa, ma, va) = emit(a)
        val (doi, mi, vi) = emit(i)
        present(Code(
          doa,
          doi,
          ma || mi || !tarray.isElementDefined(region, coerce[Long](va), coerce[Int](vi))))
      case ArrayLen(a) =>
        val (doa, ma, va) = emit(a)
        (doa, ma, TContainer.loadLength(region, coerce[Long](va)))
      case x@ArrayMap(a, name, body, elementTyp) =>
        val tin = coerce[TArray](a.typ)
        val tout = x.typ
        val srvb = new StagedRegionValueBuilder(fb, tout)
        val addElement = srvb.addIRIntermediate(tout.elementType)
        val etiin = coerce[Any](typeToTypeInfo(tin.elementType))
        val xa = fb.newLocal[Long]("am_a")
        val xmv = mb.newBit()
        val xvv = fb.newLocal(name)(etiin)
        val i = fb.newLocal[Int]("am_i")
        val len = fb.newLocal[Int]("am_len")
        val out = fb.newLocal[Long]("am_out")
        val bodyenv = env.bind(name -> (etiin, xmv, xvv))
        val lmissing = new LabelNode()
        val lnonmissing = new LabelNode()
        val ltop = new LabelNode()
        val lnext = new LabelNode()
        val lend = new LabelNode()
        val (doa, ma, va) = emit(a)
        val (dobody, mbody, vbody) = emit(body, env = bodyenv)

        (doa, ma, Code(
          xa := coerce[Long](va),
          len := TContainer.loadLength(region, xa),
          i := 0,
          srvb.start(len, init = true),
          Code.whileLoop(i < len,
            xmv := !tin.isElementDefined(region, xa, i),
            xvv := xmv.mux(
              defaultValue(tin.elementType),
              region.loadIRIntermediate(tin.elementType)(tin.loadElement(region, xa, i))),
            dobody,
            mbody.mux(srvb.setMissing(), addElement(vbody)),
            srvb.advance(),
            i := i + 1),
          srvb.offset))
      case ArrayFold(a, zero, name1, name2, body, typ) =>
        val tarray = coerce[TArray](a.typ)
        val tti = typeToTypeInfo(typ)
        val eti = typeToTypeInfo(tarray.elementType)
        val xma = mb.newBit()
        val xa = fb.newLocal[Long]("af_array")
        val xmv = mb.newBit()
        val xvv = coerce[Any](fb.newLocal(name2)(eti))
        val xmout = mb.newBit()
        val xvout = coerce[Any](fb.newLocal(name1)(tti))
        val i = fb.newLocal[Int]("af_i")
        val len = fb.newLocal[Int]("af_len")
        val bodyenv = env.bind(
          name1 -> (tti, xmout, xvout.load()),
          name2 -> (eti, xmv, xvv.load()))
        val lmissing = new LabelNode()
        val lnonmissing = new LabelNode()
        val ltop = new LabelNode()
        val lnext = new LabelNode()
        val lend = new LabelNode()
        val (doa, ma, va) = emit(a)
        val (dozero, mzero, vzero) = emit(zero)
        val (dobody, mbody, vbody) = emit(body, env = bodyenv)
        val setup = Code(
          doa,
          ma.mux(
            Code(xmout := true, xvout := defaultValue(typ)),
            Code(
              xa := coerce[Long](va),
              len := TContainer.loadLength(region, xa),
              i := 0,
              dozero,
              xmout := mzero,
              xvout := xmout.mux(defaultValue(typ), vzero),
              Code.whileLoop(i < len,
                xmv := !tarray.isElementDefined(region, xa, i),
                xvv := xmv.mux(
                  defaultValue(tarray.elementType),
                  region.loadIRIntermediate(tarray.elementType)(tarray.loadElement(region, xa, i))),
                dobody,
                xmout := mbody,
                xvout := xmout.mux(defaultValue(typ), vbody),
                i := i + 1))))
        (setup, xmout, xvout)

      case x@MakeStruct(fields, _) =>
        val initializers = fields.map { case (_, v) => (v.typ, emit(v)) }
        val srvb = new StagedRegionValueBuilder(fb, x.typ)
        present(Code(
          srvb.start(init = true),
          Code(initializers.map { case (t, (dov, mv, vv)) =>
            Code(
              dov,
              mv.mux(srvb.setMissing(), srvb.addIRIntermediate(t)(vv)),
              srvb.advance()) }: _*),
          srvb.offset))
      case GetField(o, name, _) =>
        val t = coerce[TStruct](o.typ)
        val fieldIdx = t.fieldIdx(name)
        val (doo, mo, vo) = emit(o)
        val xmo = mb.newBit()
        val xo = fb.newLocal[Long]
        val setup = Code(
          doo,
          xmo := mo,
          xo := coerce[Long](xmo.mux(defaultValue(t), vo)))
        (setup,
          xmo || !t.isFieldDefined(region, xo, fieldIdx),
          region.loadIRIntermediate(t.fieldType(fieldIdx))(t.fieldOffset(xo, fieldIdx)))
      case GetFieldMissingness(o, name) =>
        val t = coerce[TStruct](o.typ)
        val fieldIdx = t.fieldIdx(name)
        val (doo, mo, vo) = emit(o)
        present(Code(doo, mo || !t.isFieldDefined(region, coerce[Long](vo), fieldIdx)))

      case _: AggIn | _: AggMap | _: AggFilter | _: AggFlatMap | _: AggSum =>
        throw new RuntimeException(s"Aggregations must be extracted with ExtractAggregators before compilation: $ir")

      case In(i, typ) =>
        (Code._empty, fb.getArg[Boolean](i*2 + 3), fb.getArg(i*2 + 2)(typeToTypeInfo(typ)))
      case InMissingness(i) =>
        present(fb.getArg[Boolean](i*2 + 3))
      case Die(m) =>
        present(Code._throw(Code.newInstance[RuntimeException, String](m)))
    }
  }
}
