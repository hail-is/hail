package is.hail.expr.ir

import is.hail.annotations.{MemoryBuffer, StagedRegionValueBuilder}
import is.hail.asm4s._
import is.hail.expr
import is.hail.expr.{TArray, TBoolean, TContainer, TFloat32, TFloat64, TInt32, TInt64, TStruct}
import is.hail.utils._
import org.objectweb.asm.Opcodes._
import org.objectweb.asm.tree._

import scala.language.existentials

object Compile {
  private def typeToTypeInfo(t: expr.Type): TypeInfo[_] = t match {
    case _: TInt32 => typeInfo[Int]
    case _: TInt64 => typeInfo[Long]
    case _: TFloat32 => typeInfo[Float]
    case _: TFloat64 => typeInfo[Double]
    case _: TBoolean => typeInfo[Boolean]
    case _ => typeInfo[Long] // reference types
  }

  type E = Env[(TypeInfo[_], Code[Boolean], Code[_])]

  def toCode(ir: IR, fb: FunctionBuilder[_]): (Code[Unit], Code[Boolean], Code[_]) = {
    toCode(ir, fb, Env.empty)
  }

  private def toCode(ir: IR, fb: FunctionBuilder[_], env: E): (Code[Unit], Code[Boolean], Code[_]) = {
    compile(ir, fb, env, new StagedBitSet(fb))
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
  private def tcoerce[T <: expr.Type](x: expr.Type): T = x.asInstanceOf[T]

  // the return value is interpreted as: (precompute, missingness, value)
  // rules:
  //  1. evaluate each returned Code[_] at most once
  //  2. evaluate precompute *on all static code-paths* leading to missingness or value
  //  3. gaurd the the evaluation of value by missingness
  //
  // JVM gotcha:
  //  a variable must be initialized on all static code-paths to its use (ergo defaultValue)
  private def compile(ir: IR, fb: FunctionBuilder[_], env: E, mb: StagedBitSet): (Code[Unit], Code[Boolean], Code[_]) = {
    val region = fb.getArg[MemoryBuffer](1).load()
    def compile(ir: IR, fb: FunctionBuilder[_] = fb, env: E = env, mb: StagedBitSet = mb): (Code[Unit], Code[Boolean], Code[_]) =
      Compile.compile(ir, fb, env, mb)
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
        val (dov, mv, vv) = compile(v)
        val cast = Casts.get(v.typ, typ)
        (dov, mv, cast(vv))

      case NA(typ) =>
        (Code._empty, const(true), defaultValue(typ))
      case IsNA(v) =>
        val (dov, mv, _) = compile(v)
        (dov, const(false), mv)
      case MapNA(name, value, body, typ) =>
        val vti = typeToTypeInfo(value.typ)
        val bti = typeToTypeInfo(typ)
        val mx = mb.newBit()
        val x = fb.newLocal(name)(vti).asInstanceOf[LocalRef[Any]]
        val mout = mb.newBit()
        val out = fb.newLocal(name)(bti).asInstanceOf[LocalRef[Any]]
        val (dovalue, mvalue, vvalue) = compile(value)
        val bodyenv = env.bind(name -> (vti, mx, x))
        val (dobody, mbody, vbody) = compile(body, env = bodyenv)
        val setup = Code(
          dovalue,
          mx := mvalue,
          mx.mux(
            Code(mout := true, out := defaultValue(typ)),
            Code(x := vvalue, dobody, mout := mbody, out := vbody)))

        (setup, mout, out)

      case expr.ir.If(cond, cnsq, altr, typ) =>
        val (docond, mcond, vcond) = compile(cond)
        val xvcond = mb.newBit()
        val out = fb.newLocal()(typeToTypeInfo(typ)).asInstanceOf[LocalRef[Any]]
        val mout = mb.newBit()
        val (docnsq, mcnsq, vcnsq) = compile(cnsq)
        val (doaltr, maltr, valtr) = compile(altr)
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

      case expr.ir.Let(name, value, body, typ) =>
        val vti = typeToTypeInfo(value.typ)
        val mx = mb.newBit()
        val x = fb.newLocal(name)(vti).asInstanceOf[LocalRef[Any]]
        val (dovalue, mvalue, vvalue) = compile(value)
        val bodyenv = env.bind(name -> (vti, mx, x))
        val (dobody, mbody, vbody) = compile(body, env = bodyenv)
        val setup = Code(
          dovalue,
          mx := mvalue,
          x := vvalue,
          dobody)

        (setup, mbody, vbody)
      case Ref(name, typ) =>
        val ti = typeToTypeInfo(typ)
        val (t, m, v) = env.lookup(name)
        assert(t == ti, s"$name type annotation, $typ, doesn't match typeinfo: $ti")
        (Code._empty, m, v)

      case ApplyBinaryPrimOp(op, l, r, typ) =>
        val (dol, ml, vl) = compile(l)
        val (dor, mr, vr) = compile(r)
        (Code(dol, dor),
          ml || mr,
          BinaryOp.compile(op, l.typ, r.typ, vl, vr))
      case ApplyUnaryPrimOp(op, x, typ) =>
        val (dox, mx, vx) = compile(x)
        (dox, mx, UnaryOp.compile(op, x.typ, vx))

      case MakeArray(args, typ) =>
        val srvb = new StagedRegionValueBuilder(fb, typ)
        val addElement = srvb.addIRIntermediate(typ.elementType)
        val mvargs = args.map(compile(_))
        present(Code(
          srvb.start(args.length, init = true),
          Code(mvargs.map { case (dov, m, v) =>
            Code(dov, m.mux(srvb.setMissing(), addElement(v)), srvb.advance())
          }: _*),
          srvb.offset))
      case x@MakeArrayN(len, elementType) =>
        val srvb = new StagedRegionValueBuilder(fb, x.typ)
        val (dolen, mlen, vlen) = compile(len)
        (dolen,
          mlen,
          Code(srvb.start(coerce[Int](vlen), init = true),
            srvb.offset))
      case ArrayRef(a, i, typ) =>
        val ti = typeToTypeInfo(typ)
        val tarray = TArray(typ)
        val ati = typeToTypeInfo(tarray).asInstanceOf[TypeInfo[Long]]
        val (doa, ma, va) = compile(a)
        val (doi, mi, vi) = compile(i)
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

        (setup, xmv, region.loadPrimitive(typ)(tarray.loadElement(region, xa, xi)))
      case ArrayMissingnessRef(a, i) =>
        val tarray = tcoerce[TArray](a.typ)
        val ati = typeToTypeInfo(tarray).asInstanceOf[TypeInfo[Long]]
        val (doa, ma, va) = compile(a)
        val (doi, mi, vi) = compile(i)
        present(Code(
          doa,
          doi,
          ma || mi || !tarray.isElementDefined(region, coerce[Long](va), coerce[Int](vi))))
      case ArrayLen(a) =>
        val (doa, ma, va) = compile(a)
        (doa, ma, TContainer.loadLength(region, coerce[Long](va)))
      case x@ArrayMap(a, name, body, elementTyp) =>
        val tin = a.typ.asInstanceOf[TArray]
        val tout = x.typ.asInstanceOf[TArray]
        val srvb = new StagedRegionValueBuilder(fb, tout)
        val addElement = srvb.addIRIntermediate(tout.elementType)
        val eti = typeToTypeInfo(elementTyp).asInstanceOf[TypeInfo[Any]]
        val xa = fb.newLocal[Long]("am_a")
        val xmv = mb.newBit()
        val xvv = fb.newLocal(name)(eti)
        val i = fb.newLocal[Int]("am_i")
        val len = fb.newLocal[Int]("am_len")
        val out = fb.newLocal[Long]("am_out")
        val bodyenv = env.bind(name -> (eti, xmv, xvv))
        val lmissing = new LabelNode()
        val lnonmissing = new LabelNode()
        val ltop = new LabelNode()
        val lnext = new LabelNode()
        val lend = new LabelNode()
        val (doa, ma, va) = compile(a)
        val (dobody, mbody, vbody) = compile(body, env = bodyenv)

        (doa, ma, Code(
          xa := coerce[Long](va),
          len := TContainer.loadLength(region, xa),
          i := 0,
          srvb.start(len, init = true),
          Code.whileLoop(i < len,
            xmv := !tin.isElementDefined(region, xa, i),
            xvv := xmv.mux(
              defaultValue(elementTyp),
              region.loadPrimitive(tin.elementType)(tin.loadElement(region, xa, i))),
            dobody,
            mbody.mux(srvb.setMissing(), addElement(vbody)),
            srvb.advance(),
            i := i + 1),
          srvb.offset))
      case ArrayFold(a, zero, name1, name2, body, typ) =>
        val tarray = a.typ.asInstanceOf[TArray]
        val tti = typeToTypeInfo(typ)
        val eti = typeToTypeInfo(tarray.elementType)
        val xma = mb.newBit()
        val xa = fb.newLocal[Long]("af_array")
        val xmv = mb.newBit()
        val xvv = fb.newLocal(name2)(eti).asInstanceOf[LocalRef[Any]]
        val xmout = mb.newBit()
        val xvout = fb.newLocal(name1)(tti).asInstanceOf[LocalRef[Any]]
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
        val (doa, ma, va) = compile(a)
        val (dozero, mzero, vzero) = compile(zero)
        val (dobody, mbody, vbody) = compile(body, env = bodyenv)
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
                  region.loadPrimitive(tarray.elementType)(tarray.loadElement(region, xa, i))),
                dobody,
                xmout := mbody,
                xvout := xmout.mux(defaultValue(typ), vbody),
                i := i + 1))))
        (setup, xmout, xvout)

      case MakeStruct(fields) =>
        val t = TStruct(fields.map { case (name, t, _) => (name, t) }: _*)
        val initializers = fields.map { case (_, t, v) => (t, compile(v)) }
        val srvb = new StagedRegionValueBuilder(fb, t)
        present(Code(
          srvb.start(true),
          Code(initializers.map { case (t, (dov, mv, vv)) =>
            Code(
              dov,
              mv.mux(srvb.setMissing(), srvb.addIRIntermediate(t)(vv)),
              srvb.advance()) }: _*),
          srvb.offset))
      case GetField(o, name, _) =>
        val t = o.typ.asInstanceOf[TStruct]
        val fieldIdx = t.fieldIdx(name)
        val (doo, mo, vo) = compile(o)
        val xmo = mb.newBit()
        val xo = fb.newLocal[Long]
        val setup = Code(
          doo,
          xmo := mo,
          xo := coerce[Long](xmo.mux(defaultValue(t), vo)))
        (setup,
          xmo || !t.isFieldDefined(region, xo, fieldIdx),
          region.loadPrimitive(t.fieldType(fieldIdx))(t.fieldOffset(xo, fieldIdx)))
      case GetFieldMissingness(o, name) =>
        val t = o.typ.asInstanceOf[TStruct]
        val fieldIdx = t.fieldIdx(name)
        val (doo, mo, vo) = compile(o)
        present(Code(doo, mo || !t.isFieldDefined(region, coerce[Long](vo), fieldIdx)))

      case AggIn(_) | AggMap(_, _, _, _) | AggSum(_, _) =>
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
