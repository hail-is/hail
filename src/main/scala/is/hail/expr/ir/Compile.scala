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
  private def defaultValue(t: expr.Type): Code[_] = t match {
    case TBoolean => false
    case TInt32 => 0
    case TInt64 => 0L
    case TFloat32 => 0.0f
    case TFloat64 => 0.0
    case _ => 0L // reference types
  }
  private def typeToTypeInfo(t: expr.Type): TypeInfo[_] = t match {
    case TInt32 => typeInfo[Int]
    case TInt64 => typeInfo[Long]
    case TFloat32 => typeInfo[Float]
    case TFloat64 => typeInfo[Double]
    case TBoolean => typeInfo[Boolean]
    case _ => typeInfo[Long] // reference types
  }

  type E = Env[(TypeInfo[_], Code[Boolean], Code[_])]

  def apply(ir: IR, fb: FunctionBuilder[_]) {
    apply(ir, fb, new Env())
  }

  def apply(ir: IR, fb: FunctionBuilder[_], env: E) {
    fb.emit(compile(ir, fb, env, new StagedBitSet(fb))._2)
  }

  private def present(x: Code[_]) = (const(false), x)
  private def coerce[T](x: Code[_]): Code[T] = x.asInstanceOf[Code[T]]
  private def tcoerce[T <: expr.Type](x: expr.Type): T = x.asInstanceOf[T]

  def compile(ir: IR, fb: FunctionBuilder[_], env: E, mb: StagedBitSet): (Code[Boolean], Code[_]) = {
    val region = fb.getArg[MemoryBuffer](1).load()
    def compile(ir: IR, fb: FunctionBuilder[_] = fb, env: E = env, mb: StagedBitSet = mb): (Code[Boolean], Code[_]) =
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

      case NA(typ) =>
        (const(true), defaultValue(typ))
      case IsNA(v) =>
        present(compile(v)._1)
      case MapNA(name, value, body, typ) =>
        val vti = typeToTypeInfo(value.typ)
        val bti = typeToTypeInfo(typ)
        val (mvalue, vvalue) = compile(value)
        val mx = mb.newBit()
        fb.emit(mx := mvalue)
        val x = fb.newLocal(name)(vti).asInstanceOf[LocalRef[Any]]
        fb.emit(x := mx.mux(defaultValue(value.typ), vvalue))
        val bodyenv = env.bind(name -> (vti, mx, x))
        val (mbody, vbody) = compile(body, env = bodyenv)
        (mvalue || mbody, mvalue.mux(defaultValue(typ), vbody))

      case expr.ir.If(cond, cnsq, altr, typ) =>
        val (mcond, vcond) = compile(cond)
        val xvcond = mb.newBit()
        fb.emit(xvcond := coerce[Boolean](vcond))
        val (mcnsq, vcnsq) = compile(cnsq)
        val (maltr, valtr) = compile(altr)

        (mcond || (xvcond && mcnsq) || (!xvcond && maltr),
          xvcond.asInstanceOf[Code[Boolean]].mux(vcnsq, valtr))

      case expr.ir.Let(name, value, body, typ) =>
        val vti = typeToTypeInfo(value.typ)
        fb.newLocal(name)(vti) match { case x: LocalRef[v] =>
          val (mvalue, vvalue) = compile(value)
          val xmvalue = mb.newBit()
          fb.emit(xmvalue := mvalue)
          fb.emit(x := coerce[v](vvalue))
          val bodyenv = env.bind(name -> (vti, xmvalue, x))
          compile(body, env = bodyenv)
        }
      case Ref(name, typ) =>
        val ti = typeToTypeInfo(typ)
        val (t, m, v) = env.lookup(name)
        assert(t == ti, s"$name type annotation, $typ, doesn't match typeinfo: $ti")
        (m, v)

      case ApplyPrimitive(op, args, typ) =>
        val typs = args.map(_.typ)
        val (margs, vargs) = args.map(compile(_)).unzip
        val m = if (margs.isEmpty) const(false) else margs.reduce(_ || _)
        (m, Primitives.lookup(op, typs, vargs))
      case LazyApplyPrimitive(op, args, typ) =>
        ???
      case expr.ir.Lambda(names, body, typ) =>
        ???

      case MakeArray(args, typ) =>
        val srvb = new StagedRegionValueBuilder(fb, typ)
        val addElement = srvb.addAnnotation(typ.elementType)
        val mvargs = args.map(compile(_))
        present(Code(
          srvb.start(args.length, init = true),
          Code(mvargs.map { case (m, v) =>
            Code(m.mux(srvb.setMissing(), addElement(v)), srvb.advance())
          }: _*),
          srvb.offset))
      case x@MakeArrayN(len, elementType) =>
        val srvb = new StagedRegionValueBuilder(fb, x.typ)
        val (mlen, vlen) = compile(len)
        (mlen, mlen.mux(
          defaultValue(x.typ),
          Code(srvb.start(coerce[Int](vlen), init = true),
            srvb.offset)))
      case ArrayRef(a, i, typ) =>
        val ti = typeToTypeInfo(typ)
        val tarray = TArray(typ)
        val ati = typeToTypeInfo(tarray).asInstanceOf[TypeInfo[Long]]
        val (ma, va) = compile(a)
        val (mi, vi) = compile(i)
        val xma = mb.newBit()
        val xa = fb.newLocal()(ati)
        val xi = fb.newLocal[Int]
        val xmi = mb.newBit()
        val xmv = mb.newBit()
        fb.emit(Code(
          xma := ma,
          xa := coerce[Long](xma.mux(defaultValue(tarray), va)),
          xmi := mi,
          xi := coerce[Int](xmi.mux(defaultValue(TInt32), vi)),
          xmv := xma || xmi || !tarray.isElementDefined(region, xa, xi)))

        (xmv, xmv.mux(defaultValue(typ),
          region.loadAnnotation(typ)(tarray.loadElement(region, xa, xi))))
      case ArrayMissingnessRef(a, i) =>
        val tarray = tcoerce[TArray](a.typ)
        val ati = typeToTypeInfo(tarray).asInstanceOf[TypeInfo[Long]]
        val (ma, va) = compile(a)
        val (mi, vi) = compile(i)
        val xma = mb.newBit()
        val xa = fb.newLocal()(ati)
        val xi = fb.newLocal[Int]
        val xmi = mb.newBit()
        val xmv = mb.newBit()
        fb.emit(Code(
          xma := ma,
          xa := coerce[Long](xma.mux(defaultValue(tarray), va)),
          xmi := mi,
          xi := coerce[Int](xmi.mux(defaultValue(TInt32), vi))))
        present(xma || xmi || !tarray.isElementDefined(region, xa, xi))
      case ArrayLen(a) =>
        val (ma, va) = compile(a)
        (ma, TContainer.loadLength(region, coerce[Long](va)))
      case x@ArrayMap(a, Lambda(Array((name,_)), body, _), elementTyp) =>
        val tin = a.typ.asInstanceOf[TArray]
        val tout = x.typ.asInstanceOf[TArray]
        val srvb = new StagedRegionValueBuilder(fb, tout)
        val addElement = srvb.addAnnotation(tout.elementType)
        typeToTypeInfo(elementTyp) match { case eti: TypeInfo[t] =>
          val xma = mb.newBit()
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
          val (ma, va) = compile(a)
          fb.emit(xma := ma)
          fb.emit(xvv := coerce[t](defaultValue(elementTyp)))
          fb.emit(out := coerce[Long](defaultValue(tout)))
          xma.toConditional.emitConditional(fb.l, lmissing, lnonmissing)
          fb.emit(Code(
            lnonmissing,
            xa := coerce[Long](va),
            len := TContainer.loadLength(region, xa),
            i := 0,
            srvb.start(len, init = true),
            ltop))
          (i < len).toConditional.emitConditional(fb.l, lnext, lend)
          fb.emit(Code(
            lnext,
            xmv := !tin.isElementDefined(region, xa, i),
            xvv := coerce[t](xmv.mux(
              defaultValue(elementTyp),
              region.loadAnnotation(tin.elementType)(tin.loadElement(region, xa, i))))))
          val (mbody, vbody) = compile(body, env = bodyenv)
          fb.emit(Code(
            mbody.mux(
              srvb.setMissing(),
              addElement(vbody)),
            srvb.advance(),
            i := i + 1,
            new JumpInsnNode(GOTO, ltop),
            lend,
            out := srvb.offset,
            lmissing))

          (xma, out.load())
        }
      case ArrayMap(_, _, _) =>
        throw new UnsupportedOperationException(s"bad arraymap $ir")
      case ArrayFold(a, zero, Lambda(Array((name1, _), (name2, _)), body, _), typ) =>
        val tarray = a.typ.asInstanceOf[TArray]
        assert(tarray != null, s"tarray is null! $ir")
        (typeToTypeInfo(typ), typeToTypeInfo(tarray.elementType)) match { case (tti: TypeInfo[t], uti: TypeInfo[u]) =>
          val xma = mb.newBit()
          val xa = fb.newLocal[Long]("af_array")
          val xmv = mb.newBit()
          val xvv = fb.newLocal(name2)(uti)
          val xmout = mb.newBit()
          val xvout = fb.newLocal(name1)(tti)
          val i = fb.newLocal[Int]("af_i")
          val len = fb.newLocal[Int]("af_len")
          val bodyenv = env.bind(
            name1 -> (tti, xmout, xvout.load()),
            name2 -> (uti, xmv, xvv.load()))
          val lmissing = new LabelNode()
          val lnonmissing = new LabelNode()
          val ltop = new LabelNode()
          val lnext = new LabelNode()
          val lend = new LabelNode()
          val (ma, va) = compile(a)
          fb.emit(xma := ma)
          fb.emit(xvout := coerce[t](defaultValue(typ)))
          fb.emit(xvv := coerce[u](defaultValue(tarray.elementType)))
          xma.toConditional.emitConditional(fb.l, lmissing, lnonmissing)
          fb.emit(Code(
            lnonmissing,
            xa := coerce[Long](va),
            len := TContainer.loadLength(region, xa),
            i := 0))
          val (mzero, vzero) = compile(zero)
          fb.emit(Code(
            xmout := mzero,
            xvout := coerce[t](xmout.mux(defaultValue(typ), vzero)),
            ltop))
          (i < len).toConditional.emitConditional(fb.l, lnext, lend)
          fb.emit(Code(
            lnext,
            xmv := !tarray.isElementDefined(region, xa, i),
            xvv := coerce[u](xmv.mux(
              defaultValue(tarray.elementType),
              region.loadAnnotation(tarray.elementType)(tarray.loadElement(region, xa, i))))))
          val (mbody, vbody) = compile(body, env = bodyenv)
          fb.emit(Code(
            xmout := mbody,
            xvout := coerce[t](xmout.mux(defaultValue(typ), vbody)),
            i := i + 1,
            new JumpInsnNode(GOTO, ltop),
            lend,
            lmissing))
          (xmout, xvout)
        }
      case ArrayFold(_,  _, _, _) =>
        throw new UnsupportedOperationException(s"bad arrayfold $ir")

      case MakeStruct(fields) =>
        val t = TStruct(fields.map { case (name, t, _) => (name, t) }: _*)
        val initializers = fields.map { case (_, t, v) => (t, compile(v)) }
        val srvb = new StagedRegionValueBuilder(fb, t)
        present(Code(
          srvb.start(false),
          Code(initializers.map { case (t, (mv, vv)) =>
            Code(
              mv.mux(srvb.setMissing(), srvb.addAnnotation(t)(vv)),
              srvb.advance()) }: _*),
          srvb.offset))
      case GetField(o, name, _) =>
        val t = o.typ.asInstanceOf[TStruct]
        val fieldIdx = t.fieldIdx(name)
        val (mo, vo) = compile(o)
        val xmo = mb.newBit()
        val xo = fb.newLocal[Long]
        fb.emit(xmo := mo)
        fb.emit(xo := coerce[Long](xmo.mux(defaultValue(t), vo)))
        (xmo || !t.isFieldDefined(region, xo, fieldIdx),
          region.loadAnnotation(t)(t.fieldOffset(xo, fieldIdx)))
      case GetFieldMissingness(o, name) =>
        val t = o.typ.asInstanceOf[TStruct]
        val fieldIdx = t.fieldIdx(name)
        val (mo, vo) = compile(o)
        val xmo = mb.newBit()
        val xo = fb.newLocal[Long]
        fb.emit(xmo := mo)
        fb.emit(xo := coerce[Long](xmo.mux(defaultValue(t), vo)))
        (xmo, !t.isFieldDefined(region, xo, fieldIdx))

      case In(i, typ) =>
        (fb.getArg[Boolean](i*2 + 3), fb.getArg(i*2 + 2)(typeToTypeInfo(typ)))
      case InMissingness(i) =>
        present(fb.getArg[Boolean](i*2 + 3))
      case Out(v) =>
        val (mv, vv) = compile(v)
        typeToTypeInfo(v.typ) match { case ti: TypeInfo[t] =>
          present(mv.mux(
            Code._throw(Code.newInstance[RuntimeException, String]("cannot return empty")),
            Code._return(coerce[t](vv))(ti)))
        }
      case Die(m) =>
        present(Code._throw(Code.newInstance[RuntimeException, String](m)))
    }
  }
}
