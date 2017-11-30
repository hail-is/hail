package is.hail.expr.ir

import is.hail.annotations._
import is.hail.asm4s._
import is.hail.expr.{Type, TArray, TBoolean, TContainer, TFloat32, TFloat64, TInt32, TInt64, TStruct, TSet}
import is.hail.utils._
import org.objectweb.asm.Opcodes._
import org.objectweb.asm.tree._

import scala.language.existentials
import scala.language.postfixOps

object Compile {
  private def defaultValue(t: Type): Code[_] = t match {
    case _: TBoolean => false
    case _: TInt32 => 0
    case _: TInt64 => 0L
    case _: TFloat32 => 0.0f
    case _: TFloat64 => 0.0
    case _ => 0L // reference types
  }

  private def toPointer(region: Code[MemoryBuffer], t: Type): Code[_] => Code[Long] = t match {
    case _: TBoolean => v => region.appendByte(coerce[Boolean](v).toI.toB)
    case _: TInt32 => v => region.appendInt32(coerce[Int](v))
    case _: TInt64 => v => region.appendInt64(coerce[Long](v))
    case _: TFloat32 => v => region.appendFloat32(coerce[Float](v))
    case _: TFloat64 => v => region.appendFloat64(coerce[Double](v))
    case _ => v => coerce[Long](v) // reference types are already pointers
  }

  type E = Env[(TypeInfo[_], Code[Boolean], Code[_])]

  def apply(ir: IR, fb: FunctionBuilder[_]) {
    apply(ir, fb, new Env())
  }

  def apply(ir: IR, fb: FunctionBuilder[_], env: E) {
    val (dov, mv, vv) = compile(ir, fb, env, new StagedBitSet(fb))
    TypeToTypeInfo(ir.typ) match { case ti: TypeInfo[t] =>
      fb.emit(Code(dov, mv.mux(
        Code._throw(Code.newInstance[RuntimeException, String]("cannot return empty")),
        coerce[t](vv))))
    }
  }

  private def present(x: Code[_]): (Code[Unit], Code[Boolean], Code[_]) =
    (Code._empty, const(false), x)
  private def tcoerce[T <: Type](x: Type): T = x.asInstanceOf[T]

  // the return value is interpreted as: (precompute, missingness, value)
  // rules:
  //  1. evaluate each returned Code[_] at most once
  //  2. evaluate precompute *on all static code-paths* leading to missingness or value
  //  3. gaurd the the evaluation of value by missingness
  //
  // JVM gotcha:
  //  a variable must be initialized on all static code-paths to its use (ergo defaultValue)
  def compile(ir: IR, fb: FunctionBuilder[_], env: E, mb: StagedBitSet): (Code[Unit], Code[Boolean], Code[_]) = {
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
        val vti = TypeToTypeInfo(value.typ)
        val bti = TypeToTypeInfo(typ)
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

      case If(cond, cnsq, altr, typ) =>
        val (docond, mcond, vcond) = compile(cond)
        val xvcond = mb.newBit()
        val out = fb.newLocal()(TypeToTypeInfo(typ)).asInstanceOf[LocalRef[Any]]
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

      case Let(name, value, body, typ) =>
        val vti = TypeToTypeInfo(value.typ)
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
        val ti = TypeToTypeInfo(typ)
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
        val ti = TypeToTypeInfo(typ)
        val tarray = TArray(typ)
        val ati = TypeToTypeInfo(tarray).asInstanceOf[TypeInfo[Long]]
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
        val ati = TypeToTypeInfo(tarray).asInstanceOf[TypeInfo[Long]]
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
        val eti = TypeToTypeInfo(elementTyp).asInstanceOf[TypeInfo[Any]]
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
        val tti = TypeToTypeInfo(typ)
        val eti = TypeToTypeInfo(tarray.elementType)
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

      case MakeSet(args, elementType) =>
        val srvb = new StagedRegionValueBuilder(fb, TArray(elementType))
        val addElement = srvb.addIRIntermediate(elementType)
        val mvargs = args.map(compile(_))
        val sortedArray = Code(
          srvb.start(args.length, init = true),
          Code(mvargs.map { case (dov, m, v) =>
            Code(dov, m.mux(srvb.setMissing(), addElement(v)), srvb.advance())
          }: _*),
          InplaceSort(region, fb, srvb.offset, TArray(elementType)),
          srvb.offset)

        present(sortedArray)
      case SetAdd(set, element, elementType) =>
        val tArray = TArray(elementType)
        val srvb = new StagedRegionValueBuilder(fb, tArray)
        val (doset, mset, vset) = compile(set)
        val (doelement, melement, velement) = compile(element)
        val len = fb.newLocal[Int]
        val i = fb.newLocal[Int]
        val s = fb.newLocal[Long]
        val mx = mb.newBit()
        val x = fb.newLocal[Long]
        val storedElement = fb.newLocal()(TypeToTypeInfo(tArray.elementType)).asInstanceOf[LocalRef[Any]]
        val elementPointer = fb.newLocal[Long]
        val ord = tArray.unsafeOrdering(true)
        val sortedArray = Code(
          doelement,
          s := coerce[Long](vset),
          len := TContainer.loadLength(region, s),
          melement.mux(
            Code(
              srvb.start(len + 1, init = true),
              i := 0,
              Code.whileLoop(i < len,
                srvb.addRegionValue(elementType)(tArray.loadElement(region, s, i)),
                i++),
              srvb.setMissing()),
            Code(
              storedElement := velement,
              elementPointer := toPointer(region, tArray.elementType)(storedElement),
              srvb.start(len + 1, init = true),
              (len.ceq(0)).mux(
                Code(
                  melement.mux(
                    srvb.setMissing(),
                    srvb.addIRIntermediate(elementType)(velement))),
                Code(
                  i := 0,
                  mx := tArray.isElementMissing(region, s, i),
                  x := mx.mux(0L, tArray.loadElement(region, s, i)),
                  Code.whileLoop(i < len && !mx && (ord.compare(region, x, region, elementPointer) < 0),
                    mx.mux(srvb.setMissing(), srvb.addRegionValue(elementType)(x)),
                    mx := tArray.isElementMissing(region, s, i),
                    x := mx.mux(0L, tArray.loadElement(region, s, i)),
                    i++),
                  melement.mux(
                    srvb.setMissing(),
                    srvb.addIRIntermediate(elementType)(storedElement)),
                  Code.whileLoop(i < len,
                    tArray.isElementMissing(region, s, i).mux(
                      srvb.setMissing(),
                      srvb.addRegionValue(elementType)(x)),
                    i++))))),
          srvb.offset)

        (doset, mset, sortedArray)
      case SetContains(set, element) =>
        val tArray = set.typ.fundamentalType.asInstanceOf[TArray]
        val (doset, mset, vset) = compile(set)
        val (doelement, melement, velement) = compile(element)
        val len = fb.newLocal[Int]
        val i = fb.newLocal[Int]
        val s = fb.newLocal[Long]
        val notfound = mb.newBit()
        val storedElement = fb.newLocal()(TypeToTypeInfo(tArray.elementType)).asInstanceOf[LocalRef[Any]]
        val elementPointer = fb.newLocal[Long]
        val ord = tArray.unsafeOrdering(true)
        val result = Code(
          doelement,
          s := coerce[Long](vset),
          len := TContainer.loadLength(region, s),
          melement.mux(
            len.cne(0) && tArray.isElementMissing(region, s, len - 1),
            Code(
              storedElement := velement,
              elementPointer := toPointer(region, tArray.elementType)(storedElement),
              Code(
                i := 0,
                notfound := true,
                Code.whileLoop(i < len && notfound,
                  notfound := ord.compare(region, elementPointer, region, tArray.loadElement(region, s, i)).cne(0),
                  i++),
                !notfound))))

        (doset, mset, result)

      case In(i, typ) =>
        (Code._empty, fb.getArg[Boolean](i*2 + 3), fb.getArg(i*2 + 2)(TypeToTypeInfo(typ)))
      case InMissingness(i) =>
        present(fb.getArg[Boolean](i*2 + 3))
      case Die(m) =>
        present(Code._throw(Code.newInstance[RuntimeException, String](m)))
    }
  }
}
