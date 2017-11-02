package is.hail.expr.ir

import is.hail.utils.lensing._
import is.hail.utils._
import is.hail.annotations.MemoryBuffer
import is.hail.asm4s._
import is.hail.expr
import is.hail.expr.{TInt32, TInt64, TArray, TContainer, TStruct, TFloat32, TFloat64, TBoolean}
import is.hail.annotations.StagedRegionValueBuilder
import scala.collection.generic.Growable

import org.objectweb.asm.Opcodes._
import org.objectweb.asm.Type
import org.objectweb.asm.tree._

import scala.reflect.classTag
import scala.reflect.ClassTag

object Compile {
  private def dummyValue(t: expr.Type): Code[_] = t match {
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

  private def loadAnnotation(region: Code[MemoryBuffer], typ: expr.Type): Code[Long] => Code[_] = typ match {
    case TInt32 =>
      region.loadInt(_)
    case TInt64 =>
      region.loadLong(_)
    case TFloat32 =>
      region.loadFloat(_)
    case TFloat64 =>
      region.loadDouble(_)
    case _ =>
      off => off
  }

  private def storeAnnotation(region: Code[MemoryBuffer], typ: expr.Type): (Code[Long], Code[_]) => Code[_] = typ match {
    case TInt32 =>
      (off, v) => region.storeInt32(off, v.asInstanceOf[Code[Int]])
    case TInt64 =>
      (off, v) => region.storeInt64(off, v.asInstanceOf[Code[Long]])
    case TFloat32 =>
      (off, v) => region.storeFloat32(off, v.asInstanceOf[Code[Float]])
    case TFloat64 =>
      (off, v) => region.storeFloat64(off, v.asInstanceOf[Code[Double]])
    case _ =>
      (off, ptr) => region.storeAddress(off, ptr.asInstanceOf[Code[Long]])
  }

  class MissingBits(fb: FunctionBuilder[_]) {
    private var used = 0
    private var bits: LocalRef[Long] = null
    private var count = 0

    def newBit(): MissingBit = {
      if (used >= 64 || bits == null) {
        bits = fb.newLocal[Long]("missingbits"+count)
        count += 1
        fb.emit(bits.store(0L))
        used = 0
      }

      used += 1
      new MissingBit(bits, used - 1)
    }
  }

  class MissingBit(bits: LocalRef[Long], i: Int) extends Code[Boolean] with Settable[Boolean] {
    assert(i >= 0)
    assert(i < 64)

    def store(b: Code[Boolean]): Code[Unit] = {
      bits := bits & ~(1L << i) | (b.toL << i)
    }

    def emit(il: Growable[AbstractInsnNode]): Unit = {
      ((bits >> i) & 1L).toI.emit(il)
    }

    def load(): Code[Boolean] = this
  }

  type E = Env[(TypeInfo[_], Settable[_])]

  def apply(ir: IR, fb: FunctionBuilder[_]) {
    apply(ir, fb, new Env())
  }

  def apply(ir: IR, fb: FunctionBuilder[_], env: E) {
    fb.emit(expression(ir, fb, env, new MissingBits(fb)))
  }

  def expression(ir: IR, fb: FunctionBuilder[_], env: E, mb: MissingBits): Code[_] = {
    val region = fb.getArg[MemoryBuffer](1).load()
    def expression(ir: IR, fb: FunctionBuilder[_] = fb, env: E = env, mb: MissingBits = mb): Code[_] =
      Compile.expression(ir, fb, env, mb)
    ir match {
      case I32(x) =>
        const(x)
      case I64(x) =>
        const(x)
      case F32(x) =>
        const(x)
      case F64(x) =>
        const(x)
      case True() =>
        const(true)
      case False() =>
        const(false)

      case NA(typ) =>
        throw new UnsupportedOperationException("run DesugarNA first")
      case IsNA(v) =>
        throw new UnsupportedOperationException("run DesugarNA first")
      case MapNA(name, value, body, typ) =>
        throw new UnsupportedOperationException("run DesugarNA first")

      case expr.ir.If(cond, cnsq, altr, typ) =>
        expression(cond).asInstanceOf[Code[Boolean]].mux(
          expression(cnsq), expression(altr))

      case expr.ir.Let(name, value, body, typ) =>
        fb.newLocal(name)(typeToTypeInfo(value.typ)) match { case x: LocalRef[t] =>
          val vvalue = expression(value)
          val vbody = expression(body, env = env.bind(name, ((typeToTypeInfo(value.typ), x))))

          Code(x := vvalue.asInstanceOf[Code[t]], vbody)
        }
      case x@Ref(name, typ) =>
        assert(env.lookup(x)._1 == typeToTypeInfo(typ), s"bad type annotation for $name: $typ, binding in scope: ${env.lookup(x)}")
        env.lookup(name)._2.load()

      case ApplyPrimitive(op, args, typ) =>
        Primitives.lookup(op, args.map(_.typ), args.map(expression(_)))
      case LazyApplyPrimitive(op, args, typ) =>
        ???
      case expr.ir.Lambda(names, body, mnames, mbody, typ) =>
        ???
      case MakeArray(args, missingness, typ) =>
        assert(missingness != null, "run explicit missingness first")
        val srvb = new StagedRegionValueBuilder(fb, typ)
        val addElement = srvb.addAnnotation(typ.elementType)

        Code(
          srvb.start(args.length, init = false),
          Code((missingness zip args).map(onBoth(expression(_))).map { case (m: Code[Boolean] @unchecked, v) =>
            Code(m.mux(srvb.setMissing(), addElement(v)), srvb.advance())
          }: _*),
          srvb.offset)
      case x@MakeArrayN(len, elementType) =>
        val srvb = new StagedRegionValueBuilder(fb, x.typ)

        Code(srvb.start(expression(len).asInstanceOf[Code[Int]], init = true),
          srvb.offset)
      case ArrayRef(a, i, typ) =>
        val tarray = TArray(typ)

        loadAnnotation(region, typ)(
          tarray.loadElement(region,
            expression(a).asInstanceOf[Code[Long]],
            expression(i).asInstanceOf[Code[Int]]))
      case ArrayMissingnessRef(a, i) =>
        val tarray = a.typ.asInstanceOf[TArray]

        !tarray.isElementDefined(region,
          expression(a).asInstanceOf[Code[Long]],
          expression(i).asInstanceOf[Code[Int]])
      case ArrayLen(a) =>
        TContainer.loadLength(region, expression(a).asInstanceOf[Code[Long]])
      case x@ArrayMap(a, Lambda(Array((name,_)), body, Array(mname), mbody, _), elementTyp) =>
        val tin = a.typ.asInstanceOf[TArray]
        val tout = x.typ.asInstanceOf[TArray]
        val srvb = new StagedRegionValueBuilder(fb, tout)
        val addElement = srvb.addAnnotation(tout.elementType)
        typeToTypeInfo(elementTyp) match { case ti: TypeInfo[t] =>
          val xa = fb.newLocal[Long]("am_a")
          val xmv = mb.newBit()
          val xvv = fb.newLocal(name)(ti)
          val i = fb.newLocal[Int]("am_i")
          val len = fb.newLocal[Int]("am_len")
          val bodyenv = env.bind(
            name -> (ti, xvv),
            mname -> (typeInfo[Boolean], xmv))

          Code(
            xa := expression(a).asInstanceOf[Code[Long]],
            len := TContainer.loadLength(region, xa),
            i := 0,
            Code.getStatic[System, java.io.PrintStream]("out").invoke("print", "length ")(classTag[String], ClassTag(java.lang.Void.TYPE)),
            Code.getStatic[System, java.io.PrintStream]("out").invoke("println", len)(classTag[Int], ClassTag(java.lang.Void.TYPE)),
            srvb.start(len, init = false),
            Code.whileLoop(i < len,
              xmv := !tin.isElementDefined(region, xa, i),
              // FIXME ugh why isn't this handled in desugarna?
              xvv := xmv.mux(
                dummyValue(tin.elementType),
                loadAnnotation(region, tin.elementType)(
                  tin.loadElement(region, xa, i))).asInstanceOf[Code[t]],
              expression(mbody, env = bodyenv).asInstanceOf[Code[Boolean]].mux(
                srvb.setMissing(),
                addElement(expression(body, env = bodyenv))),
              srvb.advance(),
              Code.getStatic[System, java.io.PrintStream]("out").invoke[String, Void]("print", "i ")(classTag[String], ClassTag(java.lang.Void.TYPE)),
              Code.getStatic[System, java.io.PrintStream]("out").invoke[Int, Void]("println", i)(classTag[Int], ClassTag(java.lang.Void.TYPE)),
              i := i + 1),
            srvb.offset)
        }
      case ArrayMap(_, _, _) =>
        throw new UnsupportedOperationException(s"bad arraymap $ir")
      case ArrayFold(
        a,
        zero,
        Lambda(Array((name1, _), (name2, _)), body, Array(mname1, mname2), mbody, _),
        mzero,
        typ) =>
        val tarray = a.typ.asInstanceOf[TArray]
        assert(tarray != null, s"tarray is null! $ir")

        (typeToTypeInfo(typ), typeToTypeInfo(tarray.elementType)) match { case (tti: TypeInfo[t], uti: TypeInfo[u]) =>
          val xa = fb.newLocal[Long]("af_array")
          val xmv = mb.newBit()
          val xvv = fb.newLocal(name2)(uti)
          val xmout = mb.newBit()
          val temp = mb.newBit()
          val xvout = fb.newLocal(name1)(tti)
          val i = fb.newLocal[Int]("af_i")
          val len = fb.newLocal[Int]("af_len")
          val bodyenv = env.bind(
            name1 -> ((tti, xvout)),
            name2 -> ((uti, xvv)),
            mname1 -> ((typeInfo[Boolean], xmout)),
            mname2 -> ((typeInfo[Boolean], xmv)))

          Code(
            xa := expression(a).asInstanceOf[Code[Long]],
            len := TContainer.loadLength(region, xa),
            i := 0,
            xvout := expression(zero).asInstanceOf[Code[t]],
            xmout := expression(mzero).asInstanceOf[Code[Boolean]],
            Code.whileLoop(i < len,
              xmv := !tarray.isElementDefined(region, xa, i),
              xvv := xmv.mux(
                dummyValue(tarray.elementType),
                loadAnnotation(region, tarray.elementType)(
                  tarray.loadElement(region, xa, i))).asInstanceOf[Code[u]],
              temp := expression(mbody, env = bodyenv).asInstanceOf[Code[Boolean]],
              xvout := expression(body, env = bodyenv).asInstanceOf[Code[t]],
              xmout := temp,
              i := i + 1),
            xvout)
        }
      case ArrayFold(_, _, _, _, _) =>
        throw new UnsupportedOperationException(s"bad arrayfold $ir")
      case MakeStruct(fields, missingness) =>
        assert(missingness != null, "run explicit missingness first")
        val t = TStruct(fields.map { case (name, t, _) => (name, t) }: _*)
        val initializers = fields.map { case (_, t, v) => (t, expression(v)) }
        val cmissingness = missingness map (expression(_))
        val srvb = new StagedRegionValueBuilder(fb, t)

        Code(
          srvb.start(false),
          Code((cmissingness zip initializers).map { case (mv: Code[Boolean] @unchecked, (t, vv)) =>
            Code(
              mv.mux(srvb.setMissing(), srvb.addAnnotation(t)(vv)),
              srvb.advance()) }: _*),
          srvb.offset)
      case GetField(o, name, _) =>
        val t = o.typ.asInstanceOf[TStruct]
        val fieldIdx = t.fieldIdx(name)
        loadAnnotation(region, t)(t.fieldOffset(expression(o).asInstanceOf[Code[Long]], fieldIdx))
      case GetFieldMissingness(o, name) =>
        val t = o.typ.asInstanceOf[TStruct]
        val fieldIdx = t.fieldIdx(name)
        !t.isFieldDefined(region, expression(o).asInstanceOf[Code[Long]], fieldIdx)
      case Seq(stmts, typ) =>
        Code(stmts map (expression(_)): _*)
      case In(i, typ) =>
        fb.getArg(i*2 + 2)(typeToTypeInfo(typ))
      case InMissingness(i) =>
        fb.getArg[Boolean](i*2 + 3)
      case Out(v) =>
        typeToTypeInfo(v.typ) match { case ti: TypeInfo[t] =>
          Code._return(expression(v).asInstanceOf[Code[t]])(ti)
        }
      case Die(m) =>
        Code._throw(Code.newInstance[RuntimeException, String](m))
    }
  }
}
