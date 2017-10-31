package is.hail.expr.ir

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

object Compile {
  private def typeToTypeInfo(t: expr.Type): TypeInfo[_] = t match {
    case TInt32 => typeInfo[Int]
    case TInt64 => typeInfo[Long]
    case TFloat32 => typeInfo[Float]
    case TFloat64 => typeInfo[Double]
    case TBoolean => typeInfo[Boolean]
    case _ => typeInfo[Long] // reference types
  }

  private def dummyValue(t: expr.Type): Code[_] = t match {
    case TInt32 => 0
    case TInt64 => 0L
    case TFloat32 => 0.0f
    case TFloat64 => 0.0
    case _ => 0L // reference types
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

    def newBit(): MissingBit = {
      if (used >= 64 || bits == null) {
        bits = fb.newLocal[Long]
        fb.emit(bits.store(0L))
        used = 0
      }

      used += 1
      new MissingBit(bits, used - 1)
    }
  }

  class MissingBit(bits: LocalRef[Long], i: Int) extends Code[Boolean] {
    assert(i >= 0)
    assert(i < 64)

    def :=(b: Code[Boolean]): Code[_] = {
      bits := bits & ~(1L << i) | (b.toL << i)
    }

    def emit(il: Growable[AbstractInsnNode]): Unit = {
      ((bits >> i) & 1L).toI.emit(il)
    }
  }

  def apply(ir: IR, fb: FunctionBuilder[_], env: Map[String, (TypeInfo[_], MissingBit, LocalRef[_])]) {
    fb.emit(expression(ir, fb, env, new MissingBits(fb))._2)
  }

  def expression(ir: IR, fb: FunctionBuilder[_], env: Map[String, (TypeInfo[_], MissingBit, LocalRef[_])], mb: MissingBits): (Code[Boolean], Code[_]) = {
    val region = fb.getArg[MemoryBuffer](1).load()
    def expression(ir: IR, fb: FunctionBuilder[_] = fb, env: Map[String, (TypeInfo[_], MissingBit, LocalRef[_])] = env, mb: MissingBits = mb): (Code[Boolean], Code[_]) =
      Compile.expression(ir, fb, env, mb)
    ir match {
      case I32(x) =>
        (const(false), const(x))
      case I64(x) =>
        (const(false), const(x))
      case F32(x) =>
        (const(false), const(x))
      case F64(x) =>
        (const(false), const(x))
      case True() =>
        (const(false), const(true))
      case False() =>
        (const(false), const(false))

      case NA(typ) =>
        (const(true), ???)
      case IsNA(v) =>
        (const(false), expression(v)._1)
      case MapNA(name, value, body, typ) =>
        fb.newLocal()(typeToTypeInfo(value.typ)) match { case x: LocalRef[t] =>
          val mx = mb.newBit()
          val (mvalue, vvalue) = expression(value)
          val (mbody, vbody) =
            expression(body, env = env + (name -> (typeToTypeInfo(value.typ), mx, x)))

          (mvalue || mbody,
            mvalue.mux(
              dummyValue(typ),
              Code(mx := const(false), x := vvalue.asInstanceOf[Code[t]], vbody)))
        }

      case expr.ir.If(cond, cnsq, altr, typ) =>
        val (mcnsq, vcnsq) = expression(cnsq)
        val (maltr, valtr) = expression(altr)
        val (_, vcond: Code[Boolean] @unchecked) = expression(cond)

        (mcnsq || maltr, vcond.mux(vcnsq, valtr))

      case expr.ir.Let(name, value, body, typ) =>
        fb.newLocal()(typeToTypeInfo(value.typ)) match { case x: LocalRef[t] =>
          val mx = mb.newBit()
          val (mvalue, vvalue) = expression(value)
          val (mbody, vbody) =
            expression(body, env = env + (name -> ((typeToTypeInfo(value.typ), mx, x))))

          (mbody, Code(mx := mvalue, x := mx.mux(dummyValue(value.typ), vvalue).asInstanceOf[Code[t]], vbody))
        }
      case Ref(name, typ) =>
        assert(env(name)._1 == typeToTypeInfo(typ), s"bad type annotation for $name: $typ, binding in scope: ${env(name)}")
        (env(name)._2, env(name)._3)
      case Set(name, v) =>
        val (mv, vv) = expression(v)
        (const(false),
          mv.mux(
            env(name)._2 := const(false),
            env(name)._3.asInstanceOf[LocalRef[Any]] := vv))
      case ApplyPrimitive(op, args, typ) =>
        val (margs, vargs) = args.map(expression(_)).unzip
        val missing = margs.fold(const(false))(_ || _)

        (missing, Primitives.lookup(op, args.map(_.typ), vargs))
      case LazyApplyPrimitive(op, args, typ) =>
        ???
      case expr.ir.Lambda(name, paramTyp, body, typ) =>
        ???
      case MakeArray(args, typ) =>
        val srvb = new StagedRegionValueBuilder(fb, typ)
        val addElement = srvb.addAnnotation(typ.elementType)

        (const(false), Code(
          srvb.start(args.length, init = false),
          Code(args.map(expression(_)).map { case (m, v) =>
            Code(m.mux(srvb.setMissing(), addElement(v)), srvb.advance())
          }: _*),
          srvb.offset))
      case x@MakeArrayN(len, elementType) =>
        val srvb = new StagedRegionValueBuilder(fb, x.typ)
        val (mlen, vlen: Code[Int] @unchecked) = expression(len)

        (mlen, Code(
          srvb.start(vlen, init = true),
          srvb.offset))
      case ArrayRef(a, i, typ) =>
        val tarray = TArray(typ)
        val (marr, varr: Code[Long] @unchecked) = expression(a)
        val (midx, vidx: Code[Int] @unchecked) = expression(i)
        val eoff = tarray.loadElement(region, varr, vidx)
        val missing = marr || midx || !tarray.isElementDefined(region, varr, vidx)

        (missing, loadAnnotation(region, typ)(eoff))
      case ArrayLen(a) =>
        val (marr, varr: Code[Long] @unchecked) = expression(a)

        (marr, TContainer.loadLength(region, varr))
      case ArraySet(a, i, v) =>
        val tarray = a.typ.asInstanceOf[TArray]
        val t = tarray.elementType
        val (marr, varr: Code[Long] @unchecked) = expression(a)
        val (midx, vidx: Code[Int] @unchecked) = expression(i)
        val (mvalue, value) = expression(v)
        val ptr = tarray.elementOffsetInRegion(region, varr, vidx)

        (const(false), mvalue.mux(
          tarray.setElementMissing(region, varr, vidx),
          storeAnnotation(region, t)(ptr, value)))
      case For(value, i, array, body) =>
        val tarray = array.typ.asInstanceOf[TArray]
        val t = tarray.elementType
        typeToTypeInfo(t) match { case ti: TypeInfo[t] =>
          implicit val tti: TypeInfo[t] = ti
          val a = fb.newLocal[Long]
          val idx = fb.newLocal[Int]
          val midx = mb.newBit()
          val x = fb.newLocal[t]
          val mx = mb.newBit()
          val len = fb.newLocal[Int]
          val (marray, varray: Code[Long] @unchecked) = expression(array)
          val (mbody, vbody) =
            expression(body, env = env + (value -> ((ti, mx, x))) + (i -> ((typeInfo[Int], midx, idx))))

          (const(false), marray.mux(Code._empty, Code(
            a := varray,
            idx := 0,
            midx := false,
            len := TContainer.loadLength(region, a),
            Code.whileLoop(idx < len,
              mx := !tarray.isElementDefined(region, a, idx),
              x := mx.mux(dummyValue(t),
                loadAnnotation(region, t)(tarray.loadElement(region, a, idx))).asInstanceOf[Code[t]],
              // FIXME: seems odd that the body itself is missing??
              mbody.mux(Code._empty, vbody),
              idx := idx + 1))))
        }
      case MakeStruct(fields) =>
        val t = TStruct(fields.map { case (name, t, _) => (name, t) }: _*)
        val initializers = fields.map { case (_, t, v) => (t, expression(v)) }
        val srvb = new StagedRegionValueBuilder(fb, t)

        (const(false), Code(
          srvb.start(false),
          Code(initializers.map { case (t, (mv, vv)) =>
            Code(
              mv.mux(srvb.setMissing(), srvb.addAnnotation(t)(vv)),
              srvb.advance()) }: _*),
          srvb.offset))
      case GetField(o, name, _) =>
        val t = o.typ.asInstanceOf[TStruct]
        val (mstruct, vstruct: Code[Long] @unchecked) = expression(o)
        val fieldIdx = t.fieldIdx(name)
        val mfield = t.isFieldDefined(region, vstruct, fieldIdx)

        (mstruct || mfield, loadAnnotation(region, t)(t.fieldOffset(vstruct, fieldIdx)))
      case Seq(stmts, typ) =>
        val (_, vstmts) = stmts.map(expression(_)).unzip
        (const(false), Code(vstmts: _*))
      case In(i, typ) =>
        // FIXME: allow for missing arguments
        (fb.getArg[Boolean](i*2 + 3), fb.getArg(i*2 + 2)(typeToTypeInfo(typ)))
      case Out(v) =>
        val (mv, vv) = expression(v)
        (const(false), typeToTypeInfo(v.typ) match { case ti: TypeInfo[t] =>
          mv.mux(Code._throw(Code.newInstance[RuntimeException, String]("tried to return a missing value!")),
            Code._return(vv.asInstanceOf[Code[t]])(ti))
        })
    }
  }
}
