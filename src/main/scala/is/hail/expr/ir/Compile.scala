package is.hail.expr.ir

import is.hail.utils._
import is.hail.annotations.MemoryBuffer
import is.hail.asm4s._
import is.hail.expr
import is.hail.expr.{TInt32, TInt64, TArray, TContainer, TStruct, TFloat32, TFloat64, TBoolean}
import is.hail.annotations.StagedRegionValueBuilder

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


  def apply(ir: IR, fb: FunctionBuilder[_], env: Map[String, (TypeInfo[_], LocalRef[_])]) {
    fb.emit(expression(ir, fb, env))
  }

  def expression(ir: IR, fb: FunctionBuilder[_], env: Map[String, (TypeInfo[_], LocalRef[_])]): Code[_] = {
    val region = fb.getArg[MemoryBuffer](1).load()
    def expression(ir: IR, fb: FunctionBuilder[_] = fb, env: Map[String, (TypeInfo[_], LocalRef[_])] = env): Code[_] =
      Compile.expression(ir, fb, env)
    ir match {
      case NA(typ) =>
        ???
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
      case expr.ir.If(cond, cnsq, altr, typ) =>
        val x = fb.newLocal[Boolean]

        Code(x := expression(cond).asInstanceOf[Code[Boolean]],
          x.mux(expression(cnsq), expression(altr)))
      case MapNA(name, value, body, typ) =>
        ???
      case expr.ir.Let(name, value, body, typ) =>
        // FIXME: value could be a pointer
        fb.newLocal()(typeToTypeInfo(value.typ)) match { case x: LocalRef[t] =>
          Code(x := expression(value).asInstanceOf[Code[t]],
            expression(body, env = env + (name -> (typeToTypeInfo(value.typ), x))))
        }
      case Ref(name, typ) =>
        assert(env(name)._1 == typeToTypeInfo(typ), s"bad type annotation for $name: $typ, binding in scope: ${env(name)}")
        env(name)._2
      case Set(name, v) =>
        env(name)._2.asInstanceOf[LocalRef[Any]] := expression(v)
      case ApplyPrimitive(op, args, typ) =>
        Primitives.lookup(op, args.map(_.typ), args.map(expression(_)))
      case LazyApplyPrimitive(op, args, typ) =>
        ???
      case expr.ir.Lambda(name, paramTyp, body, typ) =>
        ???
      case MakeArray(args, typ) =>
        val srvb = new StagedRegionValueBuilder(fb, typ)
        val addElement = srvb.addAnnotation(typ.elementType)
        Code(
          srvb.start(args.length, init = false),
          Code(args.map(expression(_)).map(addElement): _*),
          srvb.offset)
      case ArrayRef(a, i, typ) =>
        val arr = expression(a).asInstanceOf[Code[Long]]
        val idx = expression(i).asInstanceOf[Code[Int]]
        val eoff = TArray(typ).loadElement(region, arr, idx)

        loadAnnotation(region, typ)(eoff)
      case ArrayLen(a) =>
        val arr = expression(a).asInstanceOf[Code[Long]]

        TContainer.loadLength(region, arr)
      case For(value, idx, array, body) =>
        val tarray = array.typ.asInstanceOf[TArray]
        val t = tarray.elementType
        typeToTypeInfo(t) match { case ti: TypeInfo[t] =>
        implicit val tti: TypeInfo[t] = ti
        val a = fb.newLocal[Long]
        val i = fb.newLocal[Int]
        val x = fb.newLocal[t]
        val len = fb.newLocal[Int]
        Code(
          a := expression(array).asInstanceOf[Code[Long]],
          i := 0,
          len := TContainer.loadLength(region, a),
          Code.whileLoop(i < len,
            x.store(loadAnnotation(region, t)(tarray.loadElement(region, a, i)).asInstanceOf[Code[t]]),
            expression(body, env = env + (value -> (ti, x)) + (idx -> (typeInfo[Int], i))),
            i := i + 1))
      }
      case MakeStruct(fields) =>
        val t = TStruct(fields.map { case (name, t, _) => (name, t) }: _*)
        val initializers = fields.map { case (_, t, v) => (t, expression(v)) }
        val srvb = new StagedRegionValueBuilder(fb, t)

        Code(
          srvb.start(false),
          Code(initializers.map { case (t, v) => srvb.addAnnotation(t)(v) }: _*),
          srvb.offset)
      case GetField(o, name, _) =>
        val t = o.typ.asInstanceOf[TStruct]
        val struct = expression(o).asInstanceOf[Code[Long]]

        t.fieldOffset(struct, t.fieldIdx(name))
      case Seq(stmts, typ) =>
        Code(stmts.map(expression(_)): _*)
      case In(i, typ) =>
        fb.getArg(i + 2)(typeToTypeInfo(typ))
      case Out(v) =>
        typeToTypeInfo(v.typ) match { case ti: TypeInfo[t] =>
          Code._return(expression(v).asInstanceOf[Code[t]])(ti)
        }
    }
  }
}
