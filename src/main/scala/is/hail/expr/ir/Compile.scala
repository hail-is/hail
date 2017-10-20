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

  def apply(ir: IR, fb: FunctionBuilder[_], env: Map[String, (TypeInfo[_], LocalRef[_])]) {
    fb.emit(expression(ir, fb, env, new StagedRegionValueBuilder(fb)))
  }

  def expression(ir: IR, fb: FunctionBuilder[_], env: Map[String, (TypeInfo[_], LocalRef[_])], srvb: StagedRegionValueBuilder): Code[_] = {
    val region = fb.getArg[MemoryBuffer](1).load()
    def expression(ir: IR, fb: FunctionBuilder[_] = fb, env: Map[String, (TypeInfo[_], LocalRef[_])] = env): Code[_] =
      Compile.expression(ir, fb, env, srvb)
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
        Code(
          srvb.start(TArray(typ), args.length, init = false),
          Code(args.map(expression(_)).map(srvb.addAnnotation(typ, _)): _*),
          srvb.build())
      case ArrayRef(a, i, typ) =>
        val arr = expression(a).asInstanceOf[Code[Long]]
        val idx = expression(i).asInstanceOf[Code[Int]]

        typ match {
          case TInt32 =>
            region.loadInt(TArray(typ).loadElement(region, arr, idx))
          case TInt64 =>
            region.loadLong(TArray(typ).loadElement(region, arr, idx))
          case TFloat32 =>
            region.loadFloat(TArray(typ).loadElement(region, arr, idx))
          case TFloat64 =>
            region.loadDouble(TArray(typ).loadElement(region, arr, idx))
        }
      case ArrayLen(a) =>
        val arr = expression(a).asInstanceOf[Code[Long]]

        TContainer.loadLength(region, arr)
      case For(name, start, end, body, nameTyp) => nameTyp match {
        case TInt32 =>
          val vi = fb.newLocal[Int]
          val vend = fb.newLocal[Int]
          Code(
            vi := expression(start).asInstanceOf[Code[Int]],
            vend := expression(end).asInstanceOf[Code[Int]],
            Code.whileLoop(vi < vend,
              expression(body, env = env + (name -> (typeInfo[Int], vi))),
              vi := (vi + 1)))
        case TInt64 =>
          val vi = fb.newLocal[Long]
          val vend = fb.newLocal[Long]
          Code(
            vi := expression(start).asInstanceOf[Code[Long]],
            vend := expression(end).asInstanceOf[Code[Long]],
            Code.whileLoop(vi < vend,
              expression(body, env = env + (name -> (typeInfo[Long], vi))),
              vi := (vi + 1L)))
      }
      case MakeStruct(fields) =>
        val t = TStruct(fields.map { case (name, t, _) => (name, t) }: _*)
        val initializers = fields.map { case (_, t, v) => (t, expression(v)) }

        Code(
          srvb.start(t),
          Code(initializers.map { case (t, v) => srvb.addAnnotation(t, v) }: _*),
          srvb.build())
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
