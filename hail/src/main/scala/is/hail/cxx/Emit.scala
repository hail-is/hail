package is.hail.cxx

import is.hail.expr.ir
import is.hail.expr.types._
import is.hail.expr.types.physical.{PArray, PBaseStruct, PStruct, PTuple}
import is.hail.utils.ArrayBuilder

object Emit {
  def apply(fb: FunctionBuilder, nSpecialArgs: Int, x: ir.IR): EmitTriplet = {
    val emitter = new Emitter(fb, nSpecialArgs)
    emitter.emit(x)
  }
}

class CXXUnsupportedOperation(msg: String = null) extends Exception(msg)

class Emitter(fb: FunctionBuilder, nSpecialArgs: Int) {
  type E = ir.Env[EmitTriplet]

  def emit(x: ir.IR): EmitTriplet = emit(x, ir.Env.empty[EmitTriplet])

  def emit(x: ir.IR, env: E): EmitTriplet = {
    def triplet(setup: Code, m: Code, v: Code): EmitTriplet =
      EmitTriplet(x.pType, setup, m, v)

    def present(v: Code): EmitTriplet = triplet("", "false", v)

    def emit(x: ir.IR, env: E = env): EmitTriplet = this.emit(x, env)

    val pType = x.pType
    x match {
      case ir.I64(v) =>
        present(s"INT64_C($v)")
      case ir.I32(v) =>
        present(s"INT32_C($v)")
      case ir.F64(v) =>
        present(s"$v")
      case ir.F32(v) =>
        present(s"${v}f")
      case ir.True() =>
        present("true")
      case ir.False() =>
        present("false")
      case ir.Void() =>
        present("")

      case ir.Cast(v, _) =>
        val t = emit(v)
        triplet(t.setup, t.m, s"static_cast<${ typeToCXXType(pType) }>(${ t.v })")

      case ir.NA(t) =>
        triplet("", "true", typeDefaultValue(pType))

      case ir.IsNA(v) =>
        val t = emit(v)
        triplet(t.setup, "false", t.m)

      case ir.If(cond, cnsq, altr) =>
        // FIXME
        assert(pType == cnsq.pType)
        assert(pType == altr.pType)

        val m = Variable("m", "bool")
        val v = Variable("v", typeToCXXType(pType))
        val tcond = emit(cond)
        val tcnsq = emit(cnsq)
        val taltr = emit(altr)

        triplet(s"""
${ tcond.setup }
${ m.define }
${ v.define }
if (${ tcond.m })
  $m = true;
else {
  if (${ tcond.v }) {
    ${ tcnsq.setup }
    $m = ${ tcnsq.m };
    $v = ${ tcnsq.v };
  } else {
    ${ taltr.setup }
    $m = ${ taltr.m };
    $v = ${ taltr.v };
  }
}
""",
          m.toString,
          v.toString)

      case ir.Let(name, value, body) =>
        val tvalue = emit(value)
        val m = Variable("m", "bool", tvalue.m)
        val v = Variable("v", typeToCXXType(value.pType))
        val tbody = emit(body, env.bind(name, EmitTriplet(value.pType, "", m.toString, v.toString)))

        triplet(
          Code(tvalue.setup, m.define, v.define, s"if (!$m) $v = ${ tvalue.v };", tbody.setup),
          tbody.m,
          tbody.v)

      case ir.Ref(name, _) =>
        env.lookup(name)

      case ir.ApplyBinaryPrimOp(op, l, r) =>
        assert(l.typ == r.typ)

        val lt = emit(l)
        val rt = emit(r)

        val v = op match {
          case ir.Add() => s"${ lt.v } + ${ rt.v }"
          case ir.Subtract() => s"${ lt.v } - ${ rt.v }"
          case ir.Multiply() => s"${ lt.v } * ${ rt.v }"
          case ir.FloatingPointDivide() =>
            l.typ match {
              case _: TInt32 | _: TInt64 | _: TFloat32 =>
                s"static_cast<float>(${ lt.v }) / static_cast<float>(${ rt.v })"
              case _: TFloat64 =>
                s"static_cast<double>(${ lt.v }) / static_cast<double>(${ rt.v })"
            }

          case ir.RoundToNegInfDivide() =>
            l.typ match {
              case _: TInt32 => s"floordiv(${ lt.v }, ${ rt.v })"
              case _: TInt64 => s"lfloordiv(${ lt.v }, ${ rt.v })"
              case _: TFloat32 => s"floorf(${ lt.v } / ${ rt.v })"
              case _: TFloat64 => s"floor(${ lt.v } / ${ rt.v })"
            }
        }

        triplet(Code(lt.setup, rt.setup), s"${ lt.m } || ${ rt.m }", v)

      case ir.ApplyUnaryPrimOp(op, x) =>
        val t = emit(x)

        val v = op match {
          case ir.Bang() => s"! ${ t.v }"
          case ir.Negate() => s"- ${ t.v }"
        }

        triplet(t.setup, t.m, v)

      case ir.ArrayRef(a, i) =>
        val at = emit(a)
        val it = emit(i)
        a.pType.asInstanceOf[PArray].cxxLoadElement(at, it)

      case ir.ArrayLen(a) =>
        val t = emit(a)
        triplet(t.setup, t.m, a.pType.asInstanceOf[PArray].cxxLoadLength(t.v))

      case ir.GetField(o, name) =>
        val fieldIdx = o.typ.asInstanceOf[TStruct].fieldIdx(name)
        val pStruct = o.pType.asInstanceOf[PStruct]
        val ot = emit(o).memoize(fb)
        triplet(Code(ot.setup),
          s"${ ot.m } || (${ pStruct.cxxIsFieldMissing(ot.v, fieldIdx) })",
          pStruct.cxxLoadField(ot.v, fieldIdx))

      case ir.GetTupleElement(o, idx) =>
        val pStruct = o.pType.asInstanceOf[PTuple]
        val ot = emit(o).memoize(fb)
        triplet(Code(ot.setup),
          s"${ ot.m } || (${ pStruct.cxxIsFieldMissing(ot.v, idx) })",
          pStruct.cxxLoadField(ot.v, idx))

      case ir.MakeTuple(fields) =>
        val sb = new StagedBaseStructBuilder(fb, pType.asInstanceOf[PBaseStruct])
        fields.foreach { x =>
          sb.add(emit(x))
        }
        sb.result()

      case ir.MakeStruct(fields) =>
        val sb = new StagedBaseStructBuilder(fb, pType.asInstanceOf[PBaseStruct])
        fields.foreach { case (_, x) =>
          sb.add(emit(x))
        }
        sb.result()

      case ir.MakeArray(args, _) =>
        val sab = new StagedArrayBuilder(fb, pType.asInstanceOf[PArray])
        val sb = new ArrayBuilder[Code]

        sb += sab.start(s"${ args.length }")
        args.foreach { arg =>
          val argt = emit(arg)
          sb += s"""
${ argt.setup }
if (${ argt.m })
  ${ sab.setMissing() }
else
  ${ sab.add(argt.v) }
${ sab.advance() }
"""
        }

        triplet(sb.result().mkString, "false", sab.end())

      case x@ir.ApplyIR(_, _, _) =>
        // FIXME small only
        emit(x.explicitNode)

      case ir.In(i, _) =>
        EmitTriplet(x.pType, "",
          "false", fb.getArg(nSpecialArgs + i).toString)

      case _ =>
        throw new CXXUnsupportedOperation(ir.Pretty(x))
    }
  }
}
