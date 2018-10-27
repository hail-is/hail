package is.hail.cxx

import is.hail.expr.ir
import is.hail.expr.types.TStruct
import is.hail.expr.types.physical.{PBaseStruct, PStruct, PTuple}

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
      case ir.NA(t) =>
        triplet("", "true", typeDefaultValue(pType))

      case ir.ApplyBinaryPrimOp(op, l, r) =>
        val lt = emit(l)
        val rt = emit(r)

        val v = op match {
          case ir.Add() => s"${ lt.v } + ${ rt.v }"
          case ir.Subtract() => s"${ lt.v } - ${ rt.v }"
          case ir.Multiply() => s"${ lt.v } * ${ rt.v }"
            // FIXME
          case ir.FloatingPointDivide() => s"${ lt.v } / ${ rt.v }"
          case ir.RoundToNegInfDivide() => s"${ lt.v } / ${ rt.v }"
        }

        triplet(Code(lt.setup, rt.setup), s"${ lt.m } || ${ rt.m }", v)

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
        val sb = new BaseStructBuilder(fb, pType.asInstanceOf[PBaseStruct])
        fields.foreach { x =>
          sb.add(emit(x))
        }
        sb.result()

      case ir.MakeStruct(fields) =>
        val sb = new BaseStructBuilder(fb, pType.asInstanceOf[PBaseStruct])
        fields.foreach { case (_, x) =>
          sb.add(emit(x))
        }
        sb.result()

      case ir.In(i, _) =>
        EmitTriplet(x.pType, "",
          "false", fb.getArg(nSpecialArgs + i).toString)

      case _ =>
        throw new CXXUnsupportedOperation(ir.Pretty(x))
    }
  }
}
