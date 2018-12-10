package is.hail.cxx

import is.hail.expr.ir
import is.hail.expr.types._
import is.hail.expr.types.physical._
import is.hail.expr.types.virtual._
import is.hail.utils.{ArrayBuilder, StringEscapeUtils}

import scala.collection.mutable

object Emit {
  def apply(fb: FunctionBuilder, nSpecialArgs: Int, x: ir.IR): EmitTriplet = {
    val emitter = new Emitter(fb, nSpecialArgs)
    emitter.emit(x, ir.Env.empty[EmitTriplet])
  }
}

class CXXUnsupportedOperation(msg: String = null) extends Exception(msg)

abstract class ArrayEmitter(val setup: Code, val m: Code, val setupLen: Code, val length: Option[Code]) {
  def emit(f: (Code, Code) => Code): Code
}

class Orderings {
  val typeOrdering = mutable.Map.empty[(PType, PType), String]

  private def makeOrdering(tub: TranslationUnitBuilder, lp: PType, rp: PType): String = {
    tub.include("hail/Ordering.h")
    lp.virtualType match {
      case TFloat32(_) => "FloatOrd"
      case TFloat64(_) => "DoubleOrd"
      case TInt32(_) => "IntOrd"
      case TInt64(_) => "LongOrd"
      case TBoolean(_) => "BoolOrd"

      case TString(_) | TBinary(_) => "BinaryOrd"

      case t: TContainer =>
        val lContainerP = lp.asInstanceOf[PContainer]
        val rContainerP = rp.asInstanceOf[PContainer]

        val lElemP = lContainerP.elementType
        val rElemP = rContainerP.elementType

        val elemOrd = ordering(tub, lContainerP.elementType, rContainerP.elementType)

        s"ArrayOrd<${ lContainerP.cxxImpl },${ rContainerP.cxxImpl },ExtOrd<$elemOrd>>"

      case t: TBaseStruct =>
        val lBaseStructP = lp.asInstanceOf[PBaseStruct]
        val rBaseStructP = rp.asInstanceOf[PBaseStruct]

        def buildComparisonMethod(
          cb: ClassBuilder,
          op: String,
          rType: String,
          fieldTest: (TranslationUnitBuilder, String, Code, Code, Code, Code) => Code,
          finalValue: Code) {

          val mb = cb.buildMethod(op, Array("const char *" -> "l", "const char *" -> "r"), rType)
          val l = mb.getArg(0)
          val r = mb.getArg(1)

          t.fields.zipWithIndex.foreach { case (f, i) =>
            val lfTyp  = lBaseStructP.fields(i).typ
            val rfTyp = rBaseStructP.fields(i).typ

            val fOrd = ordering(tub, lfTyp, rfTyp)

            val lm = tub.variable("lm", "bool", lBaseStructP.cxxIsFieldMissing(l.toString, i))
            val lv = tub.variable("lv", typeToCXXType(lfTyp), typeDefaultValue(lfTyp))

            val rm = tub.variable("rm", "bool", rBaseStructP.cxxIsFieldMissing(r.toString, i))
            val rv = tub.variable("lv", typeToCXXType(rfTyp), typeDefaultValue(rfTyp))

            mb +=
              s"""
                 |${ lm.define }
                 |${ lv.define }
                 |if (!$lm)
                 |  $lv = ${ lBaseStructP.cxxLoadField("l", i) };
                 |${ rm.define }
                 |${ rv.define }
                 |if (!$rm)
                 |  $rv = ${ rBaseStructP.cxxLoadField("r", i) };
                 |${ fieldTest(tub, s"ExtOrd<$fOrd>", lm.toString, lv.toString, rm.toString, rv.toString) }
               """.stripMargin
          }

          mb += s"return $finalValue;"

          mb.end()
        }

        val ord = tub.genSym("Ord")
        val cb = tub.buildClass(ord)
        cb += "using T = const char *;"

        buildComparisonMethod(cb, "compare", "static int", { (tub, efOrd, lm, lv, rm, rv) =>
          val c = cb.variable("c", "int")
          s"""
             |${ c.define }
             |$c = $efOrd::compare($lm, $lv, $rm, $rv);
             |if ($c != 0) return $c;
             """.stripMargin
        },
          "0")

        buildComparisonMethod(cb, "lt", "static bool", { (tub, efOrd, lm, lv, rm, rv) =>
          s"""
             |if ($efOrd::lt($lm, $lv, $rm, $rv)) return true;
             |if (!$efOrd::eq($lm, $lv, $rm, $rv)) return false;
               """.stripMargin
        },
          "false")

        buildComparisonMethod(cb, "lteq", "static bool", { (tub, efOrd, lm, lv, rm, rv) =>
          s"""
             |if ($efOrd::lt($lm, $lv, $rm, $rv)) return true;
             |if (!$efOrd::eq($lm, $lv, $rm, $rv)) return false;
               """.stripMargin
        },
          "true")

        buildComparisonMethod(cb, "gt", "static bool", { (tub, efOrd, lm, lv, rm, rv) =>
          s"""
             |if ($efOrd::gt($lm, $lv, $rm, $rv)) return true;
             |if (!$efOrd::eq($lm, $lv, $rm, $rv)) return false;
               """.stripMargin
        },
          "false")

        buildComparisonMethod(cb, "gteq", "static bool", { (tub, efOrd, lm, lv, rm, rv) =>
          s"""
             |if ($efOrd::gt($lm, $lv, $rm, $rv)) return true;
             |if (!$efOrd::eq($lm, $lv, $rm, $rv)) return false;
               """.stripMargin
        },
          "true")

        buildComparisonMethod(cb, "eq", "static bool", { (tub, efOrd, lm, lv, rm, rv) =>
          s"""
             |if (!$efOrd::eq($lm, $lv, $rm, $rv)) return false;
               """.stripMargin
        },
          "true")

        cb += "static bool neq(const char *l, const char *r) { return !eq(l, r); }"
        cb.end()

        ord

      case _: TInterval =>
        throw new CXXUnsupportedOperation()
    }
  }

  def ordering(tub: TranslationUnitBuilder, lp: PType, rp: PType): String = {
    typeOrdering.get((lp, rp)) match {
      case Some(o) => o
      case None =>
        val o = makeOrdering(tub, lp, rp)
        typeOrdering += (lp, rp) -> o
        o
    }
  }
}

class Emitter(fb: FunctionBuilder, nSpecialArgs: Int) {
  outer =>
  type E = ir.Env[EmitTriplet]

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
        present(s"${ v }f")
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
        assert(pType == cnsq.pType)
        assert(pType == altr.pType)

        val m = fb.variable("m", "bool")
        val v = fb.variable("v", typeToCXXType(pType))
        val tcond = emit(cond)
        val tcnsq = emit(cnsq)
        val taltr = emit(altr)

        triplet(
          s"""
             |${ tcond.setup }
             |${ m.define }
             |${ v.define }
             |if (${ tcond.m })
             |  $m = true;
             |else {
             |  if (${ tcond.v }) {
             |    ${ tcnsq.setup }
             |    $m = ${ tcnsq.m };
             |    if (!$m) {
             |      $v = ${ tcnsq.v };
             |    }
             |  } else {
             |    ${ taltr.setup }
             |    $m = ${ taltr.m };
             |    if (!$m) {
             |      $v = ${ taltr.v };
             |    }
             |  }
             |}
             |""".stripMargin,
          m.toString,
          v.toString)

      case ir.Let(name, value, body) =>
        val tvalue = emit(value)
        val m = fb.variable("let_m", "bool", tvalue.m)
        val v = fb.variable("let_v", typeToCXXType(value.pType))
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
            fb.translationUnitBuilder().include("<math.h>")
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

      case ir.ApplyComparisonOp(op, l, r) =>
        val lt = emit(l)
        val rt = emit(r)

        val o = fb.parent.ordering(l.pType, r.pType)

        val opf = op match {
          case ir.GT(_, _) => "gt"
          case ir.GTEQ(_, _) => "gteq"
          case ir.LT(_, _) => "lt"
          case ir.LTEQ(_, _) => "lteq"
          case ir.EQ(_, _) => "eq"
          case ir.NEQ(_, _) => "neq"
          case ir.EQWithNA(_, _) => "eq"
          case ir.NEQWithNA(_, _) => "neq"
        }

        if (op.strict) {
          triplet(Code(lt.setup, rt.setup),
            s"${ lt.m } || ${ rt.m }",
            s"$o::$opf(${ lt.v }, ${ rt.v })")
        } else {
          val lm = fb.variable("lm", "bool", lt.m)
          val lv = fb.variable("lv", typeToCXXType(l.pType))

          val rm = fb.variable("rm", "bool", rt.m)
          val rv = fb.variable("rv", typeToCXXType(r.pType))

          triplet(Code(lt.setup, rt.setup),
            "false",
            s"""
               |({
               |  ${ lm.define }
               |  ${ lv.define }
               |  if (!$lm)
               |    $lv = ${ lt.v };
               |  ${ rm.define }
               |  ${ rv.define }
               |  if (!$rm)
               |    $rv = ${ rt.v };
               |  ExtOrd<$o>::$opf($lm, $lv, $rm, $rv);
               |})
             """.stripMargin)
        }

      case ir.ArrayRef(a, i) =>
        val pContainer = a.pType.asInstanceOf[PContainer]
        val at = emit(a)
        val it = emit(i)

        val av = fb.variable("a", "const char *")
        val iv = fb.variable("i", "int")
        val len = fb.variable("len", "int")

        val m = fb.variable("m", "bool")

        var s = ir.Pretty(x)
        if (s.length > 100)
          s = s.substring(0, 100)
        s = StringEscapeUtils.escapeString(s)

        triplet(Code(at.setup, it.setup,
          s"""
             |${ m.define }
             |${ av.define }
             |${ iv.define }
             |${ len.define }
             |$m = ${ at.m }|| ${ it.m };
             |if (!$m) {
             |  $iv = ${ it.v };
             |  $av = ${ at.v };
             |  $len = ${ pContainer.cxxLoadLength(av.toString) };
             |  if ($iv < 0 || $iv >= $len) {
             |    NATIVE_ERROR(${ fb.getArg(0) }, 1005, "array index out of bounds: %d / %d.  IR: %s", $iv, $len, "$s");
             |    return nullptr;
             |  }
             |  $m = ${ pContainer.cxxIsElementMissing(av.toString, iv.toString) };
             |}
             |""".stripMargin), m.toString, pContainer.cxxLoadElement(av.toString, iv.toString))

      case ir.ArrayLen(a) =>
        val t = emit(a, env)
        triplet(t.setup, t.m, a.pType.asInstanceOf[PContainer].cxxLoadLength(t.v))

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
        val sb = new StagedBaseStructTripletBuilder(fb, pType.asInstanceOf[PBaseStruct])
        fields.foreach { x =>
          sb.add(emit(x))
        }
        sb.triplet()

      case ir.MakeStruct(fields) =>
        val sb = new StagedBaseStructTripletBuilder(fb, pType.asInstanceOf[PBaseStruct])
        fields.foreach { case (_, x) =>
          sb.add(emit(x))
        }
        sb.triplet()

      case ir.SelectFields(old, fields) =>
        val pStruct = pType.asInstanceOf[PStruct]
        val oldPStruct = old.pType.asInstanceOf[PStruct]
        val oldt = emit(old)
        val ov = fb.variable("old", typeToCXXType(oldPStruct), oldt.v)

        val sb = new StagedBaseStructTripletBuilder(fb, pStruct)
        fields.foreach { f =>
          val fieldIdx = oldPStruct.fieldIdx(f)
          sb.add(
            EmitTriplet(oldPStruct.fields(fieldIdx).typ, "",
              oldPStruct.cxxIsFieldMissing(ov.toString, fieldIdx),
              oldPStruct.cxxLoadField(ov.toString, fieldIdx)))
        }

        triplet(oldt.setup, oldt.m,
          s"""({
             |${ ov.define }
             |${ sb.body() }
             |${ sb.end() };
             |})
             |""".stripMargin)

      case ir.InsertFields(old, fields) =>
        val pStruct = pType.asInstanceOf[PStruct]
        val oldPStruct = old.pType.asInstanceOf[PStruct]
        val oldt = emit(old)
        val ov = fb.variable("old", typeToCXXType(oldPStruct), oldt.v)

        val fieldsMap = fields.toMap

        val sb = new StagedBaseStructTripletBuilder(fb, pStruct)
        pStruct.fields.foreach { f =>
          fieldsMap.get(f.name) match {
            case Some(fx) =>
              val fxt = emit(fx)
              sb.add(fxt)
            case None =>
              val fieldIdx = oldPStruct.fieldIdx(f.name)
              sb.add(
                EmitTriplet(f.typ, "",
                  oldPStruct.cxxIsFieldMissing(ov.toString, fieldIdx),
                  oldPStruct.cxxLoadField(ov.toString, fieldIdx)))
          }
        }

        triplet(oldt.setup, oldt.m,
          s"""({
             |${ ov.define }
             |${ sb.body() }
             |${ sb.end() };
             |})
             |""".stripMargin)


      case ir.ToArray(a) =>
        emit(a)

      case ir.ArrayFold(a, zero, accumName, valueName, body) =>
        val containerPType = a.pType.asInstanceOf[PContainer]
        val ae = emitArray(a, env)
        val am = fb.variable("am", "bool", ae.m)

        val zerot = emit(zero)

        val accm = fb.variable("accm", "bool")
        val accv = fb.variable("accv", typeToCXXType(zero.pType))
        val acct = EmitTriplet(zero.pType, "", accm.toString, accv.toString)

        triplet(
          s"""
             |${ ae.setup }
             |${ am.define }
             |${ accm.define }
             |${ accv.define }
             |if ($am)
             |  $accm = true;
             |else {
             |  ${ zerot.setup }
             |  $accm = ${ zerot.m };
             |  if (!$accm)
             |    $accv = ${ zerot.v };
             |  ${ ae.setupLen }
             |  ${
            ae.emit { case (m, v) =>
              val vm = fb.variable("vm", "bool")
              val vv = fb.variable("vv", typeToCXXType(containerPType.elementType))
              val vt = EmitTriplet(containerPType.elementType, "", vm.toString, vv.toString)

              val bodyt = emit(body, env.bind(accumName -> acct, valueName -> vt))

              // necessary because bodyt.v could be accm
              val bodym = fb.variable("bodym", "bool", bodyt.m)
              val bodyv = fb.variable("bodyv", typeToCXXType(body.pType))

              s"""
                 |${ vm.define }
                 |${ vv.define }
                 |$vm = $m;
                 |if (!$m)
                 |  $vv = $v;
                 |${ bodyt.setup }
                 |${ bodym.define }
                 |${ bodyv.define }
                 |if (!$bodym) {
                 |  $bodyv = ${ bodyt.v };
                 |  $accv = $bodyv;
                 |}
                 |$accm = $bodym;
                 |""".stripMargin
            }
          }
             |}
             |""".stripMargin,
          accm.toString, accv.toString)

      case _: ir.ArrayFilter | _: ir.ArrayRange | _: ir.ArrayMap =>
        val containerPType = x.pType.asInstanceOf[PContainer]

        val ae = emitArray(x, env)
        ae.length match {
          case Some(length) =>
            val sab = new StagedContainerBuilder(fb, fb.getArg(1).toString, containerPType)
            triplet(ae.setup, ae.m,
              s"""
                 |({
                 |  ${ ae.setupLen }
                 |  ${ sab.start(length) }
                 |  ${
                ae.emit { case (m, v) =>
                  s"""
                     |if (${ m })
                     |  ${ sab.setMissing() }
                     |else
                     |  ${ sab.add(v) }
                     |${ sab.advance() }
                     |""".stripMargin
                }
              }
                 |  ${ sab.end() };
                 |})
                 |""".stripMargin)

          case None =>
            val xs = fb.variable("xs", s"std::vector<${ typeToCXXType(containerPType.elementType) }>")
            val ms = fb.variable("ms", "std::vector<bool>")
            val i = fb.variable("i", "int")
            val sab = new StagedContainerBuilder(fb, fb.getArg(1).toString, containerPType)
            triplet(ae.setup, ae.m,
              s"""
                 |({
                 |  ${ ae.setupLen }
                 |  ${ xs.define }
                 |  ${ ms.define }
                 |  ${
                ae.emit { case (m, v) =>
                  s"""
                     |if (${ m }) {
                     |  $ms.push_back(true);
                     |  $xs.push_back(${ typeDefaultValue(containerPType.elementType) });
                     |} else {
                     |  $ms.push_back(false);
                     |  $xs.push_back($v);
                     |}
                     |""".stripMargin
                }
              }
                 |  ${ sab.start(s"$xs.size()") }
                 |  ${ i.define }
                 |  for ($i = 0; $i < $xs.size(); ++$i) {
                 |    if ($ms[$i])
                 |      ${ sab.setMissing() }
                 |   else
                 |      ${ sab.add(s"$xs[$i]") }
                 |    ${ sab.advance() }
                 |  }
                 |  ${ sab.end() };
                 |})
                 |""".stripMargin)
        }

      case ir.MakeArray(args, _) =>
        val sab = new StagedContainerBuilder(fb, fb.getArg(1).toString, pType.asInstanceOf[PArray])
        val sb = new ArrayBuilder[Code]

        sb += sab.start(s"${ args.length }")
        args.foreach { arg =>
          val argt = emit(arg)
          sb +=
            s"""
               |${ argt.setup }
               |if (${ argt.m })
               |  ${ sab.setMissing() }
               |else
               |  ${ sab.add(argt.v) }
               |${ sab.advance() }
               |""".stripMargin
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

  def emitArray(x: ir.IR, env: E): ArrayEmitter = {
    def emit(x: ir.IR, env: E = env): EmitTriplet = this.emit(x, env)

    val elemType = x.pType.asInstanceOf[PContainer].elementType

    x match {
      case ir.ArrayRange(start, stop, step) =>
        fb.translationUnitBuilder().include("<limits.h>")
        val startt = emit(start, env)
        val stopt = emit(stop, env)
        val stept = emit(step, env)

        val startv = fb.variable("start", "int", startt.v)
        val stopv = fb.variable("stop", "int", stopt.v)
        val stepv = fb.variable("step", "int", stept.v)

        val len = fb.variable("len", "int")
        val llen = fb.variable("llen", "long")

        var s = ir.Pretty(x)
        if (s.length > 100)
          s = s.substring(0, 100)
        s = StringEscapeUtils.escapeString(s)

        new ArrayEmitter(
          s"""
             |${ startt.setup }
             |${ stopt.setup }
             |${ stept.setup }
             |""".stripMargin,
          s"${ startt.m } || ${ stopt.m } || ${ stept.m }",
          s"""
             |${ startv.define }
             |${ stopv.define }
             |${ stepv.define }
             |${ len.define }
             |${ llen.define }
             |if ($stepv == 0) {
             |  NATIVE_ERROR(${ fb.getArg(0) }, 1006, "Array range step size cannot be 0.  IR: %s", "$s");
             |  return nullptr;
             |} else if ($stepv < 0)
             |  $llen = ($startv <= $stopv) ? 0l : ((long)$startv - (long)$stopv - 1l) / (long)(-$stepv) + 1l;
             |else
             |  $llen = ($startv >= $stopv) ? 0l : ((long)$stopv - (long)$startv - 1l) / (long)$stepv + 1l;
             |if ($llen > INT_MAX) {
             |  NATIVE_ERROR(${ fb.getArg(0) }, 1007, "Array range cannot have more than INT_MAX elements.  IR: %s", "$s");
             |  return nullptr;
             |} else
             |  $len = ($llen < 0) ? 0 : (int)$llen;
             |""".stripMargin, Some(len.toString)) {
          val i = fb.variable("i", "int", "0")
          val v = fb.variable("v", "int", startv.toString)

          def emit(f: (Code, Code) => Code): Code = {
            s"""
               |${ v.define }
               |for (${ i.define } $i < $len; ++$i) {
               |  ${ f("false", v.toString) }
               |  $v += $stepv;
               |}
               |""".stripMargin
          }
        }

      case ir.MakeArray(args, _) =>
        new ArrayEmitter("", "false", "", Some(args.length.toString)) {
          def emit(f: (Code, Code) => Code): Code = {
            val sb = new ArrayBuilder[Code]
            args.foreach { arg =>
              val argt = outer.emit(arg, env)
              sb += argt.setup
              sb += f(argt.m, argt.v)
            }
            sb.result().mkString
          }
        }

      case ir.ArrayFilter(a, name, cond) =>
        val ae = emitArray(a, env)
        val vm = fb.variable("m", "bool")
        val vv = fb.variable("v", typeToCXXType(elemType))
        val condt = outer.emit(cond,
          env.bind(name, EmitTriplet(elemType, "", vm.toString, vv.toString)))

        new ArrayEmitter(ae.setup, ae.m, ae.setupLen, None) {
          def emit(f: (Code, Code) => Code): Code = {
            ae.emit { (m2: Code, v2: Code) =>
              s"""
                 |{
                 |  ${ vm.define }
                 |  ${ vv.define }
                 |  $vm = $m2;
                 |  if (!$vm)
                 |    $vv = $v2;
                 |  ${ condt.setup }
                 |  if (!${ condt.m } && ${ condt.v }) {
                 |    ${ f(vm.toString, vv.toString) }
                 |  }
                 |}
                 |""".stripMargin
            }
          }
        }

      case ir.ArrayMap(a, name, body) =>
        val aElementPType = a.pType.asInstanceOf[PContainer].elementType
        val ae = emitArray(a, env)

        val vm = fb.variable("m", "bool")
        val vv = fb.variable("v", typeToCXXType(aElementPType))
        val bodyt = outer.emit(body,
          env.bind(name, EmitTriplet(aElementPType, "", vm.toString, vv.toString)))

        new ArrayEmitter(ae.setup, ae.m, ae.setupLen, ae.length) {
          def emit(f: (Code, Code) => Code): Code = {
            ae.emit { (m2: Code, v2: Code) =>
              s"""
                 |{
                 |  ${ vm.define }
                 |  ${ vv.define }
                 |  $vm = $m2;
                 |  if (!$vm)
                 |    $vv = $v2;
                 |  ${ bodyt.setup }
                 |  ${ f(bodyt.m, bodyt.v) }
                 |}
                 |""".stripMargin
            }
          }
        }

      case _ =>
        val pArray = x.pType.asInstanceOf[PArray]
        val t = emit(x, env)

        val a = fb.variable("a", "const char *", t.v)
        val len = fb.variable("len", "int", pArray.cxxLoadLength(a.toString))
        new ArrayEmitter(t.setup, t.m,
          s"""
             |${ a.define }
             |${ len.define }
             |""".stripMargin, Some(len.toString)) {
          val i = fb.variable("i", "int", "0")

          def emit(f: (Code, Code) => Code): Code = {
            s"""
               |for (${ i.define } $i < $len; ++$i) {
               |  ${
              f(pArray.cxxIsElementMissing(a.toString, i.toString),
                loadIRIntermediate(pArray.elementType, pArray.cxxElementAddress(a.toString, i.toString)))
            }
               |}
               |""".stripMargin
          }
        }
    }
  }
}
