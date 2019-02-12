package is.hail.cxx

import is.hail.expr.ir
import is.hail.expr.types._
import is.hail.expr.types.physical._
import is.hail.expr.types.virtual._
import is.hail.io.PackCodecSpec
import is.hail.utils.{ArrayBuilder, StringEscapeUtils}

import scala.collection.mutable

object Emit {
  def apply(fb: FunctionBuilder, nSpecialArgs: Int, x: ir.IR): EmitTriplet = {
    val emitter = new Emitter(fb, nSpecialArgs)
    emitter.emit(EmitRegion(fb, fb.getArg(0)), x, ir.Env.empty[EmitTriplet])
  }
}

class CXXUnsupportedOperation(msg: String = null) extends Exception(msg)

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

  def jHadoopConfiguration: Variable = fb.getArg(1)
  def emit(resultRegion: EmitRegion, x: ir.IR, env: E): EmitTriplet = {
    def triplet(setup: Code, m: Code, v: Code): EmitTriplet =
      EmitTriplet(x.pType, setup, m, v, resultRegion)

    def present(v: Code): EmitTriplet = triplet("", "false", v)

    def emit(x: ir.IR, env: E = env): EmitTriplet = this.emit(resultRegion, x, env)

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
        val tbody = emit(body, env.bind(name, EmitTriplet(value.pType, "", m.toString, v.toString, resultRegion)))

        triplet(
          Code(tvalue.setup, m.define, v.define, s"if (!$m) { $v = ${ tvalue.v }; }", tbody.setup),
          tbody.m,
          tbody.v)

      case ir.Ref(name, _) =>
        env.lookup(name)

      case ir.ApplyBinaryPrimOp(op, l, r) =>

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
          case ir.BitAnd() => s"${ lt.v } & ${ rt.v }"
          case ir.BitOr() => s"${ lt.v } | ${ rt.v }"
          case ir.BitXOr() => s"${ lt.v } ^ ${ rt.v }"
          case ir.LeftShift() => s"${ lt.v } << ${ rt.v }"
          case ir.RightShift() => s"${ lt.v } >> ${ rt.v }"
          case ir.LogicalRightShift() =>
            l.typ match {
              case _: TInt32 => s"(int)((unsigned int)${ lt.v } >> ${ rt.v })"
              case _: TInt64 => s"(long)((unsigned long)${ lt.v } >> ${ rt.v })"
            }

        }

        triplet(Code(lt.setup, rt.setup), s"${ lt.m } || ${ rt.m }", v)

      case ir.ApplyUnaryPrimOp(op, x) =>
        val t = emit(x)

        val v = op match {
          case ir.Bang() => s"! ${ t.v }"
          case ir.Negate() => s"- ${ t.v }"
          case ir.BitNot() => s"~ ${ t.v }"
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
          case ir.Compare(_, _) => "compare"
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
             |    ${ fb.nativeError(s"array index out of bounds: %d / %d.  IR: $s", iv.toString, len.toString) }
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
        val sb = resultRegion.structBuilder(fb, pType.asInstanceOf[PBaseStruct])
        fields.foreach { x =>
          sb.add(emit(x))
        }
        sb.triplet()

      case ir.MakeStruct(fields) =>
        val sb = resultRegion.structBuilder(fb, pType.asInstanceOf[PBaseStruct])
        fields.foreach { case (_, x) =>
          sb.add(emit(x))
        }
        sb.triplet()

      case ir.SelectFields(old, fields) =>
        val pStruct = pType.asInstanceOf[PStruct]
        val oldPStruct = old.pType.asInstanceOf[PStruct]
        val oldt = emit(old)
        val ov = fb.variable("old", typeToCXXType(oldPStruct), oldt.v)

        val sb = resultRegion.structBuilder(fb, pStruct)
        fields.foreach { f =>
          val fieldIdx = oldPStruct.fieldIdx(f)
          sb.add(
            EmitTriplet(oldPStruct.fields(fieldIdx).typ, "",
              oldPStruct.cxxIsFieldMissing(ov.toString, fieldIdx),
              oldPStruct.cxxLoadField(ov.toString, fieldIdx), resultRegion))
        }

        triplet(oldt.setup, oldt.m,
          s"""({
             |${ ov.define }
             |${ sb.body() }
             |${ sb.end() };
             |})
             |""".stripMargin)

      case ir.InsertFields(old, fields, fieldOrder) =>
        val pStruct = pType.asInstanceOf[PStruct]
        val oldPStruct = old.pType.asInstanceOf[PStruct]
        val oldt = emit(old)
        val ov = fb.variable("old", typeToCXXType(oldPStruct), oldt.v)

        val fieldsMap = fields.toMap

        val sb = resultRegion.structBuilder(fb, pStruct)
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
                  oldPStruct.cxxLoadField(ov.toString, fieldIdx), resultRegion)) //FIXME
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
        val ae = emitArray(resultRegion, a, env, sameRegion = false)
        val am = fb.variable("am", "bool", ae.m)

        val zerot = emit(zero)

        val accm = fb.variable("accm", "bool")
        val accv = fb.variable("accv", typeToCXXType(zero.pType))
        val acct = EmitTriplet(zero.pType, "", accm.toString, accv.toString, resultRegion)

        triplet(
          s"""
             |${ ae.setup }
             |${ am.define }
             |${ accm.define }
             |${ accv.define }
             |if ($am) {
             |  $accm = true;
             |} else {
             |  ${ zerot.setup }
             |  $accm = ${ zerot.m };
             |  if (!$accm)
             |    $accv = ${ zerot.v };
             |  ${ ae.setupLen }
             |  ${
            ae.emit { case (m, v) =>
              val vm = fb.variable("vm", "bool")
              val vv = fb.variable("vv", typeToCXXType(containerPType.elementType))
              val vt = EmitTriplet(containerPType.elementType, "", vm.toString, vv.toString, ae.arrayRegion)

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

      case _: ir.ArrayFilter | _: ir.ArrayRange | _: ir.ArrayMap | _: ir.ArrayFlatMap | _: ir.MakeArray =>
        val containerPType = x.pType.asInstanceOf[PContainer]
        val useOneRegion = !containerPType.elementType.isPrimitive

        val ae = emitArray(resultRegion, x, env, useOneRegion)
        ae.length match {
          case Some(length) =>
            val sab = resultRegion.arrayBuilder(fb, containerPType)
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
            val sab = resultRegion.arrayBuilder(fb, containerPType)
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

      case x@ir.ApplyIR(_, _, _) =>
        // FIXME small only
        emit(x.explicitNode)

      case ir.In(i, _) =>
        EmitTriplet(x.pType, "",
          "false", fb.getArg(nSpecialArgs + i).toString, null)

      case _ =>
        throw new CXXUnsupportedOperation(ir.Pretty(x))
    }
  }

  def emitArray(resultRegion: EmitRegion, x: ir.IR, env: E, sameRegion: Boolean): ArrayEmitter = {

    val elemType = x.pType.asInstanceOf[PContainer].elementType

    x match {
      case ir.ArrayRange(start, stop, step) =>
        fb.translationUnitBuilder().include("<limits.h>")
        val startt = emit(resultRegion, start, env)
        val stopt = emit(resultRegion, stop, env)
        val stept = emit(resultRegion, step, env)

        val startv = fb.variable("start", "int", startt.v)
        val stopv = fb.variable("stop", "int", stopt.v)
        val stepv = fb.variable("step", "int", stept.v)

        val len = fb.variable("len", "int")
        val llen = fb.variable("llen", "long")

        var s = ir.Pretty(x)
        if (s.length > 100)
          s = s.substring(0, 100)
        s = StringEscapeUtils.escapeString(s)


        val arrayRegion = EmitRegion.from(resultRegion, sameRegion)
        new ArrayEmitter(
          s"""
             |${ startt.setup }
             |${ stopt.setup }
             |${ stept.setup }
             |""".stripMargin,
          s"(${ startt.m } || ${ stopt.m } || ${ stept.m })",
          s"""
             |${ startv.define }
             |${ stopv.define }
             |${ stepv.define }
             |${ len.define }
             |${ llen.define }
             |if ($stepv == 0) {
             |  ${ fb.nativeError("Array range step size cannot be 0.  IR: %s".format(s)) }
             |} else if ($stepv < 0)
             |  $llen = ($startv <= $stopv) ? 0l : ((long)$startv - (long)$stopv - 1l) / (long)(-$stepv) + 1l;
             |else
             |  $llen = ($startv >= $stopv) ? 0l : ((long)$stopv - (long)$startv - 1l) / (long)$stepv + 1l;
             |if ($llen > INT_MAX) {
             |  ${ fb.nativeError("Array range cannot have more than INT_MAX elements.  IR: %s".format(s)) }
             |} else
             |  $len = ($llen < 0) ? 0 : (int)$llen;
             |""".stripMargin, Some(len.toString), arrayRegion) {

          def emit(f: (Code, Code) => Code): Code = {
            val i = fb.variable("i", "int", "0")
            val v = fb.variable("v", "int", startv.toString)
            s"""
               |${ v.define }
               |for (${ i.define } $i < $len; ++$i) {
               |  ${ arrayRegion.defineIfUsed(sameRegion) }
               |  ${ f("false", v.toString) }
               |  $v += $stepv;
               |}
               |""".stripMargin
          }
        }

      case ir.MakeArray(args, t) =>
        val arrayRegion = EmitRegion.from(resultRegion, sameRegion)
        val triplets = args.map { arg => outer.emit(arrayRegion, arg, env) }
        new ArrayEmitter("", "false", "", Some(args.length.toString), arrayRegion) {
          def emit(f: (Code, Code) => Code): Code = {
            val sb = new ArrayBuilder[Code]
            val m = fb.variable("argm", "bool")
            val v = fb.variable("argv", typeToCXXType(t.elementType.physicalType))
            val cont = f(m.toString, v.toString)

            triplets.foreach { argt =>
              sb +=
                s"""
                   |{
                   |${ arrayRegion.defineIfUsed(sameRegion) }
                   |${ argt.setup }
                   |${ m.defineWith(argt.m) }
                   |${ v.defineWith(argt.v) }
                   |$cont
                   |}
                 """.stripMargin
            }
            sb.result().mkString
          }
        }

      case ir.ArrayFilter(a, name, cond) =>
        val ae = emitArray(resultRegion, a, env, sameRegion)
        val arrayRegion = ae.arrayRegion
        val vm = fb.variable("m", "bool")
        val vv = fb.variable("v", typeToCXXType(elemType))
        val condt = outer.emit(arrayRegion, cond,
          env.bind(name, EmitTriplet(elemType, "", vm.toString, vv.toString, arrayRegion)))

        new ArrayEmitter(ae.setup, ae.m, ae.setupLen, None, arrayRegion) {
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
        val ae = emitArray(resultRegion, a, env, sameRegion)
        val arrayRegion = ae.arrayRegion

        val vm = fb.variable("m", "bool")
        val vv = fb.variable("v", typeToCXXType(aElementPType))
        val bodyt = outer.emit(arrayRegion, body,
          env.bind(name, EmitTriplet(aElementPType, "", vm.toString, vv.toString, arrayRegion)))

        new ArrayEmitter(ae.setup, ae.m, ae.setupLen, ae.length, arrayRegion) {
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

      case ir.ArrayFlatMap(a, name, body) =>
        val aElementPType = a.pType.asInstanceOf[PContainer].elementType

        val ae = emitArray(resultRegion, a, env, sameRegion)
        val arrayRegion = ae.arrayRegion

        val vm = fb.variable("m", "bool")
        val vv = fb.variable("v", typeToCXXType(aElementPType))
        val bodyt = outer.emitArray(arrayRegion, body,
          env.bind(name, EmitTriplet(aElementPType, "", vm.toString, vv.toString, arrayRegion)), sameRegion)

        new ArrayEmitter(ae.setup, ae.m, ae.setupLen, None, bodyt.arrayRegion) {
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
                 |  if (!${ bodyt.m }) {
                 |    ${ bodyt.setupLen }
                 |    ${ bodyt.emit(f) }
                 |  }
                 |}
                 |""".stripMargin
            }
          }
        }

      case x@ir.ReadPartition(pathIR, spec, _) =>
        assert(nSpecialArgs > 1)
        val arrayRegion = EmitRegion.from(resultRegion, sameRegion)
        arrayRegion.use()
        val dec = spec.buildNativeDecoderClass(x.pType, x.pType, fb.translationUnitBuilder())
        val path = emit(resultRegion, pathIR, env)

        new ArrayEmitter(path.setup, path.m, "", None, arrayRegion) {
          def emit(f: (Code, Code) => Code): Code = {
            s"""
               |InputStream is {};
               |$dec dec { std::make_shared<InputStream>(${path.v}, $jHadoopConfiguration) };
               |while (dec.read_byte()) {
               |  ${ arrayRegion.defineIfUsed(sameRegion) }
               |  ${ f("false", s"dec.decode_row($arrayRegion)") }
               |}
             """.stripMargin
          }
        }

      case _ =>
        val pArray = x.pType.asInstanceOf[PArray]
        val t = emit(resultRegion, x, env)
        val arrayRegion = EmitRegion.from(resultRegion, sameRegion)

        val a = fb.variable("a", "const char *", t.v)
        val len = fb.variable("len", "int", pArray.cxxLoadLength(a.toString))
        new ArrayEmitter(t.setup, t.m,
          s"""
             |${ a.define }
             |${ len.define }
             |""".stripMargin, Some(len.toString), arrayRegion) {
          val i = fb.variable("i", "int", "0")

          def emit(f: (Code, Code) => Code): Code = {
            s"""
               |for (${ i.define } $i < $len; ++$i) {
               |  ${ arrayRegion.defineIfUsed(sameRegion) }
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
