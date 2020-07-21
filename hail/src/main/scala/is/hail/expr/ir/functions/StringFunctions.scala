package is.hail.expr.ir.functions

import is.hail.annotations.{Region, StagedRegionValueBuilder}
import is.hail.asm4s
import is.hail.asm4s._
import is.hail.expr.ir._
import is.hail.types.physical._
import is.hail.types.virtual._
import is.hail.utils._
import java.util.Locale
import java.time.{Instant, ZoneId}
import java.time.temporal.ChronoField

import is.hail.expr.JSONAnnotationImpex
import org.apache.spark.sql.Row
import org.json4s.JValue
import org.json4s.jackson.JsonMethods

import scala.collection.mutable

object StringFunctions extends RegistryFunctions {

  def reverse(s: String): String = {
    val sb = new StringBuilder
    sb.append(s)
    sb.reverseContents().result()
  }

  def upper(s: String): String = s.toUpperCase

  def lower(s: String): String = s.toLowerCase

  def strip(s: String): String = s.trim()

  def contains(s: String, t: String): Boolean = s.contains(t)

  def startswith(s: String, t: String): Boolean = s.startsWith(t)

  def endswith(s: String, t: String): Boolean = s.endsWith(t)

  def firstMatchIn(s: String, regex: String): IndexedSeq[String] = {
    regex.r.findFirstMatchIn(s).map(_.subgroups.toArray.toFastIndexedSeq).orNull
  }

  def regexMatch(regex: String, s: String): Boolean = regex.r.findFirstIn(s).isDefined

  def concat(s: String, t: String): String = s + t

  def replace(str: String, pattern1: String, pattern2: String): String =
    str.replaceAll(pattern1, pattern2)

  def split(s: String, p: String): IndexedSeq[String] = s.split(p, -1)

  def translate(s: String, d: Map[String, String]): String = {
    val charD = new mutable.HashMap[Char, String]
    d.foreach { case (k, v) =>
      if (k.length != 1)
        fatal(s"translate: mapping keys must be one character, found '$k'")
        charD += ((k(0), v))
    }

    val sb = new StringBuilder
    var i = 0
    while (i < s.length) {
      val charI = s(i)
      charD.get(charI) match {
        case Some(replacement) => sb.append(replacement)
        case None => sb.append(charI)
      }
      i += 1
    }
    sb.result()
  }

  def splitLimited(s: String, p: String, n: Int): IndexedSeq[String] = s.split(p, n)

  def arrayMkString(a: IndexedSeq[String], sep: String): String = a.mkString(sep)

  def setMkString(s: Set[String], sep: String): String = s.mkString(sep)

  def escapeString(s: String): String = StringEscapeUtils.escapeString(s)

  def softBounds(i: IR, len: IR): IR =
    If(i < -len, 0, If(i < 0, i + len, If(i >= len, len, i)))


  private val locale: Locale = Locale.US

  def strftime(fmtStr: String, epochSeconds: Long, zoneId: String): String =
    DateFormatUtils.parseDateFormat(fmtStr, locale).withZone(ZoneId.of(zoneId))
      .format(Instant.ofEpochSecond(epochSeconds))

  def strptime(timeStr: String, fmtStr: String, zoneId: String): Long =
    DateFormatUtils.parseDateFormat(fmtStr, locale).withZone(ZoneId.of(zoneId))
      .parse(timeStr)
      .getLong(ChronoField.INSTANT_SECONDS)

  def registerAll(): Unit = {
    val thisClass = getClass

    registerPCode1("length", TString, TInt32, (_: Type, _: PType) => PInt32()) { case (r: EmitRegion, rt, s: PStringCode) =>
      PCode(rt, s.loadString().invoke[Int]("length"))
    }

    registerCode3("substring", TString, TInt32, TInt32, TString, {
      (_: Type, _: PType, _: PType, _: PType) => PCanonicalString()
    }) {
      case (r: EmitRegion, rt, (sT: PString, s: Code[Long]), (startT, start: Code[Int]), (endT, end: Code[Int])) =>
      unwrapReturn(r, rt)(asm4s.coerce[String](wrapArg(r, sT)(s)).invoke[Int, Int, String]("substring", start, end))
    }

    registerIR3("slice", TString, TInt32, TInt32, TString) { (_, str, start, end) =>
      val len = Ref(genUID(), TInt32)
      val s = Ref(genUID(), TInt32)
      val e = Ref(genUID(), TInt32)
      Let(len.name, invoke("length", TInt32, str),
        Let(s.name, softBounds(start, len),
          Let(e.name, softBounds(end, len),
            invoke("substring", TString, str, s, If(e < s, s, e)))))
    }

    registerIR2("index", TString, TInt32, TString) { (_, s, i) =>
      val len = Ref(genUID(), TInt32)
      val idx = Ref(genUID(), TInt32)
      Let(len.name, invoke("length", TInt32, s),
        Let(idx.name,
          If((i < -len) || (i >= len),
            Die(invoke("concat", TString,
              Str("string index out of bounds: "),
              invoke("concat", TString,
                invoke("str", TString, i),
                invoke("concat", TString, Str(" / "), invoke("str", TString, len)))), TInt32),
            If(i < 0, i + len, i)),
        invoke("substring", TString, s, idx, idx + 1)))
    }

    registerIR2("sliceRight", TString, TInt32, TString) { (_, s, start) => invoke("slice", TString, s, start, invoke("length", TInt32, s)) }
    registerIR2("sliceLeft", TString, TInt32, TString) { (_, s, end) => invoke("slice", TString, s, I32(0), end) }

    registerCode1("str", tv("T"), TString, (_: Type, _: PType) => PCanonicalString()) { case (r, rt, (aT, a)) =>
      val annotation = boxArg(r, aT)(a)
      val str = r.mb.getType(aT.virtualType).invoke[Any, String]("str", annotation)
      unwrapReturn(r, rt)(str)
    }

    registerEmitCode1("showStr", tv("T"), TString, {
      (_: Type, _: PType) => PCanonicalString(true)
    }) { case (r, rt, a) =>
      val annotation = Code(a.setup, a.m).muxAny(Code._null(boxedTypeInfo(a.pt)), boxArg(r, a.pt)(a.v))
      val str = r.mb.getType(a.pt.virtualType).invoke[Any, String]("showStr", annotation)
      EmitCode.present(PCode(rt, unwrapReturn(r, rt)(str)))
    }

    registerEmitCode2("showStr", tv("T"), TInt32, TString, {
      (_: Type, _: PType, truncType: PType) => PCanonicalString(truncType.required)
    }) { case (r, rt, a, trunc) =>
      val annotation = Code(a.setup, a.m).muxAny(Code._null(boxedTypeInfo(a.pt)), boxArg(r, a.pt)(a.v))
      val str = r.mb.getType(a.pt.virtualType).invoke[Any, Int, String]("showStr", annotation, trunc.value[Int])
      EmitCode(trunc.setup, trunc.m, PCode(rt, unwrapReturn(r, rt)(str)))
    }

    registerEmitCode1("json", tv("T"), TString, (_: Type, _: PType) => PCanonicalString(true)) { case (r, rt, a) =>
      val bti = boxedTypeInfo(a.pt)
      val annotation = Code(a.setup, a.m).muxAny(Code._null(bti), boxArg(r, a.pt)(a.v))
      val json = r.mb.getType(a.pt.virtualType).invoke[Any, JValue]("toJSON", annotation)
      val str = Code.invokeScalaObject1[JValue, String](JsonMethods.getClass, "compact", json)
      EmitCode(Code._empty, false, PCode(rt, unwrapReturn(r, rt)(str)))
    }

    registerWrappedScalaFunction1("reverse", TString, TString, (_: Type, _: PType) => PCanonicalString())(thisClass,"reverse")
    registerWrappedScalaFunction1("upper", TString, TString, (_: Type, _: PType) => PCanonicalString())(thisClass,"upper")
    registerWrappedScalaFunction1("lower", TString, TString, (_: Type, _: PType) => PCanonicalString())(thisClass,"lower")
    registerWrappedScalaFunction1("strip", TString, TString, (_: Type, _: PType) => PCanonicalString())(thisClass,"strip")
    registerWrappedScalaFunction2("contains", TString, TString, TBoolean, {
      case (_: Type, _: PType, _: PType) => PBoolean()
    })(thisClass, "contains")
    registerWrappedScalaFunction2("translate", TString, TDict(TString, TString), TString, {
      case (_: Type, _: PType, _: PType) => PCanonicalString()
    })(thisClass, "translate")
    registerWrappedScalaFunction2("startswith", TString, TString, TBoolean, {
      case (_: Type, _: PType, _: PType) => PBoolean()
    })(thisClass, "startswith")
    registerWrappedScalaFunction2("endswith", TString, TString, TBoolean, {
      case (_: Type, _: PType, _: PType) => PBoolean()
    })(thisClass, "endswith")
    registerWrappedScalaFunction2("regexMatch", TString, TString, TBoolean, {
      case (_: Type, _: PType, _: PType) => PBoolean()
    })(thisClass, "regexMatch")
    registerWrappedScalaFunction2("concat", TString, TString, TString, {
      case (_: Type, _: PType, _: PType) => PCanonicalString()
    })(thisClass, "concat")

    registerWrappedScalaFunction2("split", TString, TString, TArray(TString), {
      case (_: Type, _: PType, _: PType) =>
        PCanonicalArray(PCanonicalString(true))
    })(thisClass, "split")

    registerWrappedScalaFunction3("split", TString, TString, TInt32, TArray(TString), {
      case (_: Type, _: PType, _: PType, _: PType) =>
        PCanonicalArray(PCanonicalString(true))
    })(thisClass, "splitLimited")

    registerWrappedScalaFunction3("replace", TString, TString, TString, TString, {
      case (_: Type, _: PType, _: PType, _: PType) => PCanonicalString()
    })(thisClass, "replace")

    registerWrappedScalaFunction2("mkString", TSet(TString), TString, TString, {
      case (_: Type, _: PType, _: PType) => PCanonicalString()
    })(thisClass, "setMkString")

    registerWrappedScalaFunction2("mkString", TArray(TString), TString, TString, {
      case (_: Type, _: PType, _: PType) => PCanonicalString()
    })(thisClass, "arrayMkString")

    registerEmitCode2("firstMatchIn", TString, TString, TArray(TString), {
      case(_: Type, _: PType, _: PType) => PCanonicalArray(PCanonicalString(true))
    }) {
      case (er: EmitRegion, rt: PArray, s: EmitCode, r: EmitCode) =>
      val out: LocalRef[IndexedSeq[String]] = er.mb.newLocal[IndexedSeq[String]]()

      val srvb: StagedRegionValueBuilder = new StagedRegionValueBuilder(er, rt)
      val len: LocalRef[Int] = er.mb.newLocal[Int]()
      val elt: LocalRef[String] = er.mb.newLocal[String]()

      val setup = Code(s.setup, r.setup)
      val missing = s.m || r.m || Code(
        out := Code.invokeScalaObject2[String, String, IndexedSeq[String]](
          thisClass, "firstMatchIn",
          asm4s.coerce[String](wrapArg(er, s.pt)(s.value[Long])),
          asm4s.coerce[String](wrapArg(er, r.pt)(r.value[Long]))),
        out.isNull)
      val value =
        out.ifNull(
          defaultValue(rt),
          Code(
            len := out.invoke[Int]("size"),
            srvb.start(len),
            Code.whileLoop(srvb.arrayIdx < len,
              elt := out.invoke[Int, String]("apply", srvb.arrayIdx),
              elt.ifNull(
                srvb.setMissing(),
                srvb.addString(elt)),
              srvb.advance()),
            srvb.end()))

      EmitCode(setup, missing, PCode(rt, value))
    }

    registerEmitCode2("hamming", TString, TString, TInt32, {
      case(_: Type, _: PType, _: PType) => PInt32()
    }) { case (r: EmitRegion, rt, e1: EmitCode, e2: EmitCode) =>
      EmitCode.fromI(r.mb) { cb =>
        e1.toI(cb).flatMap(cb) { case (sc1: PStringCode) =>
          e2.toI(cb).flatMap(cb) { case (sc2: PStringCode) =>
            val n = cb.newLocal("hamming_n", 0)
            val i = cb.newLocal("hamming_i", 0)

            val v1 = sc1.asBytes().memoize(cb, "hamming_bytes_1")
            val v2 = sc2.asBytes().memoize(cb, "hamming_bytes_2")

            val m = v1.loadLength().cne(v2.loadLength())

            IEmitCode(cb, m, {
              cb.whileLoop(i < v1.loadLength(), {
                cb.ifx(v1.loadByte(i).cne(v2.loadByte(i)),
                  cb.assign(n, n + 1))
                cb.assign(i, i + 1)
              })
              PCode(rt, n)
            })
          }
        }
      }
    }

    registerWrappedScalaFunction1("escapeString", TString, TString, (_: Type, _: PType) => PCanonicalString())(thisClass, "escapeString")
    registerWrappedScalaFunction3("strftime", TString, TInt64, TString, TString, {
      case(_: Type, _: PType, _: PType, _: PType) => PCanonicalString()
    })(thisClass, "strftime")
    registerWrappedScalaFunction3("strptime", TString, TString, TString, TInt64, {
      case (_: Type, _: PType, _: PType, _: PType) => PInt64()
    })(thisClass, "strptime")

    registerPCode("parse_json", Array(TString), TTuple(tv("T")),
      (rType: Type, _: Seq[PType]) => PType.canonical(rType, true), typeParameters = Array(tv("T"))) { case (er, resultType, Array(s: PStringCode)) =>

      PCode(resultType, StringFunctions.unwrapReturn(er, resultType)(
        Code.invokeScalaObject2[String, Type, Row](JSONAnnotationImpex.getClass, "irImportAnnotation",
          s.loadString(), er.mb.ecb.getType(resultType.virtualType.asInstanceOf[TTuple].types(0)))
      ))
    }
  }
}
