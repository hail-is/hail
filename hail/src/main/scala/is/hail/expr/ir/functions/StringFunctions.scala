package is.hail.expr.ir.functions

import is.hail.annotations.Region
import is.hail.asm4s._
import is.hail.expr.JSONAnnotationImpex
import is.hail.expr.ir._
import is.hail.types.physical.stypes._
import is.hail.types.physical.stypes.concrete.{SJavaArrayString, SJavaArrayStringSettable, SJavaArrayStringValue, SJavaString}
import is.hail.types.physical.stypes.interfaces._
import is.hail.types.physical.stypes.primitives.{SBoolean, SInt32, SInt64}
import is.hail.types.virtual._
import is.hail.utils._
import org.apache.spark.sql.Row
import org.json4s.JValue
import org.json4s.jackson.JsonMethods

import java.time.temporal.ChronoField
import java.time.{Instant, ZoneId}
import java.util.Locale
import java.util.regex.{Matcher, Pattern}
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

  def firstMatchIn(s: String, regex: String): Array[String] = {
    regex.r.findFirstMatchIn(s).map(_.subgroups.toArray).orNull
  }

  def regexMatch(regex: String, s: String): Boolean = regex.r.findFirstIn(s).isDefined

  def regexFullMatch(regex: String, s: String): Boolean = s.matches(regex)

  def concat(s: String, t: String): String = s + t

  def replace(str: String, pattern1: String, pattern2: String): String =
    str.replaceAll(pattern1, pattern2)
   
  def split(s: String, p: String): Array[String] = s.split(p, -1)

  def translate(s: String, d: Map[String, String]): String = {
    val charD = new mutable.HashMap[Char, String]
    d.foreach { case (k, v) =>
      if (k.length != 1)
        fatal(s"translate: mapping keys must be one character, found '$k'", ErrorIDs.NO_ERROR)
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

  def splitLimited(s: String, p: String, n: Int): Array[String] = s.split(p, n)

  def arrayMkString(a: Array[String], sep: String): String = a.mkString(sep)

  def setMkString(s: Set[String], sep: String): String = s.mkString(sep)

  def escapeString(s: String): String = StringEscapeUtils.escapeString(s)

  def addValueOrNull(ab: StringArrayBuilder, value: String, missingValues: Array[String]): Unit = {
    var i = 0
    while (i < missingValues.length) {
      if (missingValues(i) == value) {
        ab += null
        return
      }
      i += 1
    }
    ab += value
  }

  def matchPattern(s: String, i: Int, m: Matcher): Int = {
    m.region(i, s.length)
    if (m.lookingAt())
      m.end() - m.start()
    else
      -1
  }

  def generateSplitQuotedRegex(
    cb: EmitCodeBuilder,
    string: Value[String],
    separator: Either[Value[Char], Value[String]],
    quoteChar: Option[Value[Char]],
    missingSV: SIndexableValue,
    errorID: Value[Int]
  ): Value[Array[String]] = {

    // note: it will be inefficient to convert a SIndexablePointer to SJavaArrayString to split each line.
    // We should really choose SJavaArrayString as the stype for a literal if used in a place like this,
    // but this is a non-local stype decision that is hard in the current system
    val missing: Value[Array[String]] = missingSV.st match {
      case SJavaArrayString(elementRequired) => missingSV.asInstanceOf[SJavaArrayStringSettable].array
      case _ =>
        val mb = cb.emb.ecb.newEmitMethod("convert_region_to_str_array", FastSeq(missingSV.st.paramType), arrayInfo[String])
        mb.emitWithBuilder[Array[String]] { cb =>
          val sv = mb.getSCodeParam(1).asIndexable
          val m = cb.newLocal[Array[String]]("missingvals", Code.newArray[String](sv.loadLength()))
          sv.forEachDefined(cb) { case (cb, idx, sc) => cb += (m(idx) = sc.asString.loadString(cb)) }
          m
        }
        cb.newLocal[Array[String]]("missing_arr", cb.invokeCode(mb, missingSV))
    }

    // lazy field reused across calls to split functions
    val ab = cb.emb.getOrDefineLazyField[StringArrayBuilder](Code.newInstance[StringArrayBuilder, Int](16), "generate_split_quoted_regex_ab")
    cb += ab.invoke[Unit]("clear")

    // takes the current position and current char value, returns the number of matching chars
    // in the separator, or -1 if not a separator
    val getPatternMatch: (Value[Int], Value[Char]) => Value[Int] = {
      val x = cb.newLocal[Int]("sepCharMatch");
      separator match {
        case Left(sepChar) =>
          (_: Value[Int], char: Value[Char]) => {
            cb.ifx(char.ceq(sepChar), cb.assign(x, 1), cb.assign(x, -1));
            x
          }
        case Right(regex) =>
          val m = cb.newLocal[Matcher]("matcher",
            Code.invokeStatic1[Pattern, String, Pattern]("compile", regex)
              .invoke[CharSequence, Matcher]("matcher", string));
          (idx: Value[Int], _: Value[Char]) => {
            cb.assign(x, Code.invokeScalaObject3[String, Int, Matcher, Int](
              StringFunctions.getClass, "matchPattern", string, idx, m));
            x
          }
      }
    }

    val i = cb.newLocal[Int]("i", 0)
    val lastFieldStart = cb.newLocal[Int]("lastfieldstart", 0)

    def addValueOrNA(cb: EmitCodeBuilder, endIdx: Code[Int]): Unit = {
      cb += Code.invokeScalaObject3[StringArrayBuilder, String, Array[String], Unit](
        StringFunctions.getClass, "addValueOrNull", ab, string.invoke[Int, Int, String]("substring", lastFieldStart, endIdx), missing)
    }

    val LreturnWithoutAppending = CodeLabel()

    cb.whileLoop(i < string.length(), {
      val c = cb.newLocal[Char]("c", string(i))

      val l = getPatternMatch(i, c)
      cb.ifx(l.cne(-1), {
        addValueOrNA(cb, i)
        cb.assign(i, i + l) // skip delim
        cb.assign(lastFieldStart, i)
      }, {
        quoteChar match {
          case Some(qc) =>
            cb.ifx(c.ceq(qc), {
              cb.ifx(i.cne(lastFieldStart),
                cb._fatalWithError(errorID, "opening quote character '", qc.toS, "' not at start of field"))
              cb.assign(i, i + 1) // skip quote
              cb.assign(lastFieldStart, i)

              cb.whileLoop(i < string.length() && string(i).cne(qc), {
                cb.assign(i, i + 1)
              })

              addValueOrNA(cb, i)

              cb.ifx(i.ceq(string.length()),
                cb._fatalWithError(errorID, "missing terminating quote character '", qc.toS, "'"))
              cb.assign(i, i + 1) // skip quote

              cb.ifx(i < string.length, {
                cb.assign(c, string(i))
                val l = getPatternMatch(i, c)
                cb.ifx(l.ceq(-1), {
                  cb._fatalWithError(errorID, "terminating quote character '", qc.toS, "' not at end of field")
                })
                cb.assign(i, i + l) // skip delim
                cb.assign(lastFieldStart, i)
              }, {
                cb.goto(LreturnWithoutAppending)
              })
            }, {
              cb.assign(i, i + 1)
            })
          case None =>
            cb.assign(i, i + 1)
        }
      })
    })

    addValueOrNA(cb, string.length())
    cb.define(LreturnWithoutAppending)
    cb.memoize(ab.invoke[Array[String]]("result"), "generateSplitQuotedRegexResult")
  }

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

    registerSCode1("length", TString, TInt32, (_: Type, _: SType) => SInt32) { case (r: EmitRegion, cb, _, s: SStringValue, _) =>
      primitive(cb.memoize(s.loadString(cb).invoke[Int]("length")))
    }

    registerSCode3("substring", TString, TInt32, TInt32, TString, {
      (_: Type, _: SType, _: SType, _: SType) => SJavaString
    }) {
      case (r: EmitRegion, cb, st: SJavaString.type, s, start, end, _) =>
        val str = s.asString.loadString(cb).invoke[Int, Int, String]("substring", start.asInt.value, end.asInt.value)
        st.construct(cb, str)
    }

    registerIR3("slice", TString, TInt32, TInt32, TString) { (_, str, start, end, _) =>
      val len = Ref(genUID(), TInt32)
      val s = Ref(genUID(), TInt32)
      val e = Ref(genUID(), TInt32)
      Let(len.name, invoke("length", TInt32, str),
        Let(s.name, softBounds(start, len),
          Let(e.name, softBounds(end, len),
            invoke("substring", TString, str, s, If(e < s, s, e)))))
    }

    registerIR2("index", TString, TInt32, TString) { (_, s, i, errorID) =>
      val len = Ref(genUID(), TInt32)
      val idx = Ref(genUID(), TInt32)
      Let(len.name, invoke("length", TInt32, s),
        Let(idx.name,
          If((i < -len) || (i >= len),
            Die(invoke("concat", TString,
              Str("string index out of bounds: "),
              invoke("concat", TString,
                invoke("str", TString, i),
                invoke("concat", TString, Str(" / "), invoke("str", TString, len)))), TInt32, errorID),
            If(i < 0, i + len, i)),
          invoke("substring", TString, s, idx, idx + 1)))
    }

    registerIR2("sliceRight", TString, TInt32, TString) { (_, s, start, _) => invoke("slice", TString, s, start, invoke("length", TInt32, s)) }
    registerIR2("sliceLeft", TString, TInt32, TString) { (_, s, end, _) => invoke("slice", TString, s, I32(0), end) }

    registerSCode1("str", tv("T"), TString, (_: Type, _: SType) => SJavaString) { case (r, cb, st: SJavaString.type, a, _) =>
      val annotation = svalueToJavaValue(cb, r.region, a)
      val str = cb.emb.getType(a.st.virtualType).invoke[Any, String]("str", annotation)
      st.construct(cb, str)
    }

    registerIEmitCode1("showStr", tv("T"), TString, {
      (_: Type, _: EmitType) => EmitType(SJavaString, true)
    }) { case (cb, r, st: SJavaString.type, _, a) =>
      val jObj = cb.newLocal("showstr_java_obj")(boxedTypeInfo(a.st.virtualType))
      a.toI(cb).consume(cb,
        cb.assignAny(jObj, Code._null(boxedTypeInfo(a.st.virtualType))),
        sc => cb.assignAny(jObj, svalueToJavaValue(cb, r, sc)))

      val str = cb.emb.getType(a.st.virtualType).invoke[Any, String]("showStr", jObj)

      IEmitCode.present(cb, st.construct(cb, str))
    }

    registerIEmitCode2("showStr", tv("T"), TInt32, TString, {
      (_: Type, _: EmitType, truncType: EmitType) => EmitType(SJavaString, truncType.required)
    }) { case (cb, r, st: SJavaString.type, _, a, trunc) =>
      val jObj = cb.newLocal("showstr_java_obj")(boxedTypeInfo(a.st.virtualType))
      trunc.toI(cb).map(cb) { trunc =>

        a.toI(cb).consume(cb,
          cb.assignAny(jObj, Code._null(boxedTypeInfo(a.st.virtualType))),
          sc => cb.assignAny(jObj, svalueToJavaValue(cb, r, sc)))

        val str = cb.emb.getType(a.st.virtualType).invoke[Any, Int, String]("showStr", jObj, trunc.asInt.value)
        st.construct(cb, str)
      }
    }

    registerIEmitCode1("json", tv("T"), TString, (_: Type, _: EmitType) => EmitType(SJavaString, true)) {
      case (cb, r, st: SJavaString.type, _, a) =>
        val ti = boxedTypeInfo(a.st.virtualType)
        val inputJavaValue = cb.newLocal("json_func_input_jv")(ti)
        a.toI(cb).consume(cb,
          cb.assignAny(inputJavaValue, Code._null(ti)),
          { sc =>
            val jv = svalueToJavaValue(cb, r, sc)
            cb.assignAny(inputJavaValue, jv)
          })
        val json = cb.emb.getType(a.st.virtualType).invoke[Any, JValue]("toJSON", inputJavaValue)
        val str = Code.invokeScalaObject1[JValue, String](JsonMethods.getClass, "compact", json)
        IEmitCode.present(cb, st.construct(cb, str))
    }


    registerWrappedScalaFunction1("reverse", TString, TString, (_: Type, _: SType) => SJavaString)(thisClass, "reverse")
    registerWrappedScalaFunction1("upper", TString, TString, (_: Type, _: SType) => SJavaString)(thisClass, "upper")
    registerWrappedScalaFunction1("lower", TString, TString, (_: Type, _: SType) => SJavaString)(thisClass, "lower")
    registerWrappedScalaFunction1("strip", TString, TString, (_: Type, _: SType) => SJavaString)(thisClass, "strip")
    registerWrappedScalaFunction2("contains", TString, TString, TBoolean, {
      case (_: Type, _: SType, _: SType) => SBoolean
    })(thisClass, "contains")
    registerWrappedScalaFunction2("translate", TString, TDict(TString, TString), TString, {
      case (_: Type, _: SType, _: SType) => SJavaString
    })(thisClass, "translate")
    registerWrappedScalaFunction2("startswith", TString, TString, TBoolean, {
      case (_: Type, _: SType, _: SType) => SBoolean
    })(thisClass, "startswith")
    registerWrappedScalaFunction2("endswith", TString, TString, TBoolean, {
      case (_: Type, _: SType, _: SType) => SBoolean
    })(thisClass, "endswith")
    registerWrappedScalaFunction2("regexMatch", TString, TString, TBoolean, {
      case (_: Type, _: SType, _: SType) => SBoolean
    })(thisClass, "regexMatch")
    registerWrappedScalaFunction2("regexFullMatch", TString, TString, TBoolean, {
      case (_: Type, _: SType, _: SType) => SBoolean
    })(thisClass, "regexFullMatch")
    registerWrappedScalaFunction2("concat", TString, TString, TString, {
      case (_: Type, _: SType, _: SType) => SJavaString
    })(thisClass, "concat")

    registerWrappedScalaFunction2("split", TString, TString, TArray(TString), {
      case (_: Type, _: SType, _: SType) =>
        SJavaArrayString(true)
    })(thisClass, "split")

    registerWrappedScalaFunction3("split", TString, TString, TInt32, TArray(TString), {
      case (_: Type, _: SType, _: SType, _: SType) =>
        SJavaArrayString(true)
    })(thisClass, "splitLimited")

    registerWrappedScalaFunction3("replace", TString, TString, TString, TString, {
      case (_: Type, _: SType, _: SType, _: SType) => SJavaString
    })(thisClass, "replace")

    registerWrappedScalaFunction2("mkString", TSet(TString), TString, TString, {
      case (_: Type, _: SType, _: SType) => SJavaString
    })(thisClass, "setMkString")

    registerSCode4("splitQuotedRegex", TString, TString, TArray(TString), TString, TArray(TString), {
      case (_: Type, _: SType, _: SType, _: SType, _: SType) => SJavaArrayString(false)
    }) { case (r, cb, st: SJavaArrayString, s, separator, missing, quote, errorID) =>
      val quoteStr = cb.newLocal[String]("quoteStr", quote.asString.loadString(cb))
      val quoteChar = cb.newLocal[Char]("quoteChar")
      cb.ifx(quoteStr.length().cne(1), cb._fatalWithError(errorID, "quote must be a single character"))
      cb.assign(quoteChar, quoteStr(0))

      val string = cb.newLocal[String]("string", s.asString.loadString(cb))
      val sep = cb.newLocal[String]("sep", separator.asString.loadString(cb))
      val mv = missing.asIndexable

      new SJavaArrayStringValue(st, generateSplitQuotedRegex(cb, string, Right(sep), Some(quoteChar), mv, errorID))
    }

    registerSCode4("splitQuotedChar", TString, TString, TArray(TString), TString, TArray(TString), {
      case (_: Type, _: SType, _: SType, _: SType, _: SType) => SJavaArrayString(false)
    }) { case (r, cb, st: SJavaArrayString, s, separator, missing, quote, errorID) =>
      val quoteStr = cb.newLocal[String]("quoteStr", quote.asString.loadString(cb))
      val quoteChar = cb.newLocal[Char]("quoteChar")
      cb.ifx(quoteStr.length().cne(1), cb._fatalWithError(errorID, "quote must be a single character"))
      cb.assign(quoteChar, quoteStr(0))

      val string = cb.newLocal[String]("string", s.asString.loadString(cb))
      val sep = cb.newLocal[String]("sep", separator.asString.loadString(cb))
      val sepChar = cb.newLocal[Char]("sepChar")
      cb.ifx(sep.length().cne(1), cb._fatalWithError(errorID, "splitQuotedChar expected a single character for separator"))
      cb.assign(sepChar, sep(0))
      val mv = missing.asIndexable

      new SJavaArrayStringValue(st, generateSplitQuotedRegex(cb, string, Left(sepChar), Some(quoteChar), mv, errorID))
    }

    registerSCode3("splitRegex", TString, TString, TArray(TString), TArray(TString), {
      case (_: Type, _: SType, _: SType, _: SType) => SJavaArrayString(false)
    }) { case (r, cb, st: SJavaArrayString, s, separator, missing, errorID) =>
      val string = cb.newLocal[String]("string", s.asString.loadString(cb))
      val sep = cb.newLocal[String]("sep", separator.asString.loadString(cb))
      val mv = missing.asIndexable
      new SJavaArrayStringValue(st, generateSplitQuotedRegex(cb, string, Right(sep), None, mv, errorID))
    }

    registerSCode3("splitChar", TString, TString, TArray(TString), TArray(TString), {
      case (_: Type, _: SType, _: SType, _: SType) => SJavaArrayString(false)
    }) { case (r, cb, st: SJavaArrayString, s, separator, missing, errorID) =>
      val string = cb.newLocal[String]("string", s.asString.loadString(cb))
      val sep = cb.newLocal[String]("sep", separator.asString.loadString(cb))
      val sepChar = cb.newLocal[Char]("sepChar")
      cb.ifx(sep.length().cne(1), cb._fatalWithError(errorID, "splitChar expected a single character for separator"))
      cb.assign(sepChar, sep(0))
      val mv = missing.asIndexable

      new SJavaArrayStringValue(st, generateSplitQuotedRegex(cb, string, Left(sepChar), None, mv, errorID))
    }

    registerWrappedScalaFunction2("mkString", TArray(TString), TString, TString, {
      case (_: Type, _: SType, _: SType) => SJavaString
    })(thisClass, "arrayMkString")

    registerIEmitCode2("firstMatchIn", TString, TString, TArray(TString), {
      case (_: Type, _: EmitType, _: EmitType) => EmitType(SJavaArrayString(true), false)
    }) { case (cb: EmitCodeBuilder, region: Value[Region], st: SJavaArrayString, _,
    s: EmitCode, r: EmitCode) =>
      s.toI(cb).flatMap(cb) { case sc: SStringValue =>
        r.toI(cb).flatMap(cb) { case rc: SStringValue =>
          val out = cb.newLocal[Array[String]]("out",
            Code.invokeScalaObject2[String, String, Array[String]](
              thisClass, "firstMatchIn", sc.loadString(cb), rc.loadString(cb)))
          IEmitCode(cb, out.isNull, st.construct(cb, out))
        }
      }
    }

    registerEmitCode2("hamming", TString, TString, TInt32, {
      case (_: Type, _: EmitType, _: EmitType) => EmitType(SInt32, false)
    }) { case (r: EmitRegion, rt, _, e1: EmitCode, e2: EmitCode) =>
      EmitCode.fromI(r.mb) { cb =>
        e1.toI(cb).flatMap(cb) { case sc1: SStringValue =>
          e2.toI(cb).flatMap(cb) { case sc2: SStringValue =>
            val n = cb.newLocal("hamming_n", 0)
            val i = cb.newLocal("hamming_i", 0)

            val v1 = cb.newLocal[String]("hamming_str_1", sc1.loadString(cb))
            val v2 = cb.newLocal[String]("hamming_str_2", sc2.loadString(cb))

            val l1 = cb.newLocal[Int]("hamming_len_1", v1.invoke[Int]("length"))
            val l2 = cb.newLocal[Int]("hamming_len_2", v2.invoke[Int]("length"))
            val m = l1.cne(l2)

            IEmitCode(cb, m, {
              cb.whileLoop(i < l1, {
                cb.ifx(v1.invoke[Int, Char]("charAt", i).toI.cne(v2.invoke[Int, Char]("charAt", i).toI),
                  cb.assign(n, n + 1))
                cb.assign(i, i + 1)
              })
              primitive(n)
            })
          }
        }
      }
    }

    registerWrappedScalaFunction1("escapeString", TString, TString, (_: Type, _: SType) => SJavaString)(thisClass, "escapeString")
    registerWrappedScalaFunction3("strftime", TString, TInt64, TString, TString, {
      case (_: Type, _: SType, _: SType, _: SType) => SJavaString
    })(thisClass, "strftime")
    registerWrappedScalaFunction3("strptime", TString, TString, TString, TInt64, {
      case (_: Type, _: SType, _: SType, _: SType) => SInt64
    })(thisClass, "strptime")

    registerSCode("parse_json", Array(TString), TTuple(tv("T")),
      (rType: Type, _: Seq[SType]) => SType.canonical(rType), typeParameters = Array(tv("T"))
    ) { case (er, cb, _, resultType, Array(s: SStringValue), _) =>

      val warnCtx = cb.emb.genFieldThisRef[mutable.HashSet[String]]("parse_json_context")
      cb.ifx(warnCtx.load().isNull, cb.assign(warnCtx, Code.newInstance[mutable.HashSet[String]]()))

      val row = Code.invokeScalaObject3[String, Type, mutable.HashSet[String], Row](JSONAnnotationImpex.getClass, "irImportAnnotation",
        s.loadString(cb), er.mb.ecb.getType(resultType.virtualType.asInstanceOf[TTuple].types(0)), warnCtx)

      unwrapReturn(cb, er.region, resultType, row)
    }
  }
}
