package is.hail.expr.ir.functions

import is.hail.annotations.Region
import is.hail.asm4s.{coerce => _, _}
import is.hail.backend.HailStateManager
import is.hail.expr.ir._
import is.hail.io.fs.FS
import is.hail.io.vcf.{LoadVCF, VCFHeaderInfo}
import is.hail.types.physical.stypes._
import is.hail.types.physical.stypes.concrete.SJavaString
import is.hail.types.physical.stypes.interfaces._
import is.hail.types.physical.stypes.primitives._
import is.hail.types.virtual._
import is.hail.utils._

import java.util.IllegalFormatConversionException
import scala.reflect.ClassTag

import org.apache.spark.sql.Row

object UtilFunctions extends RegistryFunctions {

  def parseBoolean(s: String, errID: Int): Boolean =
    try
      s.toBoolean
    catch {
      case _: IllegalArgumentException => fatal(
          s"cannot parse boolean from input string '${StringEscapeUtils.escapeString(s)}'",
          errID,
        )
    }

  def parseInt32(s: String, errID: Int): Int =
    try
      s.toInt
    catch {
      case _: IllegalArgumentException =>
        fatal(s"cannot parse int32 from input string '${StringEscapeUtils.escapeString(s)}'", errID)
    }

  def parseInt64(s: String, errID: Int): Long =
    try
      s.toLong
    catch {
      case _: IllegalArgumentException =>
        fatal(s"cannot parse int64 from input string '${StringEscapeUtils.escapeString(s)}'", errID)
    }

  def parseSpecialNum32(s: String, errID: Int): Float = {
    s.length match {
      case 3 =>
        if (s.equalsCaseInsensitive("nan")) return Float.NaN
        if (s.equalsCaseInsensitive("inf")) return Float.PositiveInfinity
      case 4 =>
        if (s.equalsCaseInsensitive("+nan") || s.equalsCaseInsensitive("-nan")) return Float.NaN
        if (s.equalsCaseInsensitive("+inf")) return Float.PositiveInfinity
        if (s.equalsCaseInsensitive("-inf")) return Float.NegativeInfinity
      case 8 =>
        if (s.equalsCaseInsensitive("infinity")) return Float.PositiveInfinity
      case 9 =>
        if (s.equalsCaseInsensitive("+infinity")) return Float.PositiveInfinity
        if (s.equalsCaseInsensitive("-infinity")) return Float.NegativeInfinity
      case _ =>
    }
    fatal(s"cannot parse float32 from input string '${StringEscapeUtils.escapeString(s)}'", errID)
  }

  def parseSpecialNum64(s: String, errID: Int): Double = {
    s.length match {
      case 3 =>
        if (s.equalsCaseInsensitive("nan")) return Double.NaN
        if (s.equalsCaseInsensitive("inf")) return Double.PositiveInfinity
      case 4 =>
        if (s.equalsCaseInsensitive("+nan") || s.equalsCaseInsensitive("-nan")) return Double.NaN
        if (s.equalsCaseInsensitive("+inf")) return Double.PositiveInfinity
        if (s.equalsCaseInsensitive("-inf")) return Double.NegativeInfinity
      case 8 =>
        if (s.equalsCaseInsensitive("infinity")) return Double.PositiveInfinity
      case 9 =>
        if (s.equalsCaseInsensitive("+infinity")) return Double.PositiveInfinity
        if (s.equalsCaseInsensitive("-infinity")) return Double.NegativeInfinity
      case _ =>
    }
    fatal(s"cannot parse float64 from input string '${StringEscapeUtils.escapeString(s)}'", errID)
  }

  def parseFloat32(s: String, errID: Int): Float = {
    try
      s.toFloat
    catch {
      case _: NumberFormatException =>
        parseSpecialNum32(s, errID)
    }
  }

  def parseFloat64(s: String, errID: Int): Double = {
    try
      s.toDouble
    catch {
      case _: NumberFormatException =>
        parseSpecialNum64(s, errID)
    }
  }

  def isValidBoolean(s: String): Boolean =
    s.equalsCaseInsensitive("true") || s.equalsCaseInsensitive("false")

  def isValidInt32(s: String): Boolean =
    try {
      s.toInt; true
    } catch {
      case _: NumberFormatException => false
    }

  def isValidInt64(s: String): Boolean =
    try {
      s.toLong; true
    } catch {
      case _: NumberFormatException => false
    }

  def isValidFloat32(s: String): Boolean =
    try {
      parseFloat32(s, -1)
      true
    } catch {
      case _: NumberFormatException => false
      case _: HailException => false
    }

  def isValidFloat64(s: String): Boolean =
    try {
      parseFloat64(s, -1)
      true
    } catch {
      case _: NumberFormatException => false
      case _: HailException => false
    }

  def min_ignore_missing(l: Int, lMissing: Boolean, r: Int, rMissing: Boolean): Int =
    if (lMissing) r else if (rMissing) l else Math.min(l, r)

  def min_ignore_missing(l: Long, lMissing: Boolean, r: Long, rMissing: Boolean): Long =
    if (lMissing) r else if (rMissing) l else Math.min(l, r)

  def min_ignore_missing(l: Float, lMissing: Boolean, r: Float, rMissing: Boolean): Float =
    if (lMissing) r else if (rMissing) l else Math.min(l, r)

  def min_ignore_missing(l: Double, lMissing: Boolean, r: Double, rMissing: Boolean): Double =
    if (lMissing) r else if (rMissing) l else Math.min(l, r)

  def max_ignore_missing(l: Int, lMissing: Boolean, r: Int, rMissing: Boolean): Int =
    if (lMissing) r else if (rMissing) l else Math.max(l, r)

  def max_ignore_missing(l: Long, lMissing: Boolean, r: Long, rMissing: Boolean): Long =
    if (lMissing) r else if (rMissing) l else Math.max(l, r)

  def max_ignore_missing(l: Float, lMissing: Boolean, r: Float, rMissing: Boolean): Float =
    if (lMissing) r else if (rMissing) l else Math.max(l, r)

  def max_ignore_missing(l: Double, lMissing: Boolean, r: Double, rMissing: Boolean): Double =
    if (lMissing) r else if (rMissing) l else Math.max(l, r)

  def nanmax(l: Double, r: Double): Double =
    if (java.lang.Double.isNaN(l)) r else if (java.lang.Double.isNaN(r)) l else Math.max(l, r)

  def nanmax(l: Float, r: Float): Float =
    if (java.lang.Float.isNaN(l)) r else if (java.lang.Float.isNaN(r)) l else Math.max(l, r)

  def nanmin(l: Double, r: Double): Double =
    if (java.lang.Double.isNaN(l)) r else if (java.lang.Double.isNaN(r)) l else Math.min(l, r)

  def nanmin(l: Float, r: Float): Float =
    if (java.lang.Float.isNaN(l)) r else if (java.lang.Float.isNaN(r)) l else Math.min(l, r)

  def nanmin_ignore_missing(l: Float, lMissing: Boolean, r: Float, rMissing: Boolean): Float =
    if (lMissing) r else if (rMissing) l else nanmin(l, r)

  def nanmin_ignore_missing(l: Double, lMissing: Boolean, r: Double, rMissing: Boolean): Double =
    if (lMissing) r else if (rMissing) l else nanmin(l, r)

  def nanmax_ignore_missing(l: Float, lMissing: Boolean, r: Float, rMissing: Boolean): Float =
    if (lMissing) r else if (rMissing) l else nanmax(l, r)

  def nanmax_ignore_missing(l: Double, lMissing: Boolean, r: Double, rMissing: Boolean): Double =
    if (lMissing) r else if (rMissing) l else nanmax(l, r)

  def intMin(a: IR, b: IR): IR = If(ApplyComparisonOp(LT(a.typ), a, b), a, b)

  def intMax(a: IR, b: IR): IR = If(ApplyComparisonOp(GT(a.typ), a, b), a, b)

  def format(f: String, args: Row): String =
    try
      String.format(f, args.toSeq.map(_.asInstanceOf[java.lang.Object]): _*)
    catch {
      case e: IllegalFormatConversionException =>
        fatal(
          s"Encountered invalid type for format string $f: format specifier ${e.getConversion} does not accept type ${e.getArgumentClass.getCanonicalName}"
        )
    }

  def registerAll() {
    val thisClass = getClass

    registerSCode4(
      "valuesSimilar",
      tv("T"),
      tv("U"),
      TFloat64,
      TBoolean,
      TBoolean,
      { case (_: Type, _: SType, _: SType, _: SType, _: SType) => SBoolean },
    ) { case (er, cb, _, l, r, tol, abs, _) =>
      assert(
        l.st.virtualType == r.st.virtualType,
        s"\n  lt=${l.st.virtualType}\n  rt=${r.st.virtualType}",
      )
      val lb = svalueToJavaValue(cb, er.region, l)
      val rb = svalueToJavaValue(cb, er.region, r)
      primitive(
        cb.memoize(
          er.mb
            .getType(l.st.virtualType).invoke[Any, Any, Double, Boolean, Boolean](
              "valuesSimilar",
              lb,
              rb,
              tol.asDouble.value,
              abs.asBoolean.value,
            )
        )
      )
    }

    registerCode1("triangle", TInt32, TInt32, (_: Type, _: SType) => SInt32) {
      case (cb, _, _, nn) =>
        val n = nn.asInt.value
        cb.memoize((n * (n + 1)) / 2)
    }

    registerSCode1("toInt32", TBoolean, TInt32, (_: Type, _: SType) => SInt32) {
      case (_, cb, _, x, _) =>
        primitive(cb.memoize(x.asBoolean.value.toI))
    }
    registerSCode1("toInt64", TBoolean, TInt64, (_: Type, _: SType) => SInt64) {
      case (_, cb, _, x, _) =>
        primitive(cb.memoize(x.asBoolean.value.toI.toL))
    }
    registerSCode1("toFloat32", TBoolean, TFloat32, (_: Type, _: SType) => SFloat32) {
      case (_, cb, _, x, _) =>
        primitive(cb.memoize(x.asBoolean.value.toI.toF))
    }
    registerSCode1("toFloat64", TBoolean, TFloat64, (_: Type, _: SType) => SFloat64) {
      case (_, cb, _, x, _) =>
        primitive(cb.memoize(x.asBoolean.value.toI.toD))
    }

    for (
      (name, t, rpt, ct) <- Seq[(String, Type, SType, ClassTag[_])](
        ("Boolean", TBoolean, SBoolean, implicitly[ClassTag[Boolean]]),
        ("Int32", TInt32, SInt32, implicitly[ClassTag[Int]]),
        ("Int64", TInt64, SInt64, implicitly[ClassTag[Long]]),
        ("Float64", TFloat64, SFloat64, implicitly[ClassTag[Double]]),
        ("Float32", TFloat32, SFloat32, implicitly[ClassTag[Float]]),
      )
    ) {
      val ctString: ClassTag[String] = implicitly[ClassTag[String]]
      registerSCode1(s"to$name", TString, t, (_: Type, _: SType) => rpt) {
        case (_, cb, rt, x: SStringValue, err) =>
          val s = x.loadString(cb)
          primitive(
            rt.virtualType,
            cb.memoizeAny(
              Code.invokeScalaObject2(thisClass, s"parse$name", s, err)(
                ctString,
                implicitly[ClassTag[Int]],
                ct,
              ),
              typeInfoFromClassTag(ct),
            ),
          )
      }
      registerIEmitCode1(
        s"to${name}OrMissing",
        TString,
        t,
        (_: Type, _: EmitType) => EmitType(rpt, false),
      ) { case (cb, _, rt, err, x) =>
        x.toI(cb).flatMap(cb) { case sc: SStringValue =>
          val sv = cb.newLocal[String]("s", sc.loadString(cb))
          IEmitCode(
            cb,
            !Code.invokeScalaObject1[String, Boolean](thisClass, s"isValid$name", sv),
            primitive(
              rt.virtualType,
              cb.memoizeAny(
                Code.invokeScalaObject2(thisClass, s"parse$name", sv, err)(
                  ctString,
                  implicitly[ClassTag[Int]],
                  ct,
                ),
                typeInfoFromClassTag(ct),
              ),
            ),
          )
        }
      }
    }

    Array(TInt32, TInt64).foreach { t =>
      registerIR2("min", t, t, t)((_, a, b, _) => intMin(a, b))
      registerIR2("max", t, t, t)((_, a, b, _) => intMax(a, b))
    }

    Array("min", "max").foreach { name =>
      registerCode2(name, TFloat32, TFloat32, TFloat32, (_: Type, _: SType, _: SType) => SFloat32) {
        case (cb, _, _, v1, v2) =>
          cb.memoize(
            Code.invokeStatic2[Math, Float, Float, Float](name, v1.asFloat.value, v2.asFloat.value)
          )
      }

      registerCode2(name, TFloat64, TFloat64, TFloat64, (_: Type, _: SType, _: SType) => SFloat64) {
        case (cb, _, _, v1, v2) =>
          cb.memoize(
            Code.invokeStatic2[Math, Double, Double, Double](
              name,
              v1.asDouble.value,
              v2.asDouble.value,
            )
          )
      }

      val ignoreMissingName = name + "_ignore_missing"
      val ignoreNanName = "nan" + name
      val ignoreBothName = ignoreNanName + "_ignore_missing"

      registerCode2(
        ignoreNanName,
        TFloat32,
        TFloat32,
        TFloat32,
        (_: Type, _: SType, _: SType) => SFloat32,
      ) { case (cb, _, _, v1, v2) =>
        cb.memoize(
          Code.invokeScalaObject2[Float, Float, Float](
            thisClass,
            ignoreNanName,
            v1.asFloat.value,
            v2.asFloat.value,
          )
        )
      }

      registerCode2(
        ignoreNanName,
        TFloat64,
        TFloat64,
        TFloat64,
        (_: Type, _: SType, _: SType) => SFloat64,
      ) { case (cb, _, _, v1, v2) =>
        cb.memoize(
          Code.invokeScalaObject2[Double, Double, Double](
            thisClass,
            ignoreNanName,
            v1.asDouble.value,
            v2.asDouble.value,
          )
        )
      }

      def ignoreMissingTriplet[T: ClassTag](
        cb: EmitCodeBuilder,
        rt: SType,
        v1: EmitCode,
        v2: EmitCode,
        name: String,
        f: (Code[T], Code[T]) => Code[T],
      )(implicit ti: TypeInfo[T]
      ): IEmitCode = {
        val value = cb.newLocal[T](s"ignore_missing_${name}_value")
        val v1Value = v1.toI(cb).memoize(cb, "ignore_missing_v1")
        val v2Value = v2.toI(cb).memoize(cb, "ignore_missing_v2")

        val Lmissing = CodeLabel()
        val Ldefined = CodeLabel()
        v1Value.toI(cb)
          .consume(
            cb,
            v2Value.toI(cb).consume(
              cb,
              cb.goto(Lmissing),
              sc2 => cb.assignAny(value, sc2.asPrimitive.primitiveValue[T]),
            ),
            { sc1 =>
              cb.assign(value, sc1.asPrimitive.primitiveValue[T])
              v2Value.toI(cb).consume(
                cb,
                {},
                sc2 => cb.assignAny(value, f(value, sc2.asPrimitive.primitiveValue[T])),
              )
            },
          )
        cb.goto(Ldefined)

        IEmitCode(Lmissing, Ldefined, primitive(rt.virtualType, value), v1.required || v2.required)
      }

      registerIEmitCode2(
        ignoreMissingName,
        TInt32,
        TInt32,
        TInt32,
        (_: Type, t1: EmitType, t2: EmitType) => EmitType(SInt32, t1.required || t2.required),
      ) { case (cb, _, rt, _, v1, v2) =>
        ignoreMissingTriplet[Int](
          cb,
          rt,
          v1,
          v2,
          name,
          Code.invokeStatic2[Math, Int, Int, Int](name, _, _),
        )
      }

      registerIEmitCode2(
        ignoreMissingName,
        TInt64,
        TInt64,
        TInt64,
        (_: Type, t1: EmitType, t2: EmitType) => EmitType(SInt64, t1.required || t2.required),
      ) { case (cb, _, rt, _, v1, v2) =>
        ignoreMissingTriplet[Long](
          cb,
          rt,
          v1,
          v2,
          name,
          Code.invokeStatic2[Math, Long, Long, Long](name, _, _),
        )
      }

      registerIEmitCode2(
        ignoreMissingName,
        TFloat32,
        TFloat32,
        TFloat32,
        (_: Type, t1: EmitType, t2: EmitType) => EmitType(SFloat32, t1.required || t2.required),
      ) { case (cb, _, rt, _, v1, v2) =>
        ignoreMissingTriplet[Float](
          cb,
          rt,
          v1,
          v2,
          name,
          Code.invokeStatic2[Math, Float, Float, Float](name, _, _),
        )
      }

      registerIEmitCode2(
        ignoreMissingName,
        TFloat64,
        TFloat64,
        TFloat64,
        (_: Type, t1: EmitType, t2: EmitType) => EmitType(SFloat64, t1.required || t2.required),
      ) { case (cb, _, rt, _, v1, v2) =>
        ignoreMissingTriplet[Double](
          cb,
          rt,
          v1,
          v2,
          name,
          Code.invokeStatic2[Math, Double, Double, Double](name, _, _),
        )
      }

      registerIEmitCode2(
        ignoreBothName,
        TFloat32,
        TFloat32,
        TFloat32,
        (_: Type, t1: EmitType, t2: EmitType) => EmitType(SFloat32, t1.required || t2.required),
      ) { case (cb, _, rt, _, v1, v2) =>
        ignoreMissingTriplet[Float](
          cb,
          rt,
          v1,
          v2,
          ignoreNanName,
          Code.invokeScalaObject2[Float, Float, Float](thisClass, ignoreNanName, _, _),
        )
      }

      registerIEmitCode2(
        ignoreBothName,
        TFloat64,
        TFloat64,
        TFloat64,
        (_: Type, t1: EmitType, t2: EmitType) => EmitType(SFloat64, t1.required || t2.required),
      ) { case (cb, _, rt, _, v1, v2) =>
        ignoreMissingTriplet[Double](
          cb,
          rt,
          v1,
          v2,
          ignoreNanName,
          Code.invokeScalaObject2[Double, Double, Double](thisClass, ignoreNanName, _, _),
        )
      }
    }

    registerSCode2(
      "format",
      TString,
      tv("T", "tuple"),
      TString,
      (_: Type, _: SType, _: SType) => SJavaString,
    ) {
      case (r, cb, st: SJavaString.type, format, args, _) =>
        val javaObjArgs = Code.checkcast[Row](svalueToJavaValue(cb, r.region, args))
        val formatted = Code.invokeScalaObject2[String, Row, String](
          thisClass,
          "format",
          format.asString.loadString(cb),
          javaObjArgs,
        )
        st.construct(cb, formatted)
    }

    registerIEmitCode2(
      "land",
      TBoolean,
      TBoolean,
      TBoolean,
      (_: Type, tl: EmitType, tr: EmitType) => EmitType(SBoolean, tl.required && tr.required),
    ) { case (cb, _, _, _, l, r) =>
      if (l.required && r.required) {
        val result = cb.newLocal[Boolean]("land_result")
        cb.if_(
          l.toI(cb).get(cb).asBoolean.value,
          cb.assign(result, r.toI(cb).get(cb).asBoolean.value),
          cb.assign(result, const(false)),
        )

        IEmitCode.present(cb, primitive(result))
      } else {
        // 00 ... 00 rv rm lv lm
        val w = cb.newLocal[Int]("land_w")

        // m/m, t/m, m/t
        val M = const((1 << 5) | (1 << 6) | (1 << 9))

        l.toI(cb)
          .consume(
            cb,
            cb.assign(w, 1),
            b1 => cb.assign(w, b1.asBoolean.value.mux(const(2), const(0))),
          )

        cb.if_(
          w.cne(0),
          r.toI(cb).consume(
            cb,
            cb.assign(w, w | const(4)),
            b2 => cb.assign(w, w | b2.asBoolean.value.mux(const(8), const(0))),
          ),
        )

        IEmitCode(cb, ((M >> w) & 1).cne(0), primitive(cb.memoize(w.ceq(10))))
      }
    }

    registerIEmitCode2(
      "lor",
      TBoolean,
      TBoolean,
      TBoolean,
      (_: Type, tl: EmitType, tr: EmitType) => EmitType(SBoolean, tl.required && tr.required),
    ) { case (cb, _, _, _, l, r) =>
      if (l.required && r.required) {
        val result = cb.newLocal[Boolean]("land_result")
        cb.if_(
          l.toI(cb).get(cb).asBoolean.value,
          cb.assign(result, const(true)),
          cb.assign(result, r.toI(cb).get(cb).asBoolean.value),
        )

        IEmitCode.present(cb, primitive(result))
      } else {
        // 00 ... 00 rv rm lv lm
        val w = cb.newLocal[Int]("lor_w")

        // m/m, f/m, m/f
        val M = const((1 << 5) | (1 << 1) | (1 << 4))

        l.toI(cb)
          .consume(
            cb,
            cb.assign(w, 1),
            b1 => cb.assign(w, b1.asBoolean.value.mux(const(2), const(0))),
          )

        cb.if_(
          w.cne(2),
          r.toI(cb).consume(
            cb,
            cb.assign(w, w | const(4)),
            b2 => cb.assign(w, w | b2.asBoolean.value.mux(const(8), const(0))),
          ),
        )

        IEmitCode(cb, ((M >> w) & 1).cne(0), primitive(cb.memoize(w.cne(0))))
      }
    }

    registerIEmitCode4(
      "getVCFHeader",
      TString,
      TString,
      TString,
      TString,
      VCFHeaderInfo.headerType,
      (_, fileET, _, _, _) => EmitType(VCFHeaderInfo.headerTypePType.sType, fileET.required),
    ) { case (cb, r, _, _, file, filter, find, replace) =>
      file.toI(cb).map(cb) { case filePath: SStringValue =>
        val filterVar = cb.newLocal[String]("filterVar")
        val findVar = cb.newLocal[String]("findVar")
        val replaceVar = cb.newLocal[String]("replaceVar")
        filter
          .toI(cb).consume(
            cb,
            cb.assign(filterVar, Code._null),
            filt => cb.assign(filterVar, filt.asString.loadString(cb)),
          )
        find.toI(cb).consume(
          cb,
          cb.assign(findVar, Code._null),
          find => cb.assign(findVar, find.asString.loadString(cb)),
        )
        replace.toI(cb).consume(
          cb,
          cb.assign(replaceVar, Code._null),
          replace => cb.assign(replaceVar, replace.asString.loadString(cb)),
        )
        val hd = Code.invokeScalaObject5[FS, String, String, String, String, VCFHeaderInfo](
          LoadVCF.getClass,
          "getVCFHeaderInfo",
          cb.emb.getFS,
          filePath.loadString(cb),
          filterVar,
          findVar,
          replaceVar,
        )
        val addr = cb.memoize(hd.invoke[HailStateManager, Region, Boolean, Long](
          "writeToRegion",
          cb.emb.getObject(cb.emb.ecb.ctx.stateManager),
          r,
          const(false),
        ))
        VCFHeaderInfo.headerTypePType.loadCheapSCode(cb, addr)
      }
    }
  }
}
