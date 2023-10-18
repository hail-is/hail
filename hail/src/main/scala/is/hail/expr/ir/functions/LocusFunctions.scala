package is.hail.expr.ir.functions

import is.hail.annotations.Region
import is.hail.asm4s._
import is.hail.expr.ir.{EmitMethodBuilder, _}
import is.hail.types.physical._
import is.hail.types.physical.stypes.concrete._
import is.hail.types.physical.stypes.interfaces._
import is.hail.types.physical.stypes.primitives._
import is.hail.types.physical.stypes.{EmitType, SType}
import is.hail.types.virtual._
import is.hail.utils._
import is.hail.variant._

object LocusFunctions extends RegistryFunctions {

  def rgCode(mb: EmitMethodBuilder[_], rg: String): Code[ReferenceGenome] =
    mb.getReferenceGenome(rg)

  def tlocus(name: String): Type = tv(name, "locus")

  def tvariant(name: String): TStruct = TStruct("locus" -> tlocus(name), "alleles" -> TArray(TString))

  def tinterval(name: String): TInterval = TInterval(tlocus(name))

  def emitLocus(cb: EmitCodeBuilder, r: Value[Region], locus: Code[Locus], rt: PCanonicalLocus): SCanonicalLocusPointerValue = {
    val loc = cb.newLocal[Locus]("emit_locus_memo", locus)
    rt.constructFromContigAndPosition(cb, r,
      loc.invoke[String]("contig"),
      loc.invoke[Int]("position"))
  }

  def emitVariant(cb: EmitCodeBuilder, r: Value[Region], variantCode: Code[(Locus, IndexedSeq[String])], rt: PCanonicalStruct): SBaseStructPointerValue = {

    val variant = cb.newLocal[(Locus, IndexedSeq[String])]("emit_variant_variant", variantCode)

    val locus = EmitCode.fromI(cb.emb) { cb => IEmitCode.present(cb, emitLocus(cb, r, variant.invoke[Locus]("_1"), rt.types(0).asInstanceOf[PCanonicalLocus])) }
    val alleles = EmitCode.fromI(cb.emb) { cb =>
      val pAlleles = rt.types(1).asInstanceOf[PCanonicalArray]
      val all = cb.newLocal[IndexedSeq[String]]("locus_alleles_parsed_alleles", variant.invoke[IndexedSeq[String]]("_2"))
      val len = cb.newLocal[Int]("locus_alleles_n_alleles", all.invoke[Int]("length"))
      val ps = pAlleles.elementType.setRequired(false).asInstanceOf[PCanonicalString]
      val ss = SStringPointer(ps)
      val (push, finish) = pAlleles.constructFromFunctions(cb, r, len, deepCopy = false)
      val i = cb.newLocal[Int]("locus_alleles_i", 0)
      cb.while_(i < len, {
        push(cb, IEmitCode.present(cb, ss.constructFromString(cb, r, all.invoke[Int, String]("apply", i))))
        cb.assign(i, i + 1)
      })
      IEmitCode.present(cb, finish(cb))
    }
    rt.constructFromFields(cb, r, FastSeq(locus, alleles), deepCopy = false)
  }

  def emitLocusInterval(cb: EmitCodeBuilder, r: Value[Region], intervalCode: Code[Interval], pt: PCanonicalInterval): SIntervalPointerValue = {
    val interval = cb.newLocal[Interval]("emit_interval_interval", intervalCode)
    val pointType = pt.pointType.asInstanceOf[PCanonicalLocus]
    pt.constructFromCodes(cb,
      r,
      EmitCode.fromI(cb.emb)(cb => IEmitCode.present(cb, emitLocus(cb, r, interval.invoke[Locus]("start"), pointType))),
      EmitCode.fromI(cb.emb)(cb => IEmitCode.present(cb, emitLocus(cb, r, interval.invoke[Locus]("end"), pointType))),
      cb.memoize(interval.invoke[Boolean]("includesStart")),
      cb.memoize(interval.invoke[Boolean]("includesEnd"))
    )
  }

  def registerLocusCodePredicate(methodName: String): Unit =
    registerSCode1(methodName, tlocus("T"), TBoolean,
      (_: Type, x: SType) => SBoolean) {
      case (r, cb, rt, locus: SLocusValue, _) =>
        primitive(cb.memoize(cb.emb.getReferenceGenome(locus.st.rg).invoke[Locus, Boolean](methodName, locus.getLocusObj(cb))))
    }

  def registerLocusCodeContigPredicate(methodName: String): Unit =
    registerSCode1(methodName, tlocus("T"), TBoolean,
      (_: Type, x: SType) => SBoolean) {
      case (r, cb, rt, locus: SLocusValue, _) =>
        primitive(cb.memoize(cb.emb.getReferenceGenome(locus.st.rg).invoke[String, Boolean](methodName, locus.contig(cb).loadString(cb))))
    }

  def registerAll() {
    val locusClass = Locus.getClass

    registerSCode1("contig", tlocus("T"), TString,
      (_: Type, x: SType) => x.asInstanceOf[SLocus].contigType) {
      case (r, cb, rt, locus: SLocusValue, _) =>
        locus.contig(cb)
    }

    registerSCode1("contig_idx", tlocus("T"), TInt32,
      (_: Type, x: SType) => SInt32) {
      case (r, cb, rt, locus: SLocusValue, _) =>
        primitive(cb.memoize(cb.emb.getReferenceGenome(locus.st.rg).invoke[String, Int]("getContigIndex", locus.contig(cb).loadString(cb))))
    }


    registerSCode1("position", tlocus("T"), TInt32, (_: Type, x: SType) => SInt32) {
      case (r, cb, rt, locus: SLocusValue, _) =>
        primitive(locus.position(cb))
    }
    registerLocusCodePredicate("isAutosomalOrPseudoAutosomal")
    registerLocusCodeContigPredicate("isAutosomal")
    registerLocusCodeContigPredicate("isMitochondrial")
    registerLocusCodePredicate("inXPar")
    registerLocusCodePredicate("inYPar")
    registerLocusCodePredicate("inXNonPar")
    registerLocusCodePredicate("inYNonPar")

    registerSCode2("add_on_contig", tlocus("T"), TInt32, tlocus("T"), (tl: Type, _:SType, _: SType) => SCanonicalLocusPointer(PCanonicalLocus(tl.asInstanceOf[TLocus].rg))) {
      case (r: EmitRegion, cb: EmitCodeBuilder, rt: SCanonicalLocusPointer, inputLocus: SLocusValue, basePairsToAdd: SInt32Value, errorID) =>

        val contig = inputLocus.contig(cb).loadString(cb)
        val basePos = inputLocus.position(cb)
        val bps = basePairsToAdd.value
        val newPos = cb.newLocal[Int]("newPos")
        cb.if_(bps <= 0,
          cb.assign(newPos, (basePos + bps).max(1)),
          cb.assign(newPos, (basePos + bps).min(cb.emb.getReferenceGenome(rt.rg).invoke[String, Int]("contigLength", contig)))
        )
        rt.pType.constructFromContigAndPosition(cb, r.region, contig, newPos)
    }

    registerSCode2("min_rep", tlocus("T"), TArray(TString), TStruct("locus" -> tv("T"), "alleles" -> TArray(TString)), {
      (returnType: Type, _: SType, _: SType) => {
        val locusPT = PCanonicalLocus(returnType.asInstanceOf[TStruct].field("locus").typ.asInstanceOf[TLocus].rg, true)
        PCanonicalStruct("locus" -> locusPT, "alleles" -> PCanonicalArray(PCanonicalString(true), true)).sType
      }
    }) {
      case (r, cb, SBaseStructPointer(rt: PCanonicalStruct), locus: SLocusValue, alleles: SIndexableValue, _) =>
        val variantTuple = Code.invokeScalaObject2[Locus, IndexedSeq[String], (Locus, IndexedSeq[String])](
          VariantMethods.getClass, "minRep",
          locus.getLocusObj(cb),
          Code.checkcast[IndexedSeq[String]](svalueToJavaValue(cb, r.region, alleles)))

        emitVariant(cb, r.region, variantTuple, rt)
    }

    registerSCode2("locus_windows_per_contig", TArray(TArray(TFloat64)), TFloat64, TTuple(TArray(TInt32), TArray(TInt32)), {
      (_: Type, _: SType, _: SType) =>
        PCanonicalTuple(false, PCanonicalArray(PInt32(true), true), PCanonicalArray(PInt32(true), true)).sType
    }) {
      case (r: EmitRegion, cb: EmitCodeBuilder, SBaseStructPointer(rt: PCanonicalTuple), grouped: SIndexableValue, radiusVal: SFloat64Value, errorID) =>
        val radius = radiusVal.value
        val ncontigs = grouped.loadLength()
        val totalLen = cb.newLocal[Int]("locuswindows_totallen", 0)

        def forAllContigs(cb: EmitCodeBuilder)(f: (EmitCodeBuilder, Value[Int], SIndexableValue) => Unit): Unit = {
          val iContig = cb.newLocal[Int]("locuswindows_icontig", 0)
          cb.while_(iContig < ncontigs, {
            val coordPerContig = grouped.loadElement(cb, iContig).get(cb, "locus_windows group cannot be missing")
              .asIndexable
            f(cb, iContig, coordPerContig)
            cb.assign(iContig, iContig + 1)
          })
        }

        val arrayType = PCanonicalArray(PInt32(true), true)
        assert(rt.types(0) == arrayType)
        assert(rt.types(1) == arrayType)

        def addIdxWithCondition(cb: EmitCodeBuilder)(cond: (EmitCodeBuilder, Value[Int], Value[Int], SIndexableValue) => Code[Boolean]): IEmitCode = {

          val (pushElement, finish) = arrayType.constructFromFunctions(cb, r.region, totalLen, deepCopy = false)
          val offset = cb.newLocal[Int]("locuswindows_offset", 0)

          val lastCoord = cb.newLocal[Double]("locuswindows_coord")

          forAllContigs(cb) { case (cb, contigIdx, coords) =>
            val i = cb.newLocal[Int]("locuswindows_i", 0)
            val idx = cb.newLocal[Int]("locuswindows_idx", 0)
            val len = coords.loadLength()
            cb.if_(len.ceq(0),
              cb.assign(lastCoord, 0.0),
              cb.assign(lastCoord, coords.loadElement(cb, 0).get(cb, "locus_windows: missing value for 'coord_expr'").asDouble.value))
            cb.while_(i < len, {

              coords.loadElement(cb, i).consume(cb,
                cb._fatalWithError(errorID, const("locus_windows: missing value for 'coord_expr' at row ")
                    .concat((offset + i).toS)),
                { sc =>
                  val currentCoord = cb.newLocal[Double]("locuswindows_coord_i", sc.asDouble.value)
                  cb.if_(lastCoord > currentCoord,
                    cb._fatalWithError(errorID, "locus_windows: 'coord_expr' must be in ascending order within each contig."),
                    cb.assign(lastCoord, currentCoord)
                  )
                })

              val Lstart = CodeLabel()
              val Lbreak = CodeLabel()

              cb.define(Lstart)
              cb.if_(idx >= len,
                cb.goto(Lbreak)
              )
              cb.if_(cond(cb, i, idx, coords),
                {
                  cb.assign(idx, idx + 1)
                  cb.goto(Lstart)
                },
                cb.goto(Lbreak)
              )
              cb.define(Lbreak)

              pushElement(cb, IEmitCode.present(cb, primitive(cb.memoize(offset + idx))))

              cb.assign(i, i + 1)
            })
            cb.assign(offset, offset + len)
          }
          IEmitCode.present(cb, finish(cb))
        }

        forAllContigs(cb) { case (cb, _, coordsPerContig) =>
          cb.assign(totalLen, totalLen + coordsPerContig.loadLength())
        }

        rt.constructFromFields(cb, r.region,
          FastSeq[EmitCode](
            EmitCode.fromI(cb.emb)(cb => addIdxWithCondition(cb) { case (cb, i, idx, coords) => coords.loadElement(cb, i)
                          .get(cb, "locus_windows: missing value for 'coord_expr'")
                          .asDouble.value > (coords.loadElement(cb, idx)
                                        .get(cb, "locus_windows: missing value for 'coord_expr'").asDouble.value + radius)
            }),
            EmitCode.fromI(cb.emb)(cb => addIdxWithCondition(cb) { case (cb, i, idx, coords) => coords.loadElement(cb, i)
                          .get(cb, "locus_windows: missing value for 'coord_expr'")
                          .asDouble.value >= (coords.loadElement(cb, idx)
                                        .get(cb, "locus_windows: missing value for 'coord_expr'").asDouble.value - radius)
            })
          ), deepCopy = false)
    }

    registerSCode1("Locus", TString, tlocus("T"), {
      (returnType: Type, _: SType) => PCanonicalLocus(returnType.asInstanceOf[TLocus].rg).sType
    }) {
      case (r, cb, SCanonicalLocusPointer(rt: PCanonicalLocus), str: SStringValue, _) =>
        val slocus = str.loadString(cb)
        emitLocus(cb,
          r.region,
          Code.invokeScalaObject2[String, ReferenceGenome, Locus](locusClass, "parse", slocus, rgCode(r.mb, rt.rg)),
          rt)
    }

    registerSCode2("Locus", TString, TInt32, tlocus("T"), {
      (returnType: Type, _: SType, _: SType) => PCanonicalLocus(returnType.asInstanceOf[TLocus].rg).sType
    }) {
      case (r, cb, SCanonicalLocusPointer(rt: PCanonicalLocus), contig, pos, _) =>
        cb += rgCode(r.mb, rt.rg).invoke[String, Int, Unit]("checkLocus", contig.asString.loadString(cb), pos.asInt.value)
        rt.constructFromContigAndPosition(cb, r.region, contig.asString.loadString(cb), pos.asInt.value)
    }

    registerSCode1("LocusAlleles", TString, tvariant("T"), {
      (returnType: Type, _: SType) => {
        val lTyp = returnType.asInstanceOf[TStruct].field("locus").typ.asInstanceOf[TLocus]
        PCanonicalStruct("locus" -> PCanonicalLocus(lTyp.rg, true), "alleles" -> PCanonicalArray(PCanonicalString(true), true)).sType
      }
    }) {
      case (r, cb, SBaseStructPointer(rt: PCanonicalStruct), variantStr: SStringValue, _) =>

        val svar = variantStr.loadString(cb)
        val plocus = rt.types(0).asInstanceOf[PCanonicalLocus]
        val variant = Code
          .invokeScalaObject2[String, ReferenceGenome, (Locus, IndexedSeq[String])](
            VariantMethods.getClass, "parse", svar, rgCode(r.mb, plocus.rg))
        emitVariant(cb, r.region, variant, rt)
    }

    registerIEmitCode2("LocusInterval", TString, TBoolean, tinterval("T"), {
      (returnType: Type, _: EmitType, _: EmitType) => {
        val lPTyp = returnType.asInstanceOf[TInterval].pointType.asInstanceOf[TLocus]
        EmitType(PCanonicalInterval(PCanonicalLocus(lPTyp.asInstanceOf[TLocus].rg)).sType, false)
      }
    }) { case (cb: EmitCodeBuilder, r: Value[Region], SIntervalPointer(rt: PCanonicalInterval), _, locusStrEC: EmitCode, invalidMissingEC: EmitCode) =>
      val plocus = rt.pointType.asInstanceOf[PLocus]


      locusStrEC.toI(cb).flatMap(cb) { locusStr =>
        invalidMissingEC.toI(cb).flatMap(cb) { invalidMissing =>

          val Lmissing = CodeLabel()
          val Ldefined = CodeLabel()

          val interval = cb.newLocal[Interval]("locus_interval_interval",
            Code.invokeScalaObject3[String, ReferenceGenome, Boolean, Interval](
              locusClass, "parseInterval",
              locusStr.asString.loadString(cb),
              rgCode(cb.emb, plocus.rg),
              invalidMissing.asBoolean.value))

          cb.if_(interval.isNull, cb.goto(Lmissing))

          val intervalCode = emitLocusInterval(cb, r, interval, rt)
          cb.goto(Ldefined)
          IEmitCode(Lmissing, Ldefined, intervalCode, false)
        }
      }
    }

    registerIEmitCode6("LocusInterval", TString, TInt32, TInt32, TBoolean, TBoolean, TBoolean, tinterval("T"), {
      (returnType: Type, _: EmitType, _: EmitType, _: EmitType, _: EmitType, _: EmitType, _: EmitType) => {
        val lPTyp = returnType.asInstanceOf[TInterval].pointType.asInstanceOf[TLocus]
        EmitType(PCanonicalInterval(PCanonicalLocus(lPTyp.rg)).sType, false)
      }
    }) {
      case (cb: EmitCodeBuilder, r: Value[Region],
      SIntervalPointer(rt: PCanonicalInterval),
      errorID: Value[Int],
      locusString: EmitCode,
      pos1: EmitCode,
      pos2: EmitCode,
      include1: EmitCode,
      include2: EmitCode,
      invalidMissing: EmitCode) =>
        val plocus = rt.pointType.asInstanceOf[PLocus]

        locusString.toI(cb).flatMap(cb) { locusString =>
          pos1.toI(cb).flatMap(cb) { pos1 =>
            pos2.toI(cb).flatMap(cb) { pos2 =>
              include1.toI(cb).flatMap(cb) { include1 =>
                include2.toI(cb).flatMap(cb) { include2 =>
                  invalidMissing.toI(cb).flatMap(cb) { invalidMissing =>

                    val Lmissing = CodeLabel()
                    val Ldefined = CodeLabel()

                    val interval = cb.newLocal[Interval]("locus_interval_interval",
                      Code.invokeScalaObject7[String, Int, Int, Boolean, Boolean, ReferenceGenome, Boolean, Interval](
                        locusClass, "makeInterval",
                        locusString.asString.loadString(cb),
                        pos1.asInt.value,
                        pos2.asInt.value,
                        include1.asBoolean.value,
                        include2.asBoolean.value,
                        rgCode(cb.emb, plocus.rg),
                        invalidMissing.asBoolean.value))

                    cb.if_(interval.isNull, cb.goto(Lmissing))

                    val intervalCode = emitLocusInterval(cb, r, interval, rt)
                    cb.goto(Ldefined)
                    IEmitCode(Lmissing, Ldefined, intervalCode, false)
                  }
                }
              }
            }
          }
        }
    }

    registerSCode1("globalPosToLocus", TInt64, tlocus("T"), {
      (returnType: Type, _: SType) =>
        PCanonicalLocus(returnType.asInstanceOf[TLocus].rg).sType
    }) {
      case (r, cb, SCanonicalLocusPointer(rt: PCanonicalLocus), globalPos, _) =>
        val locus = cb.newLocal[Locus]("global_pos_locus",
          rgCode(r.mb, rt.rg).invoke[Long, Locus]("globalPosToLocus", globalPos.asLong.value))
        rt.constructFromContigAndPosition(cb, r.region, locus.invoke[String]("contig"), locus.invoke[Int]("position"))
    }

    registerSCode1("locusToGlobalPos", tlocus("T"), TInt64, (_: Type, _: SType) => SInt64) {
      case (r, cb, rt, locus: SLocusValue, _) =>
        val locusObject = locus.getLocusObj(cb)
        val globalPos = cb.memoize(rgCode(r.mb, locus.st.rg).invoke[Locus, Long]("locusToGlobalPos", locusObject))
        primitive(globalPos)
    }

    registerIEmitCode2("liftoverLocus", tlocus("T"), TFloat64, TStruct("result" -> tv("U", "locus"), "is_negative_strand" -> TBoolean), {
      (returnType: Type, _: EmitType, _: EmitType) => {
        val lTyp = returnType.asInstanceOf[TStruct].field("result").typ.asInstanceOf[TLocus]
        EmitType(PCanonicalStruct("result" -> PCanonicalLocus(lTyp.rg, true), "is_negative_strand" -> PBoolean(true)).sType, false)
      }
    }) {
      case (cb, r, SBaseStructPointer(rt: PCanonicalStruct), _, loc, minMatch) =>
        loc.toI(cb).flatMap(cb) { case loc: SLocusValue =>
          minMatch.toI(cb).flatMap(cb) { minMatch =>

            val Lmissing = CodeLabel()
            val Ldefined = CodeLabel()

            val locT = loc.st
            val srcRG = locT.rg

            val destRG = rt.types(0).asInstanceOf[PLocus].rg
            val locusObj = loc.getLocusObj(cb)
            val lifted = cb.newLocal[(Locus, Boolean)]("lifterover_locus_ lifted",
              rgCode(cb.emb, srcRG).invoke[String, Locus, Double, (Locus, Boolean)]("liftoverLocus",
                destRG, locusObj, minMatch.asDouble.value))

            cb.if_(lifted.isNull, cb.goto(Lmissing))

            val locType = rt.types(0).asInstanceOf[PCanonicalLocus]
            val locusCode = EmitCode.present(cb.emb, emitLocus(cb, r, Code.checkcast[Locus](lifted.getField[java.lang.Object]("_1")), locType))

            val negativeStrandCode = EmitCode.present(cb.emb,
              primitive(cb.memoize(Code.checkcast[java.lang.Boolean](lifted.getField[java.lang.Object]("_2"))
                .invoke[Boolean]("booleanValue"))))

            val structCode = rt.constructFromFields(cb, r, FastSeq(locusCode, negativeStrandCode), deepCopy = false)

            cb.goto(Ldefined)
            IEmitCode(Lmissing, Ldefined, structCode, false)
          }
        }
    }

    registerIEmitCode2("liftoverLocusInterval", tinterval("T"), TFloat64, TStruct("result" -> tinterval("U"), "is_negative_strand" -> TBoolean), {
      (returnType: Type, _: EmitType, _: EmitType) => {
        val lTyp = returnType.asInstanceOf[TStruct].field("result").typ.asInstanceOf[TInterval].pointType.asInstanceOf[TLocus]
        EmitType(PCanonicalStruct("result" -> PCanonicalInterval(PCanonicalLocus(lTyp.rg, true), true), "is_negative_strand" -> PBoolean(true)).sType, false)
      }
    }) {
      case (cb, r, SBaseStructPointer(rt: PCanonicalStruct), _, interval, minMatch) =>
        interval.toI(cb).flatMap(cb) { interval =>
          minMatch.toI(cb).flatMap(cb) { minMatch =>

            val Lmissing = CodeLabel()
            val Ldefined = CodeLabel()


            val iT = interval.st.asInstanceOf[SInterval]
            val srcRG = iT.pointType.asInstanceOf[SLocus].rg
            val destRG = rt.types(0).asInstanceOf[PInterval].pointType.asInstanceOf[PLocus].rg
            val er = EmitRegion(cb.emb, r)
            val intervalObj = Code.checkcast[Interval](svalueToJavaValue(cb, r, interval))
            val lifted = cb.newLocal[(Interval, Boolean)]("liftover_locus_interval_lifted",
              rgCode(cb.emb, srcRG).invoke[String, Interval, Double, (Interval, Boolean)]("liftoverLocusInterval",
                destRG, intervalObj, minMatch.asDouble.value))


            cb.if_(lifted.isNull, cb.goto(Lmissing))

            val iType = rt.types(0).asInstanceOf[PCanonicalInterval]
            val intervalCode = EmitCode.present(cb.emb, emitLocusInterval(cb, r, Code.checkcast[Interval](lifted.getField[java.lang.Object]("_1")), iType))

            val negativeStrandCode = EmitCode.present(cb.emb,
              primitive(cb.memoize(Code.checkcast[java.lang.Boolean](lifted.getField[java.lang.Object]("_2"))
                .invoke[Boolean]("booleanValue"))))

            val structCode = rt.constructFromFields(cb, r, FastSeq(intervalCode, negativeStrandCode), deepCopy = false)


            cb.goto(Ldefined)
            IEmitCode(Lmissing, Ldefined, structCode, false)
          }
        }
    }
  }
}
