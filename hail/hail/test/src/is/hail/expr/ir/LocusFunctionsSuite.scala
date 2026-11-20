package is.hail.expr.ir

import is.hail.ExecStrategy
import is.hail.ExecStrategy.ExecStrategy
import is.hail.TestUtils._
import is.hail.annotations.{Region, RowSeq}
import is.hail.asm4s._
import is.hail.backend.ExecuteContext
import is.hail.collection.FastSeq
import is.hail.expr.ir.defs.{
  Apply, ErrorIDs, F64, False, GetTupleElement, I32, I64, MakeArray, MakeTuple, NA, Ref, Str, True,
}
import is.hail.io.reference.{Liftover, RgAttachments}
import is.hail.types.physical.{
  PBoolean, PCanonicalArray, PCanonicalInterval, PCanonicalLocus, PCanonicalString,
  PCanonicalStruct, PCanonicalTuple, PTuple, PType,
}
import is.hail.types.physical.LocusRepresentations.{
  customReference, forEachCanonicalLocus, representations, withReference,
}
import is.hail.types.physical.stypes.PTypeReferenceSingleCodeType
import is.hail.types.virtual._
import is.hail.utils.Interval
import is.hail.variant.{BuiltinLoci, Locus, ReferenceGenome}

import org.junit.jupiter.api.Test

class LocusFunctionsSuite {

  implicit val execStrats: Set[ExecStrategy] = ExecStrategy.javaOnly

  private def grch38(implicit ctx: ExecuteContext): ReferenceGenome =
    ctx.references(ReferenceGenome.GRCh38)

  private def tlocus(implicit ctx: ExecuteContext) = TLocus(grch38)

  private def tvariant(implicit ctx: ExecuteContext) =
    TStruct("locus" -> tlocus, "alleles" -> TArray(TString))

  def locusIR(implicit ctx: ExecuteContext): Apply =
    Apply("Locus", FastSeq(), FastSeq(Str("chr22"), I32(1)), tlocus, ErrorIDs.NO_ERROR)

  def locus(implicit ctx: ExecuteContext) = Locus("chr22", 1, grch38)

  /** The PType that `body` returns when passed a tuple of arguments of type `argsType`. The result
    * is made optional because requiredness inference, not the function, decides whether it is
    * required. `body` is compiled without the optimiser, which would fold a function call on
    * literals into a literal whose type doesn't come from the function.
    */
  private def returnPType(argsType: PCanonicalTuple, body: IR => IR)(implicit ctx: ExecuteContext)
    : PType =
    unoptimized { ctx =>
      val args = Ref(freshName(), argsType.virtualType)
      val (Some(PTypeReferenceSingleCodeType(pt: PTuple)), _) =
        Compile[AsmFunction2RegionLongLong](
          ctx,
          FastSeq(
            args.name -> SingleCodeEmitParamType(true, PTypeReferenceSingleCodeType(argsType))
          ),
          FastSeq(classInfo[Region], LongInfo),
          LongInfo,
          MakeTuple.ordered(FastSeq(body(args))),
        ): @unchecked
      pt.types(0).setRequired(false)
    }

  /** The PType that `body` returns when passed an argument of type `argType`. */
  private def returnPType(argType: PType)(body: IR => IR)(implicit ctx: ExecuteContext): PType =
    returnPType(PCanonicalTuple(true, argType), args => body(GetTupleElement(args, 0)))

  /** The PType that `ir` returns. */
  private def returnPType(ir: IR)(implicit ctx: ExecuteContext): PType =
    returnPType(PCanonicalTuple(true), _ => ir)

  /** Each reference genome the producers are tested on, with a contig in it. */
  private def referencesAndContigs(implicit ctx: ExecuteContext)
    : IndexedSeq[(ReferenceGenome, String)] =
    FastSeq((grch38, "chr22"), (customReference, "b"))

  /** A required locus on `rg` in each representation. */
  private def locusTypes(rg: ReferenceGenome): IndexedSeq[PType] =
    representations(PCanonicalLocus(rg, true), rg)

  @Test def addOnContig(implicit ctx: ExecuteContext): Unit =
    forEachCanonicalLocus { canonicalLocusType =>
      for {
        (rg, cases) <- FastSeq(
          grch38 -> FastSeq(
            (Locus("chr22", 5), 10, Locus("chr22", 15)),
            (Locus("chr22", 5), -10, Locus("chr22", 1)),
            (Locus("chr22", 5), 60000000, Locus("chr22", 50818468)),
          ),
          customReference -> FastSeq(
            (Locus("b", 5), 10, Locus("b", 15)),
            (Locus("a", 50), -100, Locus("a", 1)),
            (Locus("b", 5), 5000, Locus("b", 2000)),
          ),
        )
      } withReference(rg) { implicit ctx =>
        val tl = TLocus(rg)
        for (argType <- locusTypes(rg))
          assertEq(
            returnPType(argType)(l => invoke("add_on_contig", tl, l, I32(1))),
            canonicalLocusType(rg, false),
          )
        for ((l, n, expected) <- cases)
          assertEvalsTo(
            invoke(
              "add_on_contig",
              tl,
              invoke("Locus", tl, Str(l.contig), I32(l.position)),
              I32(n),
            ),
            expected,
          )
      }
    }

  @Test def parseLocus(implicit ctx: ExecuteContext): Unit =
    forEachCanonicalLocus { canonicalLocusType =>
      for ((rg, contig) <- referencesAndContigs)
        withReference(rg) { implicit ctx =>
          val ir = invoke("Locus", TLocus(rg), Str(s"$contig:5"))
          assertEq(returnPType(ir), canonicalLocusType(rg, false))
          assertEvalsTo(ir, Locus(contig, 5, rg))
          assertFatal(
            invoke("Locus", TLocus(rg), Str(s"$contig:0")),
            "Position '0' is not within the range",
          )
        }
    }

  @Test def locusFromContigAndPosition(implicit ctx: ExecuteContext): Unit =
    forEachCanonicalLocus { canonicalLocusType =>
      for ((rg, contig) <- referencesAndContigs)
        withReference(rg) { implicit ctx =>
          val ir = invoke("Locus", TLocus(rg), Str(contig), I32(5))
          assertEq(returnPType(ir), canonicalLocusType(rg, false))
          assertEvalsTo(ir, Locus(contig, 5, rg))
          assertFatal(
            invoke("Locus", TLocus(rg), Str(contig), I32(0)),
            "Position '0' is not within the range",
          )
        }
    }

  @Test def contig(implicit ctx: ExecuteContext): Unit =
    assertEvalsTo(invoke("contig", TString, locusIR), locus.contig)

  @Test def position(implicit ctx: ExecuteContext): Unit =
    assertEvalsTo(invoke("position", TInt32, locusIR), locus.position)

  @Test def isAutosomalOrPseudoAutosomal(implicit ctx: ExecuteContext): Unit =
    assertEvalsTo(
      invoke("isAutosomalOrPseudoAutosomal", TBoolean, locusIR),
      locus.isAutosomalOrPseudoAutosomal(grch38),
    )

  @Test def isAutosomal(implicit ctx: ExecuteContext): Unit =
    assertEvalsTo(invoke("isAutosomal", TBoolean, locusIR), locus.isAutosomal(grch38))

  @Test def inYNonPar(implicit ctx: ExecuteContext): Unit =
    assertEvalsTo(invoke("inYNonPar", TBoolean, locusIR), locus.inYNonPar(grch38))

  @Test def inXPar(implicit ctx: ExecuteContext): Unit =
    assertEvalsTo(invoke("inXPar", TBoolean, locusIR), locus.inXPar(grch38))

  @Test def isMitochondrial(implicit ctx: ExecuteContext): Unit =
    assertEvalsTo(invoke("isMitochondrial", TBoolean, locusIR), locus.isMitochondrial(grch38))

  @Test def inXNonPar(implicit ctx: ExecuteContext): Unit =
    assertEvalsTo(invoke("inXNonPar", TBoolean, locusIR), locus.inXNonPar(grch38))

  @Test def inYPar(implicit ctx: ExecuteContext): Unit =
    assertEvalsTo(invoke("inYPar", TBoolean, locusIR), locus.inYPar(grch38))

  @Test def minRep(implicit ctx: ExecuteContext): Unit = {
    val alleles = MakeArray(FastSeq(Str("AA"), Str("AT")), TArray(TString))
    assertEvalsTo(
      invoke("min_rep", tvariant, locusIR, alleles),
      RowSeq(Locus("chr22", 2), FastSeq("A", "T")),
    )
    assertEvalsTo(invoke("min_rep", tvariant, locusIR, NA(TArray(TString))), null)
  }

  /** The type of a variant struct whose locus has type `locusType`. */
  private def variantPType(locusType: PType): PType =
    PCanonicalStruct(
      "locus" -> locusType,
      "alleles" -> PCanonicalArray(PCanonicalString(true), true),
    )

  @Test def minRepOfEachRepresentation(implicit ctx: ExecuteContext): Unit =
    forEachCanonicalLocus { canonicalLocusType =>
      for ((rg, contig) <- referencesAndContigs)
        withReference(rg) { implicit ctx =>
          val tl = TLocus(rg)
          val tv = TStruct("locus" -> tl, "alleles" -> TArray(TString))
          val alleles = MakeArray(FastSeq(Str("AA"), Str("AT")), TArray(TString))
          for (argType <- locusTypes(rg))
            assertEq(
              returnPType(argType)(l => invoke("min_rep", tv, l, alleles)),
              variantPType(canonicalLocusType(rg, true)),
            )
          val l = invoke("Locus", tl, Str(contig), I32(5))
          assertEvalsTo(
            invoke("min_rep", tv, l, alleles),
            RowSeq(Locus(contig, 6), FastSeq("A", "T")),
          )
          assertEvalsTo(invoke("min_rep", tv, l, NA(TArray(TString))), null)
        }
    }

  @Test def parseVariant(implicit ctx: ExecuteContext): Unit =
    forEachCanonicalLocus { canonicalLocusType =>
      for ((rg, contig) <- referencesAndContigs)
        withReference(rg) { implicit ctx =>
          val tv = TStruct("locus" -> TLocus(rg), "alleles" -> TArray(TString))
          val ir = invoke("LocusAlleles", tv, Str(s"$contig:5:A:T,C"))
          assertEq(returnPType(ir), variantPType(canonicalLocusType(rg, true)))
          assertEvalsTo(ir, RowSeq(Locus(contig, 5), FastSeq("A", "T", "C")))
          assertFatal(
            invoke("LocusAlleles", tv, Str(s"$contig:0:A:T")),
            "Position '0' is not within the range",
          )
        }
    }

  @Test def parseLocusInterval(implicit ctx: ExecuteContext): Unit =
    forEachCanonicalLocus { canonicalLocusType =>
      for ((rg, contig) <- referencesAndContigs)
        withReference(rg) { implicit ctx =>
          val ti = TInterval(TLocus(rg))
          def parse(s: String, invalidMissing: Boolean): IR =
            invoke("LocusInterval", ti, Str(s), invalidMissing)

          val ir = parse(s"[$contig:5-$contig:10]", invalidMissing = false)
          assertEq(returnPType(ir), PCanonicalInterval(canonicalLocusType(rg, false)))
          assertEvalsTo(ir, Interval(Locus(contig, 5), Locus(contig, 10), true, true))

          val invalid = s"[$contig:0-$contig:10]"
          assertEvalsTo(parse(invalid, invalidMissing = true), null)
          assertFatal(
            parse(invalid, invalidMissing = false),
            s"Start '$contig:0' is not within the range",
          )
        }
    }

  @Test def locusIntervalFromContigAndPositions(implicit ctx: ExecuteContext): Unit =
    forEachCanonicalLocus { canonicalLocusType =>
      for ((rg, contig) <- referencesAndContigs)
        withReference(rg) { implicit ctx =>
          val ti = TInterval(TLocus(rg))
          def make(start: Int, end: Int, invalidMissing: Boolean): IR =
            invoke(
              "LocusInterval",
              ti,
              Str(contig),
              I32(start),
              I32(end),
              True(),
              False(),
              invalidMissing,
            )

          val ir = make(5, 10, invalidMissing = false)
          assertEq(returnPType(ir), PCanonicalInterval(canonicalLocusType(rg, false)))
          assertEvalsTo(ir, Interval(Locus(contig, 5), Locus(contig, 10), true, false))

          assertEvalsTo(make(0, 10, invalidMissing = true), null)
          assertFatal(
            make(0, 10, invalidMissing = false),
            s"Start '$contig:0' is not within the range",
          )
        }
    }

  /** Runs `f` in a session that can lift over from GRCh37 to GRCh38 on chromosome 20. */
  private def withLiftover[A](f: ExecuteContext => A)(implicit ctx: ExecuteContext): A = {
    val liftover = Liftover(
      ctx.fs,
      ctx.references(ReferenceGenome.GRCh37),
      grch38,
      getTestResource("grch37_to_grch38_chr20.over.chain.gz"),
    )
    ctx.local(attachments = RgAttachments.empty.withLiftover(liftover))(f)
  }

  private def liftedPType(resultType: PType): PType =
    PCanonicalStruct("result" -> resultType, "is_negative_strand" -> PBoolean(true))

  @Test def liftoverLocus(implicit ctx: ExecuteContext): Unit =
    forEachCanonicalLocus { canonicalLocusType =>
      withLiftover { implicit ctx =>
        val grch37 = ctx.references(ReferenceGenome.GRCh37)
        val rt = TStruct("result" -> tlocus, "is_negative_strand" -> TBoolean)
        def lift(l: IR): IR = invoke("liftoverLocus", rt, l, F64(0.95))
        def locus37(position: Int): IR = invoke("Locus", TLocus(grch37), Str("20"), I32(position))

        for (argType <- locusTypes(grch37))
          assertEq(returnPType(argType)(lift), liftedPType(canonicalLocusType(grch38, true)))
        assertEvalsTo(lift(locus37(60001)), RowSeq(Locus("chr20", 79360), false))
        assertEvalsTo(lift(locus37(1)), null)
      }
    }

  @Test def liftoverLocusInterval(implicit ctx: ExecuteContext): Unit =
    forEachCanonicalLocus { canonicalLocusType =>
      withLiftover { implicit ctx =>
        val grch37 = ctx.references(ReferenceGenome.GRCh37)
        val rt = TStruct("result" -> TInterval(tlocus), "is_negative_strand" -> TBoolean)
        def lift(i: IR): IR = invoke("liftoverLocusInterval", rt, i, F64(0.95))
        def interval37(start: Int, end: Int): IR =
          invoke("LocusInterval", TInterval(TLocus(grch37)), Str(s"[20:$start-20:$end]"), False())

        for (argType <- locusTypes(grch37))
          assertEq(
            returnPType(PCanonicalInterval(argType, true))(lift),
            liftedPType(PCanonicalInterval(canonicalLocusType(grch38, true), true)),
          )
        assertEvalsTo(
          lift(interval37(60001, 60010)),
          RowSeq(Interval(Locus("chr20", 79360), Locus("chr20", 79369), true, true), false),
        )
        assertEvalsTo(lift(interval37(1, 10)), null)
      }
    }

  @Test def globalPosition(implicit ctx: ExecuteContext): Unit =
    assertEvalsTo(invoke("locusToGlobalPos", TInt64, locusIR), grch38.locusToGlobalPos(locus))

  @Test def reverseGlobalPosition(implicit ctx: ExecuteContext): Unit = {
    val globalPosition = 2824183054L
    assertEvalsTo(
      invoke("globalPosToLocus", tlocus, I64(globalPosition)),
      grch38.globalPosToLocus(globalPosition),
    )
  }

  @Test def globalPositionToLocus(implicit ctx: ExecuteContext): Unit =
    forEachCanonicalLocus { canonicalLocusType =>
      for {
        (rg, globalPosition, expected) <- FastSeq(
          (grch38, 0L, Locus("chr1", 1)),
          // The custom reference orders its contigs c (30000), a (100), b (2000).
          (customReference, 30000L + 100L + 4L, Locus("b", 5)),
        )
      } withReference(rg) { implicit ctx =>
        val ir = invoke("globalPosToLocus", TLocus(rg), I64(globalPosition))
        assertEq(returnPType(ir), canonicalLocusType(rg, false))
        assertEvalsTo(ir, expected)
      }
    }

  @Test def contigIdx(implicit ctx: ExecuteContext): Unit =
    for {
      (rg, contigs, expected) <- FastSeq(
        (grch38, FastSeq("chr22", "chr1", "chr22", "chr22"), RowSeq(21, 0, 21, 21)),
        (customReference, FastSeq("a", "c", "a", "b"), RowSeq(1, 0, 1, 2)),
      )
    } withReference(rg) { implicit ctx =>
      val tl = TLocus(rg)
      val ir = MakeTuple.ordered(contigs.map { c =>
        invoke("contig_idx", TInt32, invoke("Locus", tl, Str(c), I32(1)))
      })
      assertEvalsTo(ir, expected)
    }

  @Test def testMultipleReferenceGenomes(implicit ctx: ExecuteContext): Unit = {
    implicit val execStrats = ExecStrategy.compileOnly

    val ir = MakeTuple.ordered(FastSeq(
      invoke("Locus", BuiltinLoci.tlocus(ReferenceGenome.GRCh37), Str("1"), I32(1)),
      invoke("Locus", BuiltinLoci.tlocus(ReferenceGenome.GRCh38), Str("chr1"), I32(1)),
    ))

    assertEvalsTo(
      ir,
      RowSeq(
        Locus("1", 1, ctx.references(ReferenceGenome.GRCh37)),
        Locus("chr1", 1, ctx.references(ReferenceGenome.GRCh38)),
      ),
    )
  }

  @Test def testMakeInterval(implicit ctx: ExecuteContext): Unit = {
    // TString, TInt32, TInt32, TBoolean, TBoolean, TBoolean
    val ir = MakeTuple.ordered(FastSeq(
      invoke(
        "LocusInterval",
        TInterval(TLocus(grch38)),
        NA(TString),
        I32(1),
        I32(100),
        True(),
        True(),
        False(),
      ),
      invoke(
        "LocusInterval",
        TInterval(TLocus(grch38)),
        Str("chr1"),
        NA(TInt32),
        I32(100),
        True(),
        True(),
        False(),
      ),
      invoke(
        "LocusInterval",
        TInterval(TLocus(grch38)),
        Str("chr1"),
        I32(1),
        NA(TInt32),
        True(),
        True(),
        False(),
      ),
      invoke(
        "LocusInterval",
        TInterval(TLocus(grch38)),
        Str("chr1"),
        I32(1),
        I32(100),
        NA(TBoolean),
        True(),
        False(),
      ),
      invoke(
        "LocusInterval",
        TInterval(TLocus(grch38)),
        Str("chr1"),
        I32(1),
        I32(100),
        True(),
        NA(TBoolean),
        False(),
      ),
      invoke(
        "LocusInterval",
        TInterval(TLocus(grch38)),
        Str("chr1"),
        I32(1),
        I32(100),
        True(),
        True(),
        NA(TBoolean),
      ),
      invoke(
        "LocusInterval",
        TInterval(TLocus(grch38)),
        Str("chr1"),
        I32(-1),
        I32(0),
        True(),
        True(),
        True(),
      ),
      invoke(
        "LocusInterval",
        TInterval(TLocus(grch38)),
        Str("chr1"),
        I32(1),
        I32(100),
        True(),
        True(),
        True(),
      ),
    ))

    assertEvalsTo(
      ir,
      RowSeq(
        null,
        null,
        null,
        null,
        null,
        null,
        null,
        Interval(Locus("chr1", 1, grch38), Locus("chr1", 100, grch38), true, true),
      ),
    )
  }
}
