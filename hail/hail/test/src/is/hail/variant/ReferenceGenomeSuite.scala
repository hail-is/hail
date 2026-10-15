package is.hail.variant

import is.hail.TestUtils._
import is.hail.annotations.{Region, SafeRow}
import is.hail.asm4s.{classInfo, AsmFunction2RegionLongLong, LongInfo}
import is.hail.backend.{ExecuteContext, HailStateManager}
import is.hail.collection.FastSeq
import is.hail.expr.ir.{freshName, invoke, Compile, EmitFunctionBuilder, SingleCodeEmitParamType}
import is.hail.expr.ir.defs.{F64, GetTupleElement, I32, MakeTuple, Ref}
import is.hail.io.reference.{FASTAReader, FASTAReaderConfig, Liftover, ReferenceSequence}
import is.hail.scalacheck.{genLocus, genNonMissing}
import is.hail.types.physical.{PBaseStruct, PCanonicalLocus, PCanonicalString, PCanonicalTuple}
import is.hail.types.physical.stypes.PTypeReferenceSingleCodeType
import is.hail.types.virtual.{TBoolean, TInterval, TLocus, TString, TStruct}
import is.hail.utils._
import is.hail.variant.BuiltinLoci.{builtin, pcanonicalLocus, tlocus}

import scala.collection.immutable.ArraySeq

import java.io.{ByteArrayInputStream, ByteArrayOutputStream, ObjectInputStream, ObjectOutputStream}

import htsjdk.samtools.reference.ReferenceSequenceFileFactory
import org.apache.spark.sql.Row
import org.junit.jupiter.api.Assertions.assertTrue
import org.junit.jupiter.api.Test
import org.scalacheck.Prop.forAll

class ReferenceGenomeSuite {

  @Test def testGRCh37(implicit ctx: ExecuteContext): Unit = {
    assertTrue(ctx.references.contains(ReferenceGenome.GRCh37))
    val grch37 = ctx.references(ReferenceGenome.GRCh37)

    assertTrue(grch37.inX("X"))
    assertTrue(grch37.inY("Y"))
    assertTrue(grch37.isMitochondrial("MT"))
    assertEq(grch37.contigLength("1"), 249250621)

    val parXLocus = Array(Locus("X", 2499520), Locus("X", 155260460))
    val parYLocus = Array(Locus("Y", 50001), Locus("Y", 59035050))
    val nonParXLocus = Array(Locus("X", 50), Locus("X", 50000000))
    val nonParYLocus = Array(Locus("Y", 5000), Locus("Y", 10000000))

    assertTrue(parXLocus.forall(grch37.inXPar))
    assertTrue(parYLocus.forall(grch37.inYPar))
    assertTrue(!nonParXLocus.forall(grch37.inXPar))
    assertTrue(!nonParYLocus.forall(grch37.inYPar))
  }

  @Test def testGRCh38(implicit ctx: ExecuteContext): Unit = {
    assertTrue(ctx.references.contains(ReferenceGenome.GRCh38))
    val grch38 = ctx.references(ReferenceGenome.GRCh38)

    assertTrue(grch38.inX("chrX"))
    assertTrue(grch38.inY("chrY"))
    assertTrue(grch38.isMitochondrial("chrM"))
    assertEq(grch38.contigLength("chr1"), 248956422)

    val parXLocus38 = Array(Locus("chrX", 2781479), Locus("chrX", 156030895))
    val parYLocus38 = Array(Locus("chrY", 50001), Locus("chrY", 57217415))
    val nonParXLocus38 = Array(Locus("chrX", 50), Locus("chrX", 50000000))
    val nonParYLocus38 = Array(Locus("chrY", 5000), Locus("chrY", 10000000))

    assertTrue(parXLocus38.forall(grch38.inXPar))
    assertTrue(parYLocus38.forall(grch38.inYPar))
    assertTrue(!nonParXLocus38.forall(grch38.inXPar))
    assertTrue(!nonParYLocus38.forall(grch38.inYPar))
  }

  @Test def testAssertions(): Unit = {
    interceptFatal("Must have at least one contig in the reference genome.")(
      ReferenceGenome("test", ArraySeq.empty[String], Map.empty[String, Int])
    )
    interceptFatal("No lengths given for the following contigs:")(ReferenceGenome(
      "test",
      ArraySeq("1", "2", "3"),
      Map("1" -> 5),
    ))
    interceptFatal("Contigs found in 'lengths' that are not present in 'contigs'")(
      ReferenceGenome(
        "test",
        ArraySeq("1", "2", "3"),
        Map("1" -> 5, "2" -> 5, "3" -> 5, "4" -> 100),
      )
    )
    interceptFatal("The following X contig names are absent from the reference:")(
      ReferenceGenome(
        "test",
        ArraySeq("1", "2", "3"),
        Map("1" -> 5, "2" -> 5, "3" -> 5),
        xContigs = Set("X"),
      )
    )
    interceptFatal("The following Y contig names are absent from the reference:")(
      ReferenceGenome(
        "test",
        ArraySeq("1", "2", "3"),
        Map("1" -> 5, "2" -> 5, "3" -> 5),
        yContigs = Set("Y"),
      )
    )
    interceptFatal(
      "The following mitochondrial contig names are absent from the reference:"
    )(ReferenceGenome(
      "test",
      ArraySeq("1", "2", "3"),
      Map("1" -> 5, "2" -> 5, "3" -> 5),
      mtContigs = Set("MT"),
    ))
    interceptFatal("The contig name for PAR interval")(ReferenceGenome(
      "test",
      ArraySeq("1", "2", "3"),
      Map("1" -> 5, "2" -> 5, "3" -> 5),
      parInput = ArraySeq((Locus("X", 1), Locus("X", 5))),
    ))
    interceptFatal("in both X and Y contigs.")(ReferenceGenome(
      "test",
      ArraySeq("1", "2", "3"),
      Map("1" -> 5, "2" -> 5, "3" -> 5),
      xContigs = Set("1"),
      yContigs = Set("1"),
    ))
  }

  @Test def testContigRemap(implicit ctx: ExecuteContext): Unit = {
    val mapping = Map("23" -> "foo")
    interceptFatal("have remapped contigs in reference genome")(
      ctx.references(ReferenceGenome.GRCh37).validateContigRemap(mapping)
    )
  }

  @Test def testComparisonOps(implicit ctx: ExecuteContext): Unit = {
    val rg = ctx.references(ReferenceGenome.GRCh37)

    // Test contigs
    assert(rg.compare("3", "18") < 0)
    assert(rg.compare("18", "3") > 0)
    assertEq(rg.compare("7", "7"), 0)

    assert(rg.compare("3", "X") < 0)
    assert(rg.compare("X", "3") > 0)
    assertEq(rg.compare("X", "X"), 0)

    assert(rg.compare("X", "Y") < 0)
    assert(rg.compare("Y", "X") > 0)
    assert(rg.compare("Y", "MT") < 0)
  }

  @Test def testWriteToFile(implicit ctx: ExecuteContext): Unit = {
    val tmpFile = ctx.createTmpPath("grWrite", "json")

    val rg = ctx.references(ReferenceGenome.GRCh37)
    rg.copy(name = "GRCh37_2").write(ctx.fs, tmpFile)
    val gr2 = ReferenceGenome.fromFile(ctx.fs, tmpFile)

    assertTrue(rg.contigs sameElements gr2.contigs)
    assertEq(rg.lengths, gr2.lengths)
    assertEq(rg.xContigs, gr2.xContigs)
    assertEq(rg.yContigs, gr2.yContigs)
    assertEq(rg.mtContigs, gr2.mtContigs)
    assertTrue(rg.parInput sameElements gr2.parInput)
  }

  @Test def testFasta(implicit ctx: ExecuteContext): Unit = {
    // qualifying the paths to avoid a bug: #15368. This dodges the bug by making sure
    // the paths don't start with "/", which forces FASTAReader.setup to copy and
    // decompress the files.
    val fastaFile = ctx.fs.makeQualified(getTestResource("fake_reference.fasta"))
    val indexFile = ctx.fs.makeQualified(getTestResource("fake_reference.fasta.fai"))

    val rg = ReferenceGenome("test", ArraySeq("a", "b", "c"), Map("a" -> 25, "b" -> 15, "c" -> 10))
    ctx.local(references = ctx.references + (rg.name -> rg)) { ctx =>
      val fr = FASTAReaderConfig(ctx.localTmpdir, ctx.fs, rg, fastaFile, indexFile, 3, 5).reader
      val refReaderPath =
        FASTAReader.getLocalFastaFile(ctx.localTmpdir, ctx.fs, fastaFile, indexFile)
      val refReader = ReferenceSequenceFileFactory.getReferenceSequenceFile(
        new java.io.File(uriPath(refReaderPath))
      )

      check(forAll(genLocus(rg)) { l =>
        val contig = l.contig
        val pos = l.position
        val expected = refReader.getSubsequenceAt(contig, pos.toLong, pos.toLong).getBaseString
        assertEq(fr.lookup(contig, pos, 0, 0), expected)
      })

      check(forAll(genNonMissing(ctx, TInterval(TLocus(rg)))) {
        case i: Interval =>
          val start = i.start.asInstanceOf[Locus]
          val end = i.end.asInstanceOf[Locus]

          val ordering = TLocus(rg).ordering

          def getHtsjdkIntervalSequence: String = {
            val sb = new StringBuilder
            var pos = start
            while (ordering.lteq(pos, end) && pos != null) {
              val endPos =
                if (pos.contig != end.contig) rg.contigLength(pos.contig) else end.position
              sb ++= refReader.getSubsequenceAt(
                pos.contig,
                pos.position.toLong,
                endPos.toLong,
              ).getBaseString
              pos =
                if (rg.getContigIndex(pos.contig) == rg.contigs.length - 1)
                  null
                else
                  Locus(rg.contigs(rg.getContigIndex(pos.contig) + 1), 1)
            }
            sb.result()
          }

          assertEq(
            fr.lookup(
              Interval(start, end, includesStart = true, includesEnd = true)
            ),
            getHtsjdkIntervalSequence,
          )
      })

      assertEq(fr.lookup("a", 25, 0, 5), "A")
      assertEq(fr.lookup("b", 1, 5, 0), "T")
      assertEq(fr.lookup("c", 5, 10, 10), "GGATCCGTGC")
      assertEq(
        fr.lookup(Interval(
          Locus("a", 1),
          Locus("a", 5),
          includesStart = true,
          includesEnd = false,
        )),
        "AGGT",
      )
      assertEq(
        fr.lookup(Interval(
          Locus("a", 20),
          Locus("b", 5),
          includesStart = false,
          includesEnd = false,
        )),
        "ACGTATAAT",
      )
      assertEq(
        fr.lookup(Interval(
          Locus("a", 20),
          Locus("c", 5),
          includesStart = false,
          includesEnd = false,
        )),
        "ACGTATAATTAAATTAGCCAGGAT",
      )
    }
  }

  @Test def testSerializeOnFB(implicit ctx: ExecuteContext): Unit = {
    val grch38 = ctx.references(ReferenceGenome.GRCh38)
    val fb = EmitFunctionBuilder[String, Boolean](ctx, "serialize_rg")
    val rgfield = fb.getReferenceGenome(grch38)
    fb.emit(rgfield.invoke[String, Boolean]("isValidContig", fb.getCodeParam[String](1)))
    ctx.scopedExecution { (cl, fs, tc, r) =>
      val f = fb.resultWithIndex()(cl, fs, tc, r)
      assertEq(f("X"), grch38.isValidContig("X"))
    }
  }

  private def liftoverGRCh37ToGRCh38(implicit ctx: ExecuteContext): Liftover =
    Liftover(
      ctx.fs,
      ctx.references(ReferenceGenome.GRCh37),
      ctx.references(ReferenceGenome.GRCh38),
      getTestResource("grch37_to_grch38_chr20.over.chain.gz"),
    )

  @Test def testSerializeWithLiftoverOnFB(implicit ctx: ExecuteContext): Unit = {
    val liftover = liftoverGRCh37ToGRCh38
    ctx.local(liftovers = Map(liftover.key -> liftover)) { ctx =>
      val fb = EmitFunctionBuilder[Locus, Double, (Locus, Boolean)](ctx, "serialize_with_liftover")
      fb.emit(fb.getLiftover(ReferenceGenome.GRCh37, ReferenceGenome.GRCh38).invoke[
        Locus,
        Double,
        (Locus, Boolean),
      ](
        "queryLocus",
        fb.getCodeParam[Locus](1),
        fb.getCodeParam[Double](2),
      ))

      ctx.scopedExecution { (cl, fs, tc, r) =>
        val f = fb.resultWithIndex()(cl, fs, tc, r)
        assertEq(f(Locus("20", 60001), 0.95), (Locus("chr20", 79360), false))
      }
    }
  }

  private def serialize(o: AnyRef): Array[Byte] = {
    val bytes = new ByteArrayOutputStream()
    using(new ObjectOutputStream(bytes))(_.writeObject(o))
    bytes.toByteArray
  }

  private def roundTrip[A <: AnyRef](a: A): A =
    using(new ObjectInputStream(new ByteArrayInputStream(serialize(a))))(_.readObject())
      .asInstanceOf[A]

  // Batch deserializes every closure in every task, so this bounds the per-task cost of shipping
  // a state manager there.
  @Test def testBuiltinReferencesSerializeSmall(): Unit = {
    val size = serialize(HailStateManager(ReferenceGenome.builtinReferences())).length
    assertTrue(size < 2048, s"serialized size $size")
  }

  @Test def testBuiltinResolvesToTheCache(): Unit = {
    val grch38 = ReferenceGenome.builtinReferences()(ReferenceGenome.GRCh38)
    assertTrue(roundTrip(grch38) eq grch38)
  }

  @Test def testBuiltinsResolveToOneCachedGenome(): Unit =
    for (name <- ReferenceGenome.hailReferences) {
      val cached = ReferenceGenome.builtinReferences()(name)
      assertTrue(ReferenceGenomeEnv.empty.resolve(name) eq cached)
      assertTrue(ReferenceGenomeEnv(Map.empty).resolve(name) eq cached)
      assertTrue(ReferenceGenome.builtinReferences()(name) eq cached)
    }

  @Test def testEqualityIsStructural(): Unit = {
    def mk(lengths: Map[String, Int]): ReferenceGenome =
      ReferenceGenome(
        "test",
        ArraySeq("1", "2", "X"),
        lengths,
        xContigs = Set("X"),
        parInput = ArraySeq((Locus("X", 2), Locus("X", 5))),
      )

    val a = mk(Map("1" -> 10, "2" -> 20, "X" -> 30))
    val b = mk(Map("1" -> 10, "2" -> 20, "X" -> 30))
    val different = mk(Map("1" -> 10, "2" -> 21, "X" -> 30))

    assertTrue(a ne b)
    assertEq(a, b)
    assertEq(a.hashCode, b.hashCode)
    assertTrue(a != different)
    assertTrue(different != a)
  }

  // A builtin is identified by identity with the cache, never by name.
  @Test def testUserGenomeWithBuiltinNameSerializesInline(): Unit = {
    val impostor = ReferenceGenome(
      ReferenceGenome.GRCh38,
      ArraySeq("chr2", "chr1", "chrX"),
      Map("chr1" -> 10, "chr2" -> 20, "chrX" -> 30),
      xContigs = Set("chrX"),
      parInput = ArraySeq((Locus("chrX", 2), Locus("chrX", 5))),
    )
    val result = roundTrip(impostor)
    assertTrue(result ne impostor)
    assertTrue(result ne ReferenceGenome.builtinReferences()(ReferenceGenome.GRCh38))
    assertEq(result, impostor)
    assertEq(result.contigs, ArraySeq("chr2", "chr1", "chrX"))
    assertEq(result.getContigIndex("chr1"), 1)
    assertEq(result.contigLength("chrX"), 30)
    assertTrue(result.inXPar(Locus("chrX", 3)))
    assertTrue(!result.inXPar(Locus("chrX", 6)))
  }

  // Compiled reaches Spark and Batch tasks by Java serialization, and each task instantiates it
  // with its own FS. The session also holds a liftover from the same source genome whose chain file
  // is gone: code that does not use it must neither ship nor restore it.
  @Test def testLiftoverSurvivesCompiledRoundTrip(implicit ctx: ExecuteContext): Unit = {
    val used = liftoverGRCh37ToGRCh38
    val test38 = ctx.references(ReferenceGenome.GRCh38).copy(name = "test38")
    val goneChain = ctx.createTmpPath("liftover", "over.chain.gz")
    ctx.fs.copy(getTestResource("grch37_to_grch38_chr20.over.chain.gz"), goneChain)
    val unused = Liftover(ctx.fs, ctx.references(ReferenceGenome.GRCh37), test38, goneChain)
    ctx.fs.delete(goneChain, recursive = false)

    ctx.local(
      references = ctx.references + (test38.name -> test38),
      liftovers = Map(used.key -> used, unused.key -> unused),
    ) { ctx =>
      val argsPType = PCanonicalTuple(true, pcanonicalLocus(ReferenceGenome.GRCh37, true))
      val args = Ref(freshName(), argsPType.virtualType)
      val lifted = invoke(
        "liftoverLocus",
        TStruct("result" -> tlocus(ReferenceGenome.GRCh38), "is_negative_strand" -> TBoolean),
        GetTupleElement(args, 0),
        F64(0.95),
      )
      val (Some(PTypeReferenceSingleCodeType(rt: PBaseStruct)), compiled) =
        Compile[AsmFunction2RegionLongLong](
          ctx,
          FastSeq(
            args.name -> SingleCodeEmitParamType(true, PTypeReferenceSingleCodeType(argsPType))
          ),
          FastSeq(classInfo[Region], LongInfo),
          LongInfo,
          MakeTuple.ordered(FastSeq(lifted)),
        ): @unchecked

      val shipped = roundTrip(compiled)
      ctx.scopedExecution { (hcl, fs, tc, r) =>
        val argsOff =
          argsPType.unstagedStoreJavaObject(Row(Locus("20", 60001)), r)
        val off = shipped(hcl, fs, tc, r)(r, argsOff)
        assertEq(
          SafeRow(rt, off).get(0),
          Row(Locus("chr20", 79360), false),
        )
      }
    }
  }

  // Qualified for the same reason as in testFasta.
  private def fakeReferenceSequence(rg: ReferenceGenome)(implicit ctx: ExecuteContext)
    : ReferenceSequence =
    ReferenceSequence(
      ctx.fs,
      rg,
      ctx.fs.makeQualified(getTestResource("fake_reference.fasta")),
      ctx.fs.makeQualified(getTestResource("fake_reference.fasta.fai")),
    )

  private val fakeGenome: ReferenceGenome =
    ReferenceGenome("test", ArraySeq("a", "b", "c"), Map("a" -> 25, "b" -> 15, "c" -> 10))

  @Test def testSequenceSurvivesRoundTrip(implicit ctx: ExecuteContext): Unit = {
    val seq = roundTrip(fakeReferenceSequence(fakeGenome))
    seq.restore(ctx.localTmpdir, ctx.fs)
    assertEq(seq.lookup("c", 5, 10, 10), "GGATCCGTGC")
  }

  @Test def testSequenceMustMatchItsGenome(implicit ctx: ExecuteContext): Unit = {
    val shorter = fakeGenome.copy(lengths = fakeGenome.lengths + ("c" -> 9))
    interceptFatal("Contig sizes in FASTA")(fakeReferenceSequence(shorter))
  }

  // Compiled ships only the reference sequences it uses, and restores them with the task's FS. The
  // session also holds a sequence for another genome, whose FASTA path must not be shipped.
  @Test def testSequenceSurvivesCompiledRoundTrip(implicit ctx: ExecuteContext): Unit = {
    val used = fakeReferenceSequence(fakeGenome)
    val otherGenome = fakeGenome.copy(name = "other")
    val otherFasta = ctx.createTmpPath("unused-sequence", "fasta")
    ctx.fs.copy(getTestResource("fake_reference.fasta"), otherFasta)
    val unused = ReferenceSequence(
      ctx.fs,
      otherGenome,
      otherFasta,
      getTestResource("fake_reference.fasta.fai"),
    )

    ctx.local(
      references =
        ctx.references + (fakeGenome.name -> fakeGenome) + (otherGenome.name -> otherGenome),
      sequences = Map(fakeGenome.name -> used, otherGenome.name -> unused),
    ) { ctx =>
      val argsPType = PCanonicalTuple(true, PCanonicalString(true))
      val args = Ref(freshName(), argsPType.virtualType)
      val seq = invoke(
        "getReferenceSequence",
        TString,
        FastSeq(TLocus(fakeGenome)),
        GetTupleElement(args, 0),
        I32(5),
        I32(10),
        I32(10),
      )
      val (Some(PTypeReferenceSingleCodeType(rt: PBaseStruct)), compiled) =
        Compile[AsmFunction2RegionLongLong](
          ctx,
          FastSeq(
            args.name -> SingleCodeEmitParamType(true, PTypeReferenceSingleCodeType(argsPType))
          ),
          FastSeq(classInfo[Region], LongInfo),
          LongInfo,
          MakeTuple.ordered(FastSeq(seq)),
        ): @unchecked

      val bytes = serialize(compiled)
      assertTrue(!bytes.containsSlice(otherFasta.getBytes("UTF-8")))

      val shipped = roundTrip(compiled)
      ctx.scopedExecution { (hcl, fs, tc, r) =>
        val argsOff = argsPType.unstagedStoreJavaObject(Row("c"), r)
        val off = shipped(hcl, fs, tc, r)(r, argsOff)
        assertEq(SafeRow(rt, off).get(0), "GGATCCGTGC")
      }
    }
  }

  // Types reach tasks inside closures, so a type on a builtin reference genome must ship it as a
  // tag: more cheaply than even the smallest user-defined genome, and within the bound on a state
  // manager holding all the builtins.
  @Test def testTypesOnBuiltinsSerializeSmall(): Unit = {
    val smallest = ReferenceGenome("smallest", ArraySeq("1"), Map("1" -> 1))
    for {
      n <- ReferenceGenome.hailReferences
      mk <- ArraySeq[ReferenceGenome => AnyRef](
        TLocus(_),
        PCanonicalLocus(_),
      )
    } {
      val size = serialize(mk(builtin(n))).length
      val inlineSize = serialize(mk(smallest)).length
      assertTrue(size < inlineSize, s"${mk(builtin(n))}: $size, inline: $inlineSize")
    }

    val all = TStruct(ReferenceGenome.hailReferences.toSeq.sorted.map(n => n -> tlocus(n)): _*)
    val size = serialize(all).length
    assertTrue(size < 2048, s"serialized size $size")
  }

  // Every locus type in a session points at the one reference genome, so a type ships it once, and
  // its loci still share one reference genome after the round trip.
  @Test def testTypeShipsItsReferenceGenomeOnce(): Unit = {
    val rg = builtin(ReferenceGenome.GRCh38).copy(name = "user_defined")
    val one = TStruct("l" -> TLocus(rg))
    val many = TStruct((0 until 20).map(i => s"l$i" -> TLocus(rg)): _*)
    val genomeSize = serialize(rg).length
    assertTrue(
      serialize(many).length < serialize(one).length + genomeSize / 2,
      s"one: ${serialize(one).length}, many: ${serialize(many).length}, genome: $genomeSize",
    )

    val result = roundTrip(many)
    assertEq(result, many)
    val rgs = result.types.map(_.asInstanceOf[TLocus].rg)
    assertTrue(rgs.forall(_ eq rgs.head))
  }

  // Locus types compare their reference genomes as reference genomes do: two copies of one layout
  // that are distinct objects are equal, and a genome of the same name with another layout is not.
  @Test def testLocusTypesCompareReferenceGenomes(): Unit = {
    val a = fakeGenome
    val b = fakeGenome.copy()
    val other = fakeGenome.copy(lengths = fakeGenome.lengths + ("c" -> 11))
    assertTrue(a ne b)

    val types = ArraySeq[ReferenceGenome => AnyRef](
      TLocus(_),
      PCanonicalLocus(_, true),
      rg => PCanonicalLocus(rg).sType,
    )
    for (mk <- types) {
      assertEq(mk(a), mk(b))
      assertEq(mk(a).hashCode, mk(b).hashCode)
      assertTrue(mk(a) != mk(other), s"${mk(a)}")
    }

    assertTrue(TLocus(a).unify(TLocus(b)))
    assertTrue(TLocus(a).isIsomorphicTo(TLocus(b)))
    assertTrue(!TLocus(a).unify(TLocus(other)))
    assertTrue(!TLocus(a).isIsomorphicTo(TLocus(other)))
  }
}
