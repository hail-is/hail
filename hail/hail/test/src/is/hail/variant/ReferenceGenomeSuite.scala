package is.hail.variant

import is.hail.HailSuite
import is.hail.backend.HailStateManager
import is.hail.collection.compat.immutable.ArraySeq
import is.hail.expr.ir.EmitFunctionBuilder
import is.hail.io.reference.{FASTAReader, FASTAReaderConfig, LiftOver}
import is.hail.scalacheck.{genLocus, genNonMissing}
import is.hail.types.virtual.{TInterval, TLocus}
import is.hail.utils._

import htsjdk.samtools.reference.ReferenceSequenceFileFactory
import org.scalacheck.Prop.forAll

class ReferenceGenomeSuite extends HailSuite with munit.ScalaCheckSuite {

  test("GRCh37") {
    assert(ctx.references.contains(ReferenceGenome.GRCh37))
    val grch37 = ctx.references(ReferenceGenome.GRCh37)

    assert(grch37.inX("X"))
    assert(grch37.inY("Y"))
    assert(grch37.isMitochondrial("MT"))
    assertEquals(grch37.contigLength("1"), 249250621)

    val parXLocus = Array(Locus("X", 2499520), Locus("X", 155260460))
    val parYLocus = Array(Locus("Y", 50001), Locus("Y", 59035050))
    val nonParXLocus = Array(Locus("X", 50), Locus("X", 50000000))
    val nonParYLocus = Array(Locus("Y", 5000), Locus("Y", 10000000))

    assert(parXLocus.forall(grch37.inXPar))
    assert(parYLocus.forall(grch37.inYPar))
    assert(!nonParXLocus.forall(grch37.inXPar))
    assert(!nonParYLocus.forall(grch37.inYPar))
  }

  test("GRCh38") {
    assert(ctx.references.contains(ReferenceGenome.GRCh38))
    val grch38 = ctx.references(ReferenceGenome.GRCh38)

    assert(grch38.inX("chrX"))
    assert(grch38.inY("chrY"))
    assert(grch38.isMitochondrial("chrM"))
    assertEquals(grch38.contigLength("chr1"), 248956422)

    val parXLocus38 = Array(Locus("chrX", 2781479), Locus("chrX", 156030895))
    val parYLocus38 = Array(Locus("chrY", 50001), Locus("chrY", 57217415))
    val nonParXLocus38 = Array(Locus("chrX", 50), Locus("chrX", 50000000))
    val nonParYLocus38 = Array(Locus("chrY", 5000), Locus("chrY", 10000000))

    assert(parXLocus38.forall(grch38.inXPar))
    assert(parYLocus38.forall(grch38.inYPar))
    assert(!nonParXLocus38.forall(grch38.inXPar))
    assert(!nonParYLocus38.forall(grch38.inYPar))
  }

  test("Assertions") {
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

  test("ContigRemap") {
    val mapping = Map("23" -> "foo")
    interceptFatal("have remapped contigs in reference genome")(
      ctx.references(ReferenceGenome.GRCh37).validateContigRemap(mapping)
    )
  }

  test("ComparisonOps") {
    val rg = ctx.references(ReferenceGenome.GRCh37)

    // Test contigs
    assert(rg.compare("3", "18") < 0)
    assert(rg.compare("18", "3") > 0)
    assertEquals(rg.compare("7", "7"), 0)

    assert(rg.compare("3", "X") < 0)
    assert(rg.compare("X", "3") > 0)
    assertEquals(rg.compare("X", "X"), 0)

    assert(rg.compare("X", "Y") < 0)
    assert(rg.compare("Y", "X") > 0)
    assert(rg.compare("Y", "MT") < 0)
  }

  test("WriteToFile") {
    val tmpFile = ctx.createTmpPath("grWrite", "json")

    val rg = ctx.references(ReferenceGenome.GRCh37)
    rg.copy(name = "GRCh37_2").write(fs, tmpFile)
    val gr2 = ReferenceGenome.fromFile(fs, tmpFile)

    assert(rg.contigs sameElements gr2.contigs)
    assertEquals(rg.lengths, gr2.lengths)
    assertEquals(rg.xContigs, gr2.xContigs)
    assertEquals(rg.yContigs, gr2.yContigs)
    assertEquals(rg.mtContigs, gr2.mtContigs)
    assert(rg.parInput sameElements gr2.parInput)
  }

  property("Fasta") {
    // qualifying the paths to avoid a bug: #15368. This dodges the bug by making sure
    // the paths don't start with "/", which forces FASTAReader.setup to copy and
    // decompress the files.
    val fastaFile = ctx.fs.makeQualified(getTestResource("fake_reference.fasta"))
    val fastaFileGzip = ctx.fs.makeQualified(getTestResource("fake_reference.fasta.gz"))
    val indexFile = ctx.fs.makeQualified(getTestResource("fake_reference.fasta.fai"))

    val rg = ReferenceGenome("test", ArraySeq("a", "b", "c"), Map("a" -> 25, "b" -> 15, "c" -> 10))
    ctx.local(references = ctx.references + (rg.name -> rg)) { ctx =>
      val fr = FASTAReaderConfig(ctx.localTmpdir, ctx.fs, rg, fastaFile, indexFile, 3, 5).reader
      val frGzip =
        FASTAReaderConfig(ctx.localTmpdir, ctx.fs, rg, fastaFileGzip, indexFile, 3, 5).reader
      val refReaderPath =
        FASTAReader.getLocalFastaFile(ctx.localTmpdir, ctx.fs, fastaFile, indexFile)
      val refReaderPathGz =
        FASTAReader.getLocalFastaFile(ctx.localTmpdir, ctx.fs, fastaFileGzip, indexFile)
      val refReader = ReferenceSequenceFileFactory.getReferenceSequenceFile(
        new java.io.File(uriPath(refReaderPath))
      )
      val refReaderGz = ReferenceSequenceFileFactory.getReferenceSequenceFile(
        new java.io.File(uriPath(refReaderPathGz))
      )

      assertEquals(fr.lookup("a", 25, 0, 5), "A")
      assertEquals(fr.lookup("b", 1, 5, 0), "T")
      assertEquals(fr.lookup("c", 5, 10, 10), "GGATCCGTGC")
      assertEquals(
        fr.lookup(Interval(
          Locus("a", 1),
          Locus("a", 5),
          includesStart = true,
          includesEnd = false,
        )),
        "AGGT",
      )
      assertEquals(
        fr.lookup(Interval(
          Locus("a", 20),
          Locus("b", 5),
          includesStart = false,
          includesEnd = false,
        )),
        "ACGTATAAT",
      )
      assertEquals(
        fr.lookup(Interval(
          Locus("a", 20),
          Locus("c", 5),
          includesStart = false,
          includesEnd = false,
        )),
        "ACGTATAATTAAATTAGCCAGGAT",
      )

      ("cache gives same base as from file" |: forAll(genLocus(rg)) { l =>
        val contig = l.contig
        val pos = l.position
        val expected = refReader.getSubsequenceAt(contig, pos.toLong, pos.toLong).getBaseString
        val expectedGz =
          refReaderGz.getSubsequenceAt(contig, pos.toLong, pos.toLong).getBaseString
        assertEquals(expected, expectedGz, "fasta files don't have the same data")
        fr.lookup(contig, pos, 0, 0) == expected && frGzip.lookup(contig, pos, 0, 0) == expectedGz
      }) ++ ("interval test" |: forAll(
        genNonMissing(ctx, TInterval(TLocus(rg.name)))
      ) {
        case i: Interval =>
          val start = i.start.asInstanceOf[Locus]
          val end = i.end.asInstanceOf[Locus]

          val ordering = TLocus(rg.name).ordering(HailStateManager(Map(rg.name -> rg)))

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
                if (rg.contigsIndex.get(pos.contig) == rg.contigs.length - 1)
                  null
                else
                  Locus(rg.contigs(rg.contigsIndex.get(pos.contig) + 1), 1)
            }
            sb.result()
          }

          fr.lookup(
            Interval(start, end, includesStart = true, includesEnd = true)
          ) == getHtsjdkIntervalSequence
      })
    }
  }

  test("SerializeOnFB") {
    val grch38 = ctx.references(ReferenceGenome.GRCh38)
    val fb = EmitFunctionBuilder[String, Boolean](ctx, "serialize_rg")
    val rgfield = fb.getReferenceGenome(grch38.name)
    fb.emit(rgfield.invoke[String, Boolean]("isValidContig", fb.getCodeParam[String](1)))
    ctx.scopedExecution { (cl, fs, tc, r) =>
      val f = fb.resultWithIndex()(cl, fs, tc, r)
      assertEquals(f("X"), grch38.isValidContig("X"))
    }
  }

  test("SerializeWithLiftoverOnFB") {
    ctx.local(references = ReferenceGenome.builtinReferences()) { ctx =>
      val grch37 = ctx.references(ReferenceGenome.GRCh37)
      val liftoverFile = getTestResource("grch37_to_grch38_chr20.over.chain.gz")

      grch37.addLiftover(ctx.references("GRCh38"), LiftOver(ctx.fs, liftoverFile))

      val fb =
        EmitFunctionBuilder[String, Locus, Double, (Locus, Boolean)](ctx, "serialize_with_liftover")
      val rgfield = fb.getReferenceGenome(grch37.name)
      fb.emit(rgfield.invoke[String, Locus, Double, (Locus, Boolean)](
        "liftoverLocus",
        fb.getCodeParam[String](1),
        fb.getCodeParam[Locus](2),
        fb.getCodeParam[Double](3),
      ))

      ctx.scopedExecution { (cl, fs, tc, r) =>
        val f = fb.resultWithIndex()(cl, fs, tc, r)
        assertEquals(
          f("GRCh38", Locus("20", 60001), 0.95),
          grch37.liftoverLocus("GRCh38", Locus("20", 60001), 0.95),
        )
      }
    }
  }
}
