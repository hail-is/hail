package is.hail.variant

import is.hail.HailSuite
import is.hail.backend.HailStateManager
import is.hail.expr.ir.EmitFunctionBuilder
import is.hail.io.reference.{FASTAReader, FASTAReaderConfig, LiftOver}
import is.hail.scalacheck.{genLocus, genNullable}
import is.hail.types.virtual.{TInterval, TLocus}
import is.hail.utils._

import htsjdk.samtools.reference.ReferenceSequenceFileFactory
import org.scalacheck.Prop.forAll
import org.scalatest
import org.testng.annotations.Test

class ReferenceGenomeSuite extends HailSuite {

  @Test def testGRCh37(): scalatest.Assertion = {
    assert(ctx.references.contains(ReferenceGenome.GRCh37))
    val grch37 = ctx.references(ReferenceGenome.GRCh37)

    assert(grch37.inX("X") && grch37.inY("Y") && grch37.isMitochondrial("MT"))
    assert(grch37.contigLength("1") == 249250621)

    val parXLocus = Array(Locus("X", 2499520), Locus("X", 155260460))
    val parYLocus = Array(Locus("Y", 50001), Locus("Y", 59035050))
    val nonParXLocus = Array(Locus("X", 50), Locus("X", 50000000))
    val nonParYLocus = Array(Locus("Y", 5000), Locus("Y", 10000000))

    assert(parXLocus.forall(grch37.inXPar) && parYLocus.forall(grch37.inYPar))
    assert(!nonParXLocus.forall(grch37.inXPar) && !nonParYLocus.forall(grch37.inYPar))
  }

  @Test def testGRCh38(): scalatest.Assertion = {
    assert(ctx.references.contains(ReferenceGenome.GRCh38))
    val grch38 = ctx.references(ReferenceGenome.GRCh38)

    assert(grch38.inX("chrX") && grch38.inY("chrY") && grch38.isMitochondrial("chrM"))
    assert(grch38.contigLength("chr1") == 248956422)

    val parXLocus38 = Array(Locus("chrX", 2781479), Locus("chrX", 156030895))
    val parYLocus38 = Array(Locus("chrY", 50001), Locus("chrY", 57217415))
    val nonParXLocus38 = Array(Locus("chrX", 50), Locus("chrX", 50000000))
    val nonParYLocus38 = Array(Locus("chrY", 5000), Locus("chrY", 10000000))

    assert(parXLocus38.forall(grch38.inXPar) && parYLocus38.forall(grch38.inYPar))
    assert(!nonParXLocus38.forall(grch38.inXPar) && !nonParYLocus38.forall(grch38.inYPar))
  }

  @Test def testAssertions(): scalatest.Assertion = {
    interceptFatal("Must have at least one contig in the reference genome.")(
      ReferenceGenome("test", Array.empty[String], Map.empty[String, Int])
    )
    interceptFatal("No lengths given for the following contigs:")(ReferenceGenome(
      "test",
      Array("1", "2", "3"),
      Map("1" -> 5),
    ))
    interceptFatal("Contigs found in 'lengths' that are not present in 'contigs'")(
      ReferenceGenome("test", Array("1", "2", "3"), Map("1" -> 5, "2" -> 5, "3" -> 5, "4" -> 100))
    )
    interceptFatal("The following X contig names are absent from the reference:")(
      ReferenceGenome(
        "test",
        Array("1", "2", "3"),
        Map("1" -> 5, "2" -> 5, "3" -> 5),
        xContigs = Set("X"),
      )
    )
    interceptFatal("The following Y contig names are absent from the reference:")(
      ReferenceGenome(
        "test",
        Array("1", "2", "3"),
        Map("1" -> 5, "2" -> 5, "3" -> 5),
        yContigs = Set("Y"),
      )
    )
    interceptFatal(
      "The following mitochondrial contig names are absent from the reference:"
    )(ReferenceGenome(
      "test",
      Array("1", "2", "3"),
      Map("1" -> 5, "2" -> 5, "3" -> 5),
      mtContigs = Set("MT"),
    ))
    interceptFatal("The contig name for PAR interval")(ReferenceGenome(
      "test",
      Array("1", "2", "3"),
      Map("1" -> 5, "2" -> 5, "3" -> 5),
      parInput = Array((Locus("X", 1), Locus("X", 5))),
    ))
    interceptFatal("in both X and Y contigs.")(ReferenceGenome(
      "test",
      Array("1", "2", "3"),
      Map("1" -> 5, "2" -> 5, "3" -> 5),
      xContigs = Set("1"),
      yContigs = Set("1"),
    ))
  }

  @Test def testContigRemap(): scalatest.Assertion = {
    val mapping = Map("23" -> "foo")
    interceptFatal("have remapped contigs in reference genome")(
      ctx.references(ReferenceGenome.GRCh37).validateContigRemap(mapping)
    )
  }

  @Test def testComparisonOps(): scalatest.Assertion = {
    val rg = ctx.references(ReferenceGenome.GRCh37)

    // Test contigs
    assert(rg.compare("3", "18") < 0)
    assert(rg.compare("18", "3") > 0)
    assert(rg.compare("7", "7") == 0)

    assert(rg.compare("3", "X") < 0)
    assert(rg.compare("X", "3") > 0)
    assert(rg.compare("X", "X") == 0)

    assert(rg.compare("X", "Y") < 0)
    assert(rg.compare("Y", "X") > 0)
    assert(rg.compare("Y", "MT") < 0)
  }

  @Test def testWriteToFile(): scalatest.Assertion = {
    val tmpFile = ctx.createTmpPath("grWrite", "json")

    val rg = ctx.references(ReferenceGenome.GRCh37)
    rg.copy(name = "GRCh37_2").write(fs, tmpFile)
    val gr2 = ReferenceGenome.fromFile(fs, tmpFile)

    assert((rg.contigs sameElements gr2.contigs) &&
      rg.lengths == gr2.lengths &&
      rg.xContigs == gr2.xContigs &&
      rg.yContigs == gr2.yContigs &&
      rg.mtContigs == gr2.mtContigs &&
      (rg.parInput sameElements gr2.parInput))
  }

  @Test def testFasta(): scalatest.Assertion = {
    val fastaFile = getTestResource("fake_reference.fasta")
    val fastaFileGzip = getTestResource("fake_reference.fasta.gz")
    val indexFile = getTestResource("fake_reference.fasta.fai")

    val rg = ReferenceGenome("test", Array("a", "b", "c"), Map("a" -> 25, "b" -> 15, "c" -> 10))
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

      {
        "cache gives same base as from file" |: forAll(genLocus(rg)) { l =>
          val contig = l.contig
          val pos = l.position
          val expected = refReader.getSubsequenceAt(contig, pos, pos).getBaseString
          val expectedGz = refReaderGz.getSubsequenceAt(contig, pos, pos).getBaseString
          assert(expected == expectedGz, "wat: fasta files don't have the same data")
          fr.lookup(contig, pos, 0, 0) == expected && frGzip.lookup(contig, pos, 0, 0) == expectedGz
        }
      }.check()

      {
        "interval test" |: forAll(
          genNullable(ctx, TInterval(TLocus(rg.name))).suchThat(_ != null)
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
                sb ++= refReader.getSubsequenceAt(pos.contig, pos.position, endPos).getBaseString
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
        }
      }.check()

      assert(fr.lookup("a", 25, 0, 5) == "A")
      assert(fr.lookup("b", 1, 5, 0) == "T")
      assert(fr.lookup("c", 5, 10, 10) == "GGATCCGTGC")
      assert(fr.lookup(Interval(
        Locus("a", 1),
        Locus("a", 5),
        includesStart = true,
        includesEnd = false,
      )) == "AGGT")
      assert(fr.lookup(Interval(
        Locus("a", 20),
        Locus("b", 5),
        includesStart = false,
        includesEnd = false,
      )) == "ACGTATAAT")
      assert(fr.lookup(Interval(
        Locus("a", 20),
        Locus("c", 5),
        includesStart = false,
        includesEnd = false,
      )) == "ACGTATAATTAAATTAGCCAGGAT")
    }
  }

  @Test def testSerializeOnFB(): scalatest.Assertion = {
    val grch38 = ctx.references(ReferenceGenome.GRCh38)
    val fb = EmitFunctionBuilder[String, Boolean](ctx, "serialize_rg")
    val rgfield = fb.getReferenceGenome(grch38.name)
    fb.emit(rgfield.invoke[String, Boolean]("isValidContig", fb.getCodeParam[String](1)))
    ctx.scopedExecution { (cl, fs, tc, r) =>
      val f = fb.resultWithIndex()(cl, fs, tc, r)
      assert(f("X") == grch38.isValidContig("X"))
    }
  }

  @Test def testSerializeWithLiftoverOnFB(): scalatest.Assertion =
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
        assert(f("GRCh38", Locus("20", 60001), 0.95) == grch37.liftoverLocus(
          "GRCh38",
          Locus("20", 60001),
          0.95,
        ))
      }
    }
}
