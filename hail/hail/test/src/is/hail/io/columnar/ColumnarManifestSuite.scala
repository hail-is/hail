package is.hail.io.columnar

import is.hail.TestUtils._
import is.hail.backend.ExecuteContext
import is.hail.collection.FastSeq
import is.hail.utils.using
import is.hail.variant.ReferenceGenome

import scala.io.Source
import scala.jdk.CollectionConverters._

import java.util.regex.Pattern

import org.json4s.{JArray, JInt, JNull, JObject, JString, JValue}
import org.json4s.jackson.JsonMethods
import org.junit.jupiter.api.Test

class ColumnarManifestSuite {

  private def fixture(name: String): String = getTestResource(s"columnar/0.1/table/$name")

  private def fixtureJson(name: String)(implicit ctx: ExecuteContext): JObject =
    using(Source.fromInputStream(ctx.fs.open(s"${fixture(name)}/manifest.json"))) { src =>
      JsonMethods.parse(src.mkString).asInstanceOf[JObject]
    }

  private def mtFixture: String = getTestResource("columnar/0.1/matrix_table/missing_entries")

  private def mtFixtureJson(implicit ctx: ExecuteContext): JObject =
    using(Source.fromInputStream(ctx.fs.open(s"$mtFixture/manifest.json"))) { src =>
      JsonMethods.parse(src.mkString).asInstanceOf[JObject]
    }

  private def parse(jv: JValue): ColumnarManifest =
    ColumnarManifest.parse(JsonMethods.compact(jv))

  /** `jv` with the field at `path` set to `value`, or removed if `value` is `None`. */
  private def edit(jv: JValue, path: String*)(value: Option[JValue]): JValue =
    (jv, path) match {
      case (JObject(fields), Seq(name)) =>
        JObject(fields.filter(_._1 != name) ++ value.map(name -> _))
      case (JObject(fields), Seq(name, rest @ _*)) =>
        JObject(fields.map { case (n, v) => n -> (if (n == name) edit(v, rest: _*)(value) else v) })
      case _ => throw new IllegalArgumentException(s"no object at $path")
    }

  private def assertRejected(jv: JValue, message: String): Unit =
    interceptFatal(Pattern.quote(message))(parse(jv))

  @Test def testReadsAKeyedTable(implicit ctx: ExecuteContext): Unit = {
    val m = ColumnarManifest.read(ctx.fs, fixture("keyed"))
    assertEq(m.key, FastSeq("idx"))
    assertEq(m.distinctlyKeyed, true)
    assertEq(m.rowIdField, "__row_id")
    assertEq(m.partitionsFile, "partitions.parquet")
    assertEq(m.globalsFile, Some("globals.parquet"))
    assertEq(
      m.rowsSchema.getFields.asScala.map(_.getName).toSeq,
      Seq("__row_id", "idx", "s", "f64", "f32", "i64", "b", "ms", "arr", "st"),
    )
    assertEq(m.globalsSchema.getFields.asScala.map(_.getName).toSeq, Seq("g", "gs"))
    assertEq(m.referenceGenomes, FastSeq.empty[ReferenceGenome])
  }

  @Test def testEmptyGlobalsHaveNoFile(implicit ctx: ExecuteContext): Unit = {
    val m = ColumnarManifest.read(ctx.fs, fixture("no_globals"))
    assertEq(m.globalsFile, None)
    assert(m.globalsSchema.getFields.isEmpty)
  }

  @Test def testUnkeyedTableIsNotDistinctlyKeyed(implicit ctx: ExecuteContext): Unit = {
    val m = ColumnarManifest.read(ctx.fs, fixture("unkeyed"))
    assertEq(m.key, FastSeq.empty[String])
    assertEq(m.distinctlyKeyed, false)
  }

  @Test def testReadsNonBuiltinReferenceGenomes(implicit ctx: ExecuteContext): Unit = {
    val m = ColumnarManifest.read(ctx.fs, fixture("all_types"))
    assertEq(m.referenceGenomes.map(_.name), FastSeq("ColumnarTestGenome"))
    assertEq(m.referenceGenomes.head.contigs, FastSeq("chr2", "chr10", "chr1", "chrX"))
  }

  @Test def testUsesTheManifestsFilePaths(implicit ctx: ExecuteContext): Unit = {
    val jv = edit(
      edit(fixtureJson("keyed"), "partitions", "path")(Some(JString("elsewhere/p.parquet"))),
      "globals",
      "file",
      "path",
    )(Some(JString("g.parquet")))
    val m = parse(jv)
    assertEq(m.partitionsFile, "elsewhere/p.parquet")
    assertEq(m.globalsFile, Some("g.parquet"))
  }

  @Test def testADirectoryWithoutAManifestIsNotATable(implicit ctx: ExecuteContext): Unit =
    interceptFatal("not a columnar table")(ColumnarManifest.read(
      ctx.fs,
      s"${fixture("keyed")}/rows",
    ))

  @Test def testRejectsAnotherFormat(implicit ctx: ExecuteContext): Unit =
    assertRejected(edit(fixtureJson("keyed"), "format")(Some(JString("other"))), "format")

  @Test def testRejectsAnotherVersion(implicit ctx: ExecuteContext): Unit = {
    assertRejected(edit(fixtureJson("keyed"), "version")(Some(JString("0.2"))), "version '0.2'")
    assertRejected(edit(fixtureJson("keyed"), "version")(Some(JString("1.0"))), "version '1.0'")
  }

  @Test def testATableHasNoMatrixComponents(implicit ctx: ExecuteContext): Unit =
    assertEq(ColumnarManifest.read(ctx.fs, fixture("keyed")).matrix, None)

  @Test def testAMatrixTableNeedsColumnsAndEntries(implicit ctx: ExecuteContext): Unit =
    assertRejected(
      edit(fixtureJson("keyed"), "kind")(Some(JString("matrix_table"))),
      "missing field 'cols'",
    )

  @Test def testRejectsMatrixComponentsInATable(implicit ctx: ExecuteContext): Unit =
    assertRejected(
      edit(fixtureJson("keyed"), "cols")(mtFixtureJson \ "cols" match {
        case o: JObject => Some(o)
        case _ => None
      }),
      "unknown field 'cols'",
    )

  @Test def testReadsAMatrixTable(implicit ctx: ExecuteContext): Unit = {
    val m = ColumnarManifest.read(ctx.fs, mtFixture)
    assertEq(m.key, FastSeq("row_idx"))
    assertEq(m.rowIdField, "__row_id")
    val mt = m.matrix.get
    assertEq(mt.colKey, FastSeq("col_idx"))
    assertEq(mt.colCount, 4)
    assertEq(mt.colsFile, "cols.parquet")
    assertEq(mt.colIdxField, "__col_idx")
    assertEq(mt.entryValidField, "__entry_valid")
    assertEq(
      mt.colsSchema.getFields.asScala.map(_.getName).toSeq,
      Seq("__col_idx", "col_idx", "s"),
    )
    assertEq(
      mt.entriesSchema.getFields.asScala.map(_.getName).toSeq,
      Seq("__row_id", "__entry_valid", "x", "y"),
    )
  }

  @Test def testRejectsUnknownMatrixFields(implicit ctx: ExecuteContext): Unit =
    for (
      path <- Seq(
        Seq("cols", "extra"),
        Seq("cols", "file", "extra"),
        Seq("entries", "extra"),
      )
    )
      assertRejected(edit(mtFixtureJson, path: _*)(Some(JNull)), "unknown field 'extra'")

  @Test def testRejectsMissingMatrixFields(implicit ctx: ExecuteContext): Unit =
    for (
      path <- Seq(
        Seq("entries"),
        Seq("cols", "count"),
        Seq("entries", "column_blocks"),
        Seq("internal_fields", "col_idx"),
        Seq("internal_fields", "entry_valid"),
      )
    )
      assertRejected(edit(mtFixtureJson, path: _*)(None), s"missing field '${path.last}'")

  @Test def testRejectsAnotherColumnsFileContainer(implicit ctx: ExecuteContext): Unit =
    assertRejected(
      edit(mtFixtureJson, "cols", "file", "container")(Some(JString("vortex"))),
      "container 'vortex'",
    )

  @Test def testRejectsTheLongLayout(implicit ctx: ExecuteContext): Unit =
    assertRejected(edit(mtFixtureJson, "entries", "layout")(Some(JString("long"))), "layout 'long'")

  @Test def testRejectsANegativeColumnCount(implicit ctx: ExecuteContext): Unit =
    assertRejected(edit(mtFixtureJson, "cols", "count")(Some(JInt(-1))), "'cols.count'")

  @Test def testRejectsAnUnknownColumnKeyField(implicit ctx: ExecuteContext): Unit =
    assertRejected(
      edit(mtFixtureJson, "cols", "key")(Some(JArray(List(JString("nope"))))),
      "column key field 'nope'",
    )

  private def columnBlocks(blocks: (Int, Int)*): Option[JValue] =
    Some(JArray(blocks.map { case (s, e) => JObject("start" -> JInt(s), "end" -> JInt(e)) }.toList))

  @Test def testRejectsColumnBlocksThatDoNotCoverTheColumns(implicit ctx: ExecuteContext): Unit =
    for (blocks <- Seq(Seq(), Seq(0 -> 3), Seq(1 -> 4), Seq(0 -> 2, 3 -> 4), Seq(0 -> 3, 2 -> 4)))
      assertRejected(
        edit(mtFixtureJson, "entries", "column_blocks")(columnBlocks(blocks: _*)),
        "'entries.column_blocks' must be contiguous ranges that cover the 4 columns",
      )

  @Test def testRejectsMoreThanOneColumnBlock(implicit ctx: ExecuteContext): Unit =
    assertRejected(
      edit(mtFixtureJson, "entries", "column_blocks")(columnBlocks(0 -> 2, 2 -> 4)),
      "only one column block",
    )

  @Test def testAMatrixTableWithNoColumnsHasOneEmptyColumnBlock(implicit ctx: ExecuteContext)
    : Unit = {
    val jv = edit(
      edit(mtFixtureJson, "cols", "count")(Some(JInt(0))),
      "entries",
      "column_blocks",
    )(columnBlocks(0 -> 0))
    assertEq(parse(jv).matrix.get.colCount, 0)
  }

  @Test def testLooksUpTheMatrixInternalFieldsByTheManifestsNames(implicit ctx: ExecuteContext)
    : Unit = {
    assertRejected(
      edit(mtFixtureJson, "internal_fields", "col_idx")(Some(JString("cidx"))),
      "column index field 'cidx'",
    )
    assertRejected(
      edit(mtFixtureJson, "internal_fields", "entry_valid")(Some(JString("ev"))),
      "entry validity field 'ev'",
    )
    assertRejected(
      edit(mtFixtureJson, "internal_fields", "row_id")(Some(JString("rid"))),
      "row ID field 'rid'",
    )
  }

  @Test def testRejectsAnUnknownKind(implicit ctx: ExecuteContext): Unit =
    assertRejected(edit(fixtureJson("keyed"), "kind")(Some(JString("tensor"))), "kind 'tensor'")

  @Test def testRejectsUnknownFields(implicit ctx: ExecuteContext): Unit = {
    val jv = fixtureJson("keyed")
    for (
      path <- Seq(
        Seq("extra"),
        Seq("rows", "extra"),
        Seq("globals", "extra"),
        Seq("globals", "file", "extra"),
        Seq("partitions", "extra"),
        Seq("internal_fields", "extra"),
      )
    )
      assertRejected(edit(jv, path: _*)(Some(JNull)), "unknown field 'extra'")
  }

  @Test def testRejectsMissingFields(implicit ctx: ExecuteContext): Unit = {
    val jv = fixtureJson("keyed")
    for (path <- Seq(Seq("reference_genomes"), Seq("rows", "key"), Seq("partitions", "path")))
      assertRejected(edit(jv, path: _*)(None), s"missing field '${path.last}'")
  }

  @Test def testRejectsAnotherContainer(implicit ctx: ExecuteContext): Unit = {
    val jv = fixtureJson("keyed")
    assertRejected(
      edit(jv, "partitions", "container")(Some(JString("vortex"))),
      "container 'vortex'",
    )
    assertRejected(
      edit(jv, "globals", "file", "container")(Some(JString("vortex"))),
      "container 'vortex'",
    )
  }

  @Test def testRejectsAnUnknownKeyField(implicit ctx: ExecuteContext): Unit =
    assertRejected(
      edit(fixtureJson("keyed"), "rows", "key")(Some(JArray(List(JString("nope"))))),
      "key field 'nope'",
    )

  @Test def testRejectsADistinctlyKeyedUnkeyedTable(implicit ctx: ExecuteContext): Unit =
    assertRejected(
      edit(fixtureJson("keyed"), "rows", "key")(Some(JArray(Nil))),
      "distinctly_keyed",
    )

  @Test def testLooksUpTheRowIdFieldByTheManifestsName(implicit ctx: ExecuteContext): Unit =
    assertRejected(
      edit(fixtureJson("keyed"), "internal_fields", "row_id")(Some(JString("rid"))),
      "row ID field 'rid'",
    )

  @Test def testRejectsAReferenceGenomeWithABuiltinsName(implicit ctx: ExecuteContext): Unit = {
    val jv = fixtureJson("all_types")
    val JArray(List(custom: JObject)) = jv \ "reference_genomes": @unchecked
    val renamed = edit(custom, "name")(Some(JString("GRCh38")))
    assertRejected(
      edit(jv, "reference_genomes")(Some(JArray(List(renamed)))),
      "builtin reference genome 'GRCh38'",
    )
  }
}
