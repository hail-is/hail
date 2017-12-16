package is.hail.io.plink

import is.hail.expr.{EvalContext, Parser, TBoolean, TFloat64, TString, Type}
import is.hail.variant.MatrixTable
import is.hail.utils._

object ExportPlink {
  def apply(vsm: MatrixTable, path: String, famExpr: String = "id = s") {
    require(vsm.wasSplit)
    vsm.requireColKeyString("export plink")

    val ec = EvalContext(Map(
      "s" -> (0, TString()),
      "sa" -> (1, vsm.saSignature),
      "global" -> (2, vsm.globalSignature)))

    ec.set(2, vsm.globalAnnotation)

    type Formatter = (Option[Any]) => String

    val formatID: Formatter = _.map(_.asInstanceOf[String]).getOrElse("0")
    val formatIsFemale: Formatter = _.map { a =>
      if (a.asInstanceOf[Boolean])
        "2"
      else
        "1"
    }.getOrElse("0")
    val formatIsCase: Formatter = _.map { a =>
      if (a.asInstanceOf[Boolean])
        "2"
      else
        "1"
    }.getOrElse("-9")
    val formatQPheno: Formatter = a => a.map(_.toString).getOrElse("-9")

    val famColumns: Map[String, (Type, Int, Formatter)] = Map(
      "famID" -> (TString(), 0, formatID),
      "id" -> (TString(), 1, formatID),
      "patID" -> (TString(), 2, formatID),
      "matID" -> (TString(), 3, formatID),
      "isFemale" -> (TBoolean(), 4, formatIsFemale),
      "qPheno" -> (TFloat64(), 5, formatQPheno),
      "isCase" -> (TBoolean(), 5, formatIsCase))

    val (names, types, f) = Parser.parseNamedExprs(famExpr, ec)

    val famFns: Array[(Array[Option[Any]]) => String] = Array(
      _ => "0", _ => "0", _ => "0", _ => "0", _ => "-9", _ => "-9")

    (names.zipWithIndex, types).zipped.foreach { case ((name, i), t) =>
      famColumns.get(name) match {
        case Some((colt, j, formatter)) =>
          if (colt != t)
            fatal(s"invalid type for .fam file column $i: expected $colt, got $t")
          famFns(j) = (a: Array[Option[Any]]) => formatter(a(i))

        case None =>
          fatal(s"no .fam file column $name")
      }
    }

    val spaceRegex = """\s+""".r
    val badSampleIds = vsm.stringSampleIds.filter(id => spaceRegex.findFirstIn(id).isDefined)
    if (badSampleIds.nonEmpty) {
      fatal(
        s"""Found ${ badSampleIds.length } sample IDs with whitespace
           |  Please run `renamesamples' to fix this problem before exporting to plink format
           |  Bad sample IDs: @1 """.stripMargin, badSampleIds)
    }

    val bedHeader = Array[Byte](108, 27, 1)

    // FIXME: don't reevaluate the upstream RDD twice
    vsm.rdd2.mapPartitions(
      ExportBedBimFam.bedRowTransformer(vsm.nSamples, vsm.rdd2.typ.rowType)
    ).saveFromByteArrays(path + ".bed", vsm.hc.tmpDir, header = Some(bedHeader))

    vsm.rdd2.mapPartitions(
      ExportBedBimFam.bimRowTransformer(vsm.rdd2.typ.rowType)
    ).writeTable(path + ".bim", vsm.hc.tmpDir)

    val famRows = vsm
      .sampleIdsAndAnnotations
      .map { case (s, sa) =>
        ec.setAll(s, sa)
        val a = f().map(Option(_))
        famFns.map(_ (a)).mkString("\t")
      }

    vsm.hc.hadoopConf.writeTextFile(path + ".fam")(out =>
      famRows.foreach(line => {
        out.write(line)
        out.write("\n")
      }))
  }
}
