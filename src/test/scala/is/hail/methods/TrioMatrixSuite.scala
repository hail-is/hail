package is.hail.methods

import is.hail.SparkSuite
import is.hail.keytable.{Ascending, KeyTable, SortColumn}
import org.testng.annotations.Test

class TrioMatrixSuite extends SparkSuite {

  @Test def test() {
    val ped = Pedigree.read("src/test/resources/tdt.fam", hadoopConf)
    val famkt = KeyTable.importFam(hc, "src/test/resources/tdt.fam")
    val vds = hc.importVCF("src/test/resources/tdt.vcf", nPartitions = Some(4))
      .annotateSamplesTable(famkt, root="sa.fam")

    val trioMatrix = vds.trioMatrix(ped, completeTrios = true)

    val dads = famkt.filter("isDefined(patID)", true)
      .annotate("isDad=true")
      .select("patID", "isDad")
      .keyBy("patID")
    val moms = famkt.filter("isDefined(matID)", true)
      .annotate("isMom=true")
      .select("matID", "isMom")
      .keyBy("matID")

    val gkt = vds
      .genotypeKT()
      .keyBy("s")
      .join(dads, "left")
      .join(moms, "left")
      .annotate("isDad = isDefined(isDad), isMom = isDefined(isMom)")
      .annotate("g = if (isMissing(g)) Genotype(Call(-1)) else g")
      .aggregate("v = v, fam = sa.fam.famID", "data = g.map(g => {role: if (isDad) 1 else if (isMom) 2 else 0, g: g}).collect()")
      .filter("data.length() == 3", keep = true)
      .explode("data")
      .select("v", "fam", "data")

    val tkt = trioMatrix.genotypeKT()
      .annotate("fam = sa.proband.annotations.fam.famID, data = [{role: 0, g: g.proband}, {role: 1, g: g.father}, {role: 2, g: g.mother}]")
      .select("v", "fam", "data")
      .explode("data")
      .filter("isDefined(data.g)", keep = true)
      .keyBy("v", "fam")

    assert(tkt.same(gkt))
  }
}
