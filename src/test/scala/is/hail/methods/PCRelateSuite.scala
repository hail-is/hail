package is.hail.methods

import java.io.InputStream
import java.io.OutputStream
import java.nio.file.Files
import java.nio.file.Paths

import breeze.linalg.{DenseMatrix => BDM}
import is.hail.keytable._
import is.hail.annotations.Annotation
import is.hail.expr.{TStruct, _}
import org.apache.spark.mllib.linalg.distributed._
import is.hail.SparkSuite
import org.apache.hadoop
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.Row
import org.apache.spark.mllib.linalg._
import org.testng.annotations.Test
import is.hail.check._
import is.hail.check.Prop._
import is.hail.expr.{TDouble, TInt, TString}
import is.hail.variant.VariantDataset
import is.hail.variant.VSMSubgen
import is.hail.stats._
import is.hail.utils.{TextTableConfiguration, TextTableReader, _}
import scala.sys.process._
import is.hail.distributedmatrix.DistributedMatrix
import is.hail.distributedmatrix.DistributedMatrix.implicits._

class PCRelateSuite extends SparkSuite {
  private def toI(a: Any): Int =
    a.asInstanceOf[Int]

  private def toD(a: Any): Double =
    a.asInstanceOf[Double]

  private def toS(a: Any): String =
    a.asInstanceOf[String]

  def runPcRelateR(
    vds: VariantDataset,
    rFile: String = "src/test/resources/is/hail/methods/runPcRelate.R"): Map[(String, String), (Double, Double, Double, Double)] = {

    val tmpfile = tmpDir.createTempFile(prefix = "pcrelate")
    val localTmpfile = tmpDir.createLocalTempFile(prefix = "pcrelate")
    val pcRelateScript = tmpDir.createLocalTempFile(prefix = "pcrelateScript")

    vds.exportPlink(tmpfile)

    for (suffix <- Seq(".bed", ".bim", ".fam")) {
      hadoopConf.copy(tmpfile + suffix, localTmpfile + suffix)
    }

    s"Rscript $rFile ${uriPath(localTmpfile)}" !

    val genomeFormat = TextTableConfiguration(
      types = Map(
        ("ID1", TString), ("ID2", TString), ("nsnp", TDouble), ("kin", TDouble), ("k0", TDouble), ("k1", TDouble), ("k2", TDouble)),
      separator = " +")

    hadoopConf.copy(localTmpfile + ".out", tmpfile + ".out")

    val (_, rdd) = TextTableReader.read(sc)(Array(tmpfile + ".out"), genomeFormat)
    rdd.collect()
      .map(_.value)
      .map { ann =>
      val row = ann.asInstanceOf[Row]
      val id1 = toS(row(0))
      val id2 = toS(row(1))
      val nsnp = toD(row(2)).toInt
      val kin = toD(row(3))
      val k0 = toD(row(4))
      val k1 = toD(row(5))
      val k2 = toD(row(6))
      ((id1, id2), (kin, k0, k1, k2))
    }
      .toMap
  }

  def runPcRelateToPairRDD(vds: VariantDataset, pcs: DenseMatrix): RDD[((String, String), (Double, Double, Double, Double))] = {
    val indexToId: Map[Int, String] = vds.sampleIds.zipWithIndex.map { case (id, index) => (index, id) }.toMap
    def upperTriangularEntires(bm: BlockMatrix): RDD[((String, String), Double)] =
      DistributedMatrix[BlockMatrix].toCoordinateMatrix(bm).entries
        .filter(me => me.i < me.j)
        .map(me => ((indexToId(me.i.toInt), indexToId(me.j.toInt)), me.value))

    val result = PCRelate.maybefast[BlockMatrix](vds, pcs)

    (upperTriangularEntires(result.phiHat) join
      upperTriangularEntires(result.k0) join
      upperTriangularEntires(result.k1) join
      upperTriangularEntires(result.k2))
      .mapValues { case (((kin, k0), k1), k2) => (kin, k0, k1, k2) }
  }

  def compareDoubleQuartuplets(cmp: (Double, Double) => Boolean)(x: (Double, Double, Double, Double), y: (Double, Double, Double, Double)): Boolean =
    cmp(x._1, y._1) && cmp(x._2, y._2) && cmp(x._3, y._3) && cmp(x._4, y._4)
    // if (!cmp(x._1, y._1)) {
    //   println(s"${x._1} and ${y._1} failed")
    //   false
    // } else if (!cmp(x._2, y._2)) {
    //   println(s"${x._2} and ${y._2} failed")
    //   false
    // } else if (!cmp(x._3, y._3)) {
    //   println(s"${x._3} and ${y._3} failed")
    //   false
    // } else if (!cmp(x._4, y._4)) {
    //   println(s"${x._4} and ${y._4} failed")
    //   false
    // } else {
    //   true
    // }

  @Test def compareToPCRelateRSamePCs() {
    // from pcrelate
    val pcs = new DenseMatrix(100, 2, Array(
      0.1189551839914291670, -0.0446579450500885072,
      -0.0233052745350766921,  0.1655183591618567540,
      0.1227206818296405350, -0.0381327963819160162,
      -0.1114140928493930760, -0.0768953358882566163,
      0.1197780922247055901, -0.0470817983169786597,
      -0.0198817575994237467,  0.1635326030309057210,
      -0.1162372593841667218, -0.0794496644100964183,
      0.1172587486130869799, -0.0417964376076797317,
      0.1134213385976453603, -0.0402400814705450860,
      -0.1142558717182596828, -0.0685709475241189359,
      -0.1115445930364524774, -0.0818498654991581254,
      -0.0226155839176865880,  0.1664031566187116062,
      0.1219548866899461542, -0.0341119201985343179,
      -0.0241632896623880100,  0.1694860430252811967,
      0.1175541926837137818, -0.0382956755917953223,
      -0.1161632502354573299, -0.0758337803128565358,
      -0.1147499238317806203, -0.0752743453842679988,
      -0.0186055990831107999,  0.1625832327497430674,
      -0.1174901963112555403, -0.0725625533253248661,
      -0.0187434656815751166,  0.1682595945456624442,
      -0.0251471549262226313,  0.1600199494228471964,
      0.1166682269017132734, -0.0474564778124868802,
      0.1184376647641543628, -0.0379791614949554490,
      0.1163938189248003019, -0.0464681294390897545,
      -0.0212692866633024946,  0.1634807521925577267,
      -0.1150241733832624014, -0.0777686765294046678,
      -0.0241928458283760250,  0.1678727704800818676,
      0.1136793605400302776, -0.0450479286937650741,
      -0.1101631372251713203, -0.0731307884006531517,
      -0.1074351187942836072, -0.0689625561211417021,
      0.1192272172144396569, -0.0431190821232532995,
      -0.0267344861123452043,  0.1659282450952285493,
      0.1175461998995781521, -0.0419058265705682109,
      -0.0261530561670911960,  0.1686928591117949072,
      -0.1099480446858744631, -0.0693399768810517153,
      0.1127826951822281792, -0.0362408732840091841,
      -0.0296458975769711558,  0.1650027341961477489,
      -0.0254164616041451102,  0.1709443410100764948,
      0.1160238029648566205, -0.0408185786223487918,
      -0.1104650911739857022, -0.0745394380540490148,
      -0.1159566516448892748, -0.0782050203668911981,
      -0.1178008993815877620, -0.0792510864603673537,
      0.1142387624014616493, -0.0443139202008340921,
      -0.1127523644289251531, -0.0698853779973312716,
      -0.1149981240006467814, -0.0764166872985991563,
      0.1199852815405234169, -0.0387997618399208294,
      0.1191421160182072059, -0.0497965058305220171,
      0.1087789409773645827, -0.0491200553944184520,
      -0.1111443752215766356, -0.0766152367204323281,
      0.1144680255791643980, -0.0454814521888796536,
      -0.1142696747400882284, -0.0772667143143159207,
      -0.1113589321365357210, -0.0758389396418310269,
      -0.1068505788839549386, -0.0831644708150781481,
      0.1157583816817154276, -0.0460137054733688530,
      0.1158366493696507266, -0.0477571520872011046,
      -0.0242339636739650338,  0.1752804671931756375,
      -0.1113732136857790156, -0.0791651676666857568,
      0.1190463076837385975, -0.0418104449714543150,
      -0.1111814270224098650, -0.0771255693045958118,
      -0.1150288214789111640, -0.0844055470484871007,
      -0.0296707935316978533,  0.1694172585100202633,
      0.1182077693759877035, -0.0343150402808204552,
      0.1168995875709475246, -0.0444353139891397603,
      0.1218107282820171944, -0.0432793056763338541,
      0.1145749565474074166, -0.0465151021603511308,
      0.1111377496563750178, -0.0506989017981283735,
      -0.1144793387143286101, -0.0709662902979083104,
      -0.1111651570416238577, -0.0861414540654116406,
      -0.0240115661725185392,  0.1675263633652575301,
      -0.0227792035178657135,  0.1684865238380965635,
      -0.1103098665460854194, -0.0759353187541915858,
      -0.0193891020719692067,  0.1652235994281077980,
      -0.1178210574377045461, -0.0730276536324879627,
      -0.1098074126734925188, -0.0722308778481396824,
      0.1200845829565512501, -0.0492148932680887760,
      -0.1159777884118543267, -0.0758819955347604935,
      -0.1122850679087848413, -0.0731249472556277785,
      0.1213258470246274950, -0.0444520252567109919,
      -0.0189435610940520485,  0.1723454814976673743,
      0.1197838567814633604, -0.0379161695801484774,
      -0.0294815250887478369,  0.1608117539767592585,
      -0.0200045329980862441,  0.1680390155986327960,
      -0.0209590699956217832,  0.1637249429792022593,
      -0.0202594298295039749,  0.1704706557797990019,
      0.1223999477691687932, -0.0424608555207330457,
      -0.1170977166947284598, -0.0706560747433918751,
      0.1093764984164675158, -0.0393716174120272017,
      0.1152977517312178957, -0.0477595140520716416,
      -0.1166354183097513819, -0.0770661425189250460,
      -0.1079335421610403828, -0.0802988058258599646,
      -0.1140804266003039719, -0.0776692269568053095,
      -0.0234034362789994205,  0.1684175733378067419,
      0.1168201347305513615, -0.0428401672217596577,
      0.1163875574075797364, -0.0461734301793094437,
      0.1236190441396658318, -0.0394292614475165726,
      0.1189383526946554415, -0.0431569067337288503,
      -0.0295736101796383946,  0.1633584108804886637,
      -0.0263682350446129421,  0.1685979255727877990,
      -0.1140458052120602722, -0.0691394464312551360,
      0.1178756604429364724, -0.0473044175474795422
    ), true)

    val vds: VariantDataset = BaldingNicholsModel(hc, 3, 100, 10000, None, None, 0, None, UniformDist(0.4,0.6)).splitMulti()

    val truth = runPcRelateR(vds)

    val hailPcRelate = runPcRelateToPairRDD(vds, pcs)
      .collect()
      .toMap

    assert(mapSameElements(hailPcRelate, truth, compareDoubleQuartuplets((x, y) => math.abs(x - y) < 0.01)))
  }

  @Test def compareToPCRelateR() {
    for {
      n <- Seq(50, 100, 500, 1000)
      seed <- Seq(0, 1, 2)
      nVariants <- Seq(1000, 10000, 50000, 100000)
    } {
      val vds: VariantDataset = BaldingNicholsModel(hc, 3, n, nVariants, None, None, seed, None, UniformDist(0.1,0.9)).splitMulti()

      val truth = runPcRelateR(vds)

      val pcs = SamplePCA.justScores(vds, 2)
      val hailPcRelate = runPcRelateToPairRDD(vds, pcs)
        .collect()
        .toMap

      assert(mapSameElements(hailPcRelate, truth, compareDoubleQuartuplets((x, y) => math.abs(x - y) < 0.01)))
    }
  }

  @Test
  def trivial() {
    val genotypeMatrix = new BDM(4,8,Array(0,0,0,0, 0,0,1,0, 0,1,0,1, 0,1,1,1, 1,0,0,0, 1,0,1,0, 1,1,0,1, 1,1,1,1)) // column-major, columns == variants
    val vds = vdsFromMatrix(hc)(genotypeMatrix, Some(Array("s1","s2","s3","s4")))
    val pcs = Array(0.0, 1.0, 1.0, 0.0,  1.0, 1.0, 0.0, 0.0) // NB: this **MUST** be the same as the PCs used by the R script
    val us = runPcRelateToPairRDD(vds, new DenseMatrix(4,2,pcs))
      .collect()
      .toMap
    println(us)
    val truth = runPcRelateR(vds, "src/test/resources/is/hail/methods/runPcRelateOnTrivialExample.R")
    assert(mapSameElements(us, truth, compareDoubleQuartuplets((x, y) => math.abs(x - y) < 0.01)))
  }

  @Test
  def thousandGenomesTrios() {
    val trios = Array("HG00702", "HG00656", "HG00657",
      "HG00733", "HG00731", "HG00732",
      "HG02024", "HG02026", "HG02025",
      "HG03715","HG03713",
      "HG03948","HG03673",
      "NA19240", "NA19239", "NA19238",
      "NA19675", "NA19679", "NA19678",
      "NA19685", "NA19661", "NA19660")

    val siblings = Array("NA19713", "NA19985",
      "NA20289", "NA20341",
      "NA20334", "NA20336")

    val secondOrder = Array("HG01936", "HG01983")

    def underStudy(s: String) =
      trios.contains(s) || siblings.contains(s) || secondOrder.contains(s)

    val r = scala.util.Random

    val profile225 = hc.read("/Users/dking/projects/hail-data/profile225-splitmulti-hardcalls.vds")
    for (fraction <- Seq(0.0625// , 0.125, 0.25, 0.5
    )) {
      val vds = profile225
        .filterSamples((s, sa) => underStudy(s) || (r.nextDouble() < fraction))
        .cache()

      val (truth, pcRelateTime) = time(runPcRelateR(vds))

      val pcs = SamplePCA.justScores(vds, 2)
      val (hailPcRelate, hailTime) = time(runPcRelateToPairRDD(vds, pcs).collect().toMap)

      println(s"on fraction: $fraction; pc relate: $pcRelateTime, hail: $hailTime, ratio: ${pcRelateTime / hailTime.toDouble}")

      assert(mapSameElements(hailPcRelate, truth, compareDoubleQuartuplets((x, y) => math.abs(x - y) < 0.01)))
    }
  }
}
