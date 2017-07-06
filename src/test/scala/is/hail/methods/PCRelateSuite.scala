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
import is.hail.utils.{TextTableReader, _}
import scala.sys.process._
import is.hail.distributedmatrix.DistributedMatrix
import is.hail.distributedmatrix.DistributedMatrix.implicits._

class PCRelateSuite extends SparkSuite {
  private val blockSize: Int = 8192

  private def toI(a: Any): Int =
    a.asInstanceOf[Int]

  private def toD(a: Any): Double =
    a.asInstanceOf[Double]

  private def toS(a: Any): String =
    a.asInstanceOf[String]

  def runPcRelateHail(vds: VariantDataset, pcs: DenseMatrix): Map[(String, String), (Double, Double, Double, Double)] =
    PCRelate.toPairRdd(vds, pcs, 0.01, blockSize).collect().toMap.asInstanceOf[Map[(String, String), (Double, Double, Double, Double)]]

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

    val columns = Map(
      ("ID1", TString),
      ("ID2", TString),
      ("nsnp", TDouble),
      ("kin", TDouble),
      ("k0", TDouble),
      ("k1", TDouble),
      ("k2", TDouble))
    val separator = " +"

    hadoopConf.copy(localTmpfile + ".out", tmpfile + ".out")

    val (_, rdd) = TextTableReader.read(sc)(Array(tmpfile + ".out"), columns, separator=separator)
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

  def compareDoubleQuartuplets(cmp: (Double, Double) => Boolean)(x: (Double, Double, Double, Double), y: (Double, Double, Double, Double)): Boolean =
    cmp(x._1, y._1) && cmp(x._2, y._2) && cmp(x._3, y._3) && cmp(x._4, y._4)

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

    val hailPcRelate = runPcRelateHail(vds, pcs)

    printToFile(new java.io.File(s"/tmp/compareToPCRelateRSamePCs.out")) { pw =>
      pw.println(Array("s1","s2","uskin","usz0","usz1","usz2","themkin","themz0","themz1","themz2").mkString(","))
      for ((k, (hkin, hz0, hz1, hz2)) <- hailPcRelate) {
        val (rkin, rz0, rz1, rz2) = truth(k)
        val (s1, s2) = k
        pw.println(Array(s1,s2,hkin,hz0,hz1,hz2,rkin,rz0,rz1,rz2).mkString(","))
      }
    }

    assert(mapSameElements(hailPcRelate, truth, compareDoubleQuartuplets((x, y) => math.abs(x - y) < 0.001)))
  }

  @Test def compareToPCRelateR() {
    for {
      n <- Seq(50, 100, 500)
      seed <- Seq(0, 1, 2)
      nVariants <- Seq(1000, 10000// , 50000
      )
    } {
      val vds: VariantDataset = BaldingNicholsModel(hc, 3, n, nVariants, None, None, seed, None, UniformDist(0.1,0.9)).splitMulti()

      val truth = runPcRelateR(vds)

      val pcs = SamplePCA.justScores(vds, 2)
      val hailPcRelate = runPcRelateHail(vds, pcs)

      printToFile(new java.io.File(s"/tmp/compareToPCRelateR-$n-$seed-$nVariants.out")) { pw =>
        pw.println(Array("s1","s2","uskin","usz0","usz1","usz2","themkin","themz0","themz1","themz2").mkString(","))
        for ((k, (hkin, hz0, hz1, hz2)) <- hailPcRelate) {
          val (rkin, rz0, rz1, rz2) = truth(k)
          val (s1, s2) = k
          pw.println(Array(s1,s2,hkin,hz0,hz1,hz2,rkin,rz0,rz1,rz2).mkString(","))
        }
      }

      println(s"$n $seed $nVariants")
      assert(mapSameElements(hailPcRelate, truth, compareDoubleQuartuplets((x, y) => D_==(x, y, tolerance=1e-2))))
    }
  }

  @Test
  def trivial() {
    val genotypeMatrix = new BDM(4,8,Array(0,0,0,0, 0,0,1,0, 0,1,0,1, 0,1,1,1, 1,0,0,0, 1,0,1,0, 1,1,0,1, 1,1,1,1)) // column-major, columns == variants
    val vds = vdsFromGtMatrix(hc)(genotypeMatrix, Some(Array("s1","s2","s3","s4")))
    val pcs = Array(0.0, 1.0, 1.0, 0.0,  1.0, 1.0, 0.0, 0.0) // NB: this **MUST** be the same as the PCs used by the R script
    val us = runPcRelateHail(vds, new DenseMatrix(4,2,pcs))
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

    val r = scala.util.Random

    val profile225 = hc.readVDS("/Users/dking/projects/hail-data/profile225-splitmulti-hardcalls.vds")
    for (fraction <- Seq(0.0625// , 0.125, 0.25, 0.5
    )) {
      val subset = r.shuffle(profile225.sampleIds).slice(0, (profile225.nSamples * fraction).toInt).toSet

      def underStudy(s: String) =
        subset.contains(s) || trios.contains(s) || siblings.contains(s) || secondOrder.contains(s)

      val vds = profile225
        .filterSamples((s, sa) => underStudy(s.asInstanceOf[String]))
        .cache()

      val (truth, pcRelateTime) = time(runPcRelateR(vds))

      val pcs = SamplePCA.justScores(vds.coalesce(10), 2)
      val (hailPcRelate, hailTime) = time(runPcRelateHail(vds, pcs))

      println(s"on fraction: $fraction; pc relate: $pcRelateTime, hail: $hailTime, ratio: ${pcRelateTime / hailTime.toDouble}")

      printToFile(new java.io.File(s"/tmp/thousandGenomesTrios-$fraction.out")) { pw =>
        pw.println(Array("s1","s2","uskin","usz0","usz1","usz2","themkin","themz0","themz1","themz2").mkString(","))
        for ((k, (hkin, hz0, hz1, hz2)) <- hailPcRelate) {
          val (rkin, rz0, rz1, rz2) = truth(k)
          val (s1, s2) = k
          pw.println(Array(s1,s2,hkin,hz0,hz1,hz2,rkin,rz0,rz1,rz2).mkString(","))
        }
      }

      assert(mapSameElements(hailPcRelate, truth, compareDoubleQuartuplets((x, y) => math.abs(x - y) < 0.01)))
    }
  }

  @Test
  def thousandGenomesTriosMAFOneHundredth() {
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

    val r = scala.util.Random

    val profile225 = hc.readVDS("/Users/dking/projects/hail-data/profile225-splitmulti-hardcalls.vds")
    for (fraction <- Seq(// 0.0625,
      0.125//, 0.25, 0.5
    )) {
      val subset = r.shuffle(profile225.sampleIds).slice(0, (profile225.nSamples * fraction).toInt).toSet

      def underStudy(s: String) =
        subset.contains(s) || trios.contains(s) || siblings.contains(s) || secondOrder.contains(s)

      val vds = profile225
        .filterSamples((s, sa) => underStudy(s.asInstanceOf[String]))
        .cache()

      val (truth, pcRelateTime) = time(runPcRelateR(vds, "src/test/resources/is/hail/methods/runPcRelateMAF0.01.R"))

      val pcs = SamplePCA.justScores(vds.coalesce(10), 2)
      val (hailPcRelate, hailTime) = time(runPcRelateHail(vds, pcs))

      println(s"on fraction: $fraction; pc relate: $pcRelateTime, hail: $hailTime, ratio: ${pcRelateTime / hailTime.toDouble}")

      printToFile(new java.io.File(s"/tmp/thousandGenomesTriosMAFOneHundredth-$fraction.out")) { pw =>
        pw.println(Array("s1","s2","uskin","usz0","usz1","usz2","themkin","themz0","themz1","themz2").mkString(","))
        for ((k, (hkin, hz0, hz1, hz2)) <- hailPcRelate) {
          val (rkin, rz0, rz1, rz2) = truth(k)
          val (s1, s2) = k
          pw.println(Array(s1,s2,hkin,hz0,hz1,hz2,rkin,rz0,rz1,rz2).mkString(","))
        }
      }

      assert(mapSameElements(hailPcRelate, truth, compareDoubleQuartuplets((x, y) => math.abs(x - y) < 0.01)))
    }
  }

  private val thousandGenomesRandomSubsetWithTrios = Array(
    "HG00105","HG00115","HG00121","HG00141","HG00187","HG00235","HG00245",
    "HG00254","HG00274","HG00276","HG00311","HG00339","HG00362","HG00367",
    "HG00524","HG00608","HG00635","HG00656","HG00657","HG00684","HG00693",
    "HG00702","HG00731","HG00732","HG00733","HG01066","HG01082","HG01086",
    "HG01088","HG01108","HG01119","HG01134","HG01140","HG01197","HG01205",
    "HG01269","HG01280","HG01305","HG01383","HG01389","HG01438","HG01512",
    "HG01528","HG01566","HG01578","HG01597","HG01613","HG01673","HG01676",
    "HG01686","HG01699","HG01761","HG01770","HG01771","HG01802","HG01846",
    "HG01926","HG01935","HG01936","HG01983","HG02024","HG02025","HG02026",
    "HG02052","HG02107","HG02111","HG02150","HG02178","HG02299","HG02322",
    "HG02396","HG02399","HG02502","HG02611","HG02661","HG02675","HG02688",
    "HG02700","HG02722","HG02885","HG02947","HG02974","HG03006","HG03054",
    "HG03105","HG03114","HG03124","HG03228","HG03268","HG03442","HG03499",
    "HG03557","HG03660","HG03673","HG03686","HG03705","HG03713","HG03715",
    "HG03722","HG03796","HG03802","HG03808","HG03833","HG03856","HG03872",
    "HG03887","HG03898","HG03905","HG03913","HG03919","HG03943","HG03947",
    "HG03948","HG03950","HG04060","HG04107","HG04152","HG04155","HG04182",
    "HG04211","NA07037","NA12234","NA12275","NA12383","NA12400","NA12546",
    "NA12812","NA12878","NA18572","NA18574","NA18593","NA18612","NA18641",
    "NA18876","NA18917","NA18950","NA18953","NA18963","NA18998","NA19001",
    "NA19027","NA19063","NA19077","NA19098","NA19130","NA19238","NA19239",
    "NA19240","NA19312","NA19317","NA19320","NA19395","NA19404","NA19434",
    "NA19457","NA19660","NA19661","NA19670","NA19675","NA19678","NA19679",
    "NA19682","NA19685","NA19713","NA19726","NA19735","NA19758","NA19783",
    "NA19985","NA20289","NA20294","NA20321","NA20334","NA20336","NA20341",
    "NA20359","NA20506","NA20521","NA20536","NA20792","NA20796","NA20802",
    "NA20899","NA20905","NA21095","NA21103","NA21108")
  private val thousandGenomesRandomSubsetPCsFromPCAiR = Array(
    -0.03317798439611171352, -0.105555396598680364950,
    -0.03080117752292399480, -0.097846789579283943716,
    -0.03725598488296377742, -0.111406429271962620353,
    -0.04336086710633513253, -0.096501516920563856772,
    -0.03194246555669463805, -0.081714390338039538664,
    -0.03425299210214009099, -0.082669013713494096662,
    -0.03704861951678539017, -0.064563846435985020045,
    -0.03731631429519954857, -0.099449798125073624044,
    -0.03601283320176438402, -0.056908801796773655912,
    -0.03731954772095325001, -0.043382123437417180467,
    -0.03115834600088445566, -0.100944566286071193972,
    -0.02608133071111052095, -0.070770567732990022347,
    -0.03981870744186469246, -0.066088069534235918678,
    -0.04031637438391501987, -0.057248231951120755190,
    -0.03940852501291541854,  0.101764767675265738189,
    -0.05589013182003956665,  0.112909386197641167793,
    -0.04007200382765920693,  0.116312727158466022725,
    -0.05654752338457936373,  0.147875572950229378089,
    -0.05404237256521750299,  0.120433614619182768890,
    -0.04480801786495598044,  0.138105166523855793503,
    -0.05340265343359396760,  0.115431267896936542994,
    -0.05738428707643981191,  0.138508378077973309805,
    -0.03412364828253028337, -0.118244127294737488842,
    -0.01917104794089595152, -0.107415146122738794365,
    -0.03447576478007832040, -0.110709644125911901846,
    -0.03398661952895949689, -0.011049559483947096403,
    -0.03856304970213631639, -0.006951564193329783960,
    -0.04121216290188788739, -0.052030424397396718428,
    0.03558432267322252096, -0.050310686724037462225,
    0.10325554208261686351, -0.015242096551448319416,
    -0.04593727352972980932,  0.004775543942115089216,
    -0.02948151537658865590, -0.079883919789846630222,
    -0.04390989053489589844,  0.014900185540904690648,
    -0.00618022337607682538, -0.010207201205800370616,
    -0.03919207130104507275, -0.065483852563926164825,
    -0.04130975652582905000, -0.070249561929351764245,
    -0.02406698414751282100, -0.077633963193464286534,
    0.04444385332535297217, -0.049974301239173360423 ,
    -0.03118908923194069552, -0.089452190187519972642,
    -0.04565223810540294802, -0.076519587653066559185,
    -0.03889865563416623762, -0.098126486113259914212,
    -0.03714905574227545665, -0.117016558471323750545,
    -0.03128650940461981389, -0.105989084859225562996,
    -0.04100832875300360753, -0.002360698344370199941,
    -0.05236543708481428139,  0.043984354291656944935,
    -0.05188622457460419918,  0.119790254858259351267,
    -0.03346708851196443513, -0.098331751101125824954,
    -0.03906628212976193115, -0.067266445980811412353,
    -0.03685763648779460250, -0.109949868102825559779,
    -0.02561179403427158580, -0.102952313436241046918,
    -0.03242290747005726476, -0.099194140452130596941,
    -0.03814438842879824820, -0.104951936510326818142,
    -0.02637752154152160980, -0.111337912135570399341,
    -0.03349488044716582291, -0.112223401321231583405,
    -0.05162968953038993281,  0.137434412152287882547,
    -0.04495682707344709345,  0.105843301053545085777,
    -0.05598368745393030033,  0.104647027888518637240,
    -0.05110027753077329288,  0.046492825966126791193,
    -0.05101414637115221640,  0.071003813884628994013,
    -0.04252875219567014975, -0.001680345053592206383,
    -0.04866397789648729127,  0.137846799009065851260,
    -0.05326224180051830626,  0.110354657533587793838,
    -0.04670890185721084803,  0.115711675298404662038,
    0.14628083001753081072,  0.012321445183849300681 ,
    0.14406666872027978910,  0.014666170581387352398 ,
    0.15574085999791881241,  0.020211482263352831962 ,
    -0.05613843904267877549,  0.109696573505964470319,
    -0.05051285424478092756,  0.135664354931773672996,
    -0.05770638605407175936,  0.095608117562905436948,
    0.13270579655671971286, -0.002641476403951568182 ,
    -0.04691670238986189040,  0.089749755177078949231,
    -0.04552216755066526566,  0.092075878822196893791,
    0.04994386981415946392, -0.019852477500011022488 ,
    0.14883651862104399499,  0.017239521404928775244 ,
    -0.03978090366723408683, -0.008434728957657698645,
    0.13035985442959580061,  0.016063047714511856368 ,
    -0.03389221229355422749,  0.008984568481202999390,
    -0.03900537634890479527, -0.021376898584395207653,
    0.12827079559608858816,  0.003503263437018402753 ,
    0.13683794486999728557, -0.007883249569306232524 ,
    0.13147993540749144947, -0.000704918801531328847 ,
    0.16159085790749078138,  0.038412399847593391156 ,
    -0.04029105544385475079,  0.037135330377591535544,
    0.13302765375089220523,  0.019390302794140966414 ,
    0.14376019546845916763,  0.014911314266544737836 ,
    0.14654573551693067524,  0.015161364086542328949 ,
    0.15622258660621390902,  0.029759435209929172017 ,
    -0.03326243530349629746, -0.039836794334065596390,
    0.15404141856982597747,  0.048223763215192672060 ,
    0.14267663298870336686,  0.035353532684051773571 ,
    0.14965313051256182164,  0.031103339698374012079 ,
    0.13605320018664857074,  0.015596004914904397151 ,
    -0.03657457319921058464, -0.013551560446187984818,
    -0.03861500940696536294, -0.006402088512355092603,
    -0.04216906610378642895,  0.006473531259519917959,
    -0.03245347140927025420, -0.024008911025431164954,
    -0.04142967800397798911, -0.018140861240426488926,
    -0.04478605704548251770, -0.028571793335912750583,
    -0.03798300033468106351,  0.020625991616814561003,
    -0.03884983239818759482,  0.018418380016451785935,
    -0.04434288049929890896,  0.011256761413419821491,
    -0.03362926861332263262,  0.017350435760939475482,
    -0.03730896025493967566, -0.001875811113562028936,
    -0.03302846845268472831,  0.013076685200586621316,
    -0.03375487727777071179,  0.003456714283704677860,
    -0.04231748403305890527,  0.011557681543361548668,
    -0.03873960960083156629,  0.002121507475846348342,
    -0.03776060022912799269, -0.021148406623229941415,
    -0.03871991675192145571,  0.020711142589573922762,
    -0.04145542828644558503,  0.021737834271289582094,
    -0.03441683297161179933, -0.001388920887335896685,
    -0.03695908937924166637,  0.007127890552780168434,
    -0.04471339999115364616, -0.019219608553290183639,
    -0.03795470019039680343,  0.016762361708746016664,
    -0.03500283258327113733, -0.011119690899445934182,
    -0.03888520258997132939, -0.021308992160638324381,
    -0.03819366395439378198,  0.016320237181094099377,
    -0.04084800920923503664, -0.002282200935669621063,
    -0.02744321451002680601, -0.000963729611466182566,
    -0.04044648099685402287,  0.016545535935458356713,
    -0.03473326145074483867, -0.096125481834183443275,
    -0.03362078880993738667, -0.113001274319375233368,
    -0.02816982780319135862, -0.104036822992776320507,
    -0.03459491526972295616, -0.102584552560472150318,
    -0.02138696712332800384, -0.115656904025562667915,
    -0.03247561656044348211, -0.081990457971833907713,
    -0.03957533492031512640, -0.087348870090116909060,
    -0.03526389477240195192, -0.071673088385067473816,
    -0.05209101726675832017,  0.140017858198559469463,
    -0.04610588184957477992,  0.133175493077203888070,
    -0.04847614309046457265,  0.138366886621968693261,
    -0.04809613285195883658,  0.124835092110043816005,
    -0.04718662512537132037,  0.142654421179346957427,
    0.15383720235450870639,  0.015571917722559946995 ,
    0.14466591603244680075,  0.013669275585229801948 ,
    -0.04675347233739906433,  0.112496932214516717763,
    -0.04740821585316967041,  0.114764697216143116143,
    -0.05219521811283932278,  0.111435902209631523818,
    -0.04803187219559690302,  0.096980458347710907230,
    -0.05102900750536420310,  0.132124201626309029090,
    0.12063218618718811459,  0.014882299223833177562 ,
    -0.05106121099986903750,  0.138146713354359323978,
    -0.05116135307913183272,  0.139809957825172248569,
    0.13243622245868921783, -0.006769572627901988826 ,
    0.15153706732938709845,  0.001004941865452415976 ,
    0.16242448170783943540,  0.023215749757208005433 ,
    0.16488743076695927536,  0.031782213587313021264 ,
    0.17432919224852655438,  0.026073561277802366487 ,
    0.12215575054701499624,  0.005434868722065559071 ,
    0.14205576700955777070,  0.030450505593628850176 ,
    0.14112923863674123326,  0.033279975026850518172 ,
    0.11918066017918772792, -0.000110937789359087478 ,
    0.14860411508263962621,  0.001733137312063831635 ,
    0.13110238434451643164,  0.020310739487029556838 ,
    0.13332909746511170668,  0.007604789041194869384 ,
    -0.02684901369448060293,  0.015664727367340265662,
    -0.04154262350792465447,  0.021599280417842214169,
    0.02268938617109600778,  0.023040636428075821074 ,
    -0.04642459954106967907, -0.019460070521016278183,
    -0.03753922242219860650, -0.074310792651304496959,
    -0.04111325243736385637, -0.003183631837859704494,
    -0.04645393459893098348,  0.007957251076164477757,
    -0.02565198769933142373,  0.050054224657258547526,
    0.08083192189671614680, -0.043057158790635978451 ,
    -0.05386543065425428156,  0.087116402162066688963,
    -0.04464342264505467839,  0.025721465525352361220,
    -0.05215039236537449829,  0.084904887946228121964,
    -0.05398206736795322508,  0.117521883210140537512,
    0.05235032554904170410, -0.064972076622229826826 ,
    0.04980845329303752039, -0.055580906217341588538 ,
    0.11889388542327836151,  0.001064617117959934479 ,
    0.12490606242291030226,  0.026524423522918724044 ,
    -0.02284907104603241257, -0.074059958826597799275,
    -0.02611218363484872976, -0.087918633760454487081,
    0.06525899736759295522, -0.050123450800467331256 ,
    0.15043734469693076372,  0.012248048938683944037 ,
    -0.04063969654347139582, -0.109193922850735564145,
    -0.02131789847010879049, -0.095902483606610605671,
    -0.03008761448204318179, -0.069761456943378180595,
    -0.02846132170549865378, -0.072898040908375524860,
    -0.02500595483665153315, -0.096000318653660446366,
    -0.03589825725622086683, -0.084685600748782244307,
    -0.02613570143107982885,  0.012277753222865409044,
    -0.03898501025896320554, -0.017867590032417966783,
    -0.04376125968606946193,  0.008801775058467572327,
    -0.03739114502663110073,  0.018504234152740557662,
    -0.04101772684035039779, -0.044548102352768403911)
  @Test
  def thousandGenomesSubsetSamePCsTest() {
    val subset = thousandGenomesRandomSubsetWithTrios.toSet
    def underStudy(s: String) =
      subset.contains(s)

    val vds = hc.readVDS("/Users/dking/projects/hail-data/profile225-splitmulti-hardcalls.vds")
      .filterSamples((s, sa) => underStudy(s.asInstanceOf[String]))
      .cache()

    val (truth, pcRelateTime) = time(runPcRelateR(vds))

    val pcs = new DenseMatrix(thousandGenomesRandomSubsetWithTrios.length, 2, thousandGenomesRandomSubsetPCsFromPCAiR)
    val (hailPcRelate, hailTime) = time(runPcRelateHail(vds, pcs))

    println(s"pc relate: $pcRelateTime, hail: $hailTime, ratio: ${pcRelateTime / hailTime.toDouble}")

    printToFile(new java.io.File("/tmp/thousandGenomesSubsetSamePCsTest.out")) { pw =>
      pw.println(Array("s1","s2","uskin","usz0","usz1","usz2","themkin","themz0","themz1","themz2").mkString(","))
      for ((k, (hkin, hz0, hz1, hz2)) <- hailPcRelate) {
        val (rkin, rz0, rz1, rz2) = truth(k)
        val (s1, s2) = k
        pw.println(Array(s1,s2,hkin,hz0,hz1,hz2,rkin,rz0,rz1,rz2).mkString(","))
      }
    }

    assert(mapSameElements(hailPcRelate, truth, compareDoubleQuartuplets((x, y) => math.abs(x - y) < 0.01)))
  }

  @Test
  def thousandGenomesSubsetSamePCsMAFOneHundredthTest() {
    val subset = thousandGenomesRandomSubsetWithTrios.toSet
    def underStudy(s: String) =
      subset.contains(s)

    val vds = hc.readVDS("/Users/dking/projects/hail-data/profile225-splitmulti-hardcalls.vds")
      .filterSamples((s, sa) => underStudy(s.asInstanceOf[String]))
      .cache()

    val (truth, pcRelateTime) = time(runPcRelateR(vds, "src/test/resources/is/hail/methods/runPcRelateMAF0.01.R"))

    val pcs = new DenseMatrix(thousandGenomesRandomSubsetWithTrios.length, 2, thousandGenomesRandomSubsetPCsFromPCAiR)
    val (hailPcRelate, hailTime) = time(runPcRelateHail(vds, pcs))

    println(s"pc relate: $pcRelateTime, hail: $hailTime, ratio: ${pcRelateTime / hailTime.toDouble}")

    printToFile(new java.io.File("/tmp/thousandGenomesSubsetSamePCsMAFOneHundredthTest.out")) { pw =>
      pw.println(Array("s1","s2","uskin","usz0","usz1","usz2","themkin","themz0","themz1","themz2").mkString(","))
      for ((k, (hkin, hz0, hz1, hz2)) <- hailPcRelate) {
        val (rkin, rz0, rz1, rz2) = truth(k)
        val (s1, s2) = k
        pw.println(Array(s1,s2,hkin,hz0,hz1,hz2,rkin,rz0,rz1,rz2).mkString(","))
      }
    }

    assert(mapSameElements(hailPcRelate, truth, compareDoubleQuartuplets((x, y) => math.abs(x - y) < 0.01)))
  }

  @Test
  def thousandGenomesTrios10PCs() {
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

    val r = scala.util.Random

    val profile225 = hc.readVDS("/Users/dking/projects/hail-data/profile225-splitmulti-hardcalls.vds")
    for (fraction <- Seq(0.0625// , 0.125, 0.25, 0.5
    )) {
      val subset = r.shuffle(profile225.sampleIds).slice(0, (profile225.nSamples * fraction).toInt).toSet

      def underStudy(s: String) =
        subset.contains(s) || trios.contains(s) || siblings.contains(s) || secondOrder.contains(s)

      val vds = profile225
        .filterSamples((s, sa) => underStudy(s.asInstanceOf[String]))
        .cache()

      val (truth, pcRelateTime) = time(runPcRelateR(vds, "src/test/resources/is/hail/methods/runPcRelate10PCs.R"))

      val pcs = SamplePCA.justScores(vds.coalesce(10), 10)
      val (hailPcRelate, hailTime) = time(runPcRelateHail(vds, pcs))

      println(s"on fraction: $fraction; pc relate: $pcRelateTime, hail: $hailTime, ratio: ${pcRelateTime / hailTime.toDouble}")

      printToFile(new java.io.File("/tmp/thousandGenomesTrios10PCs.out")) { pw =>
        pw.println(Array("s1","s2","uskin","usz0","usz1","usz2","themkin","themz0","themz1","themz2").mkString(","))
        for ((k, (hkin, hz0, hz1, hz2)) <- hailPcRelate) {
          val (rkin, rz0, rz1, rz2) = truth(k)
          val (s1, s2) = k
          pw.println(Array(s1,s2,hkin,hz0,hz1,hz2,rkin,rz0,rz1,rz2).mkString(","))
        }
      }

      assert(mapSameElements(hailPcRelate, truth, compareDoubleQuartuplets((x, y) => math.abs(x - y) < 0.01)))
    }
  }

}
