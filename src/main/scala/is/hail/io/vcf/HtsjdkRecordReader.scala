package is.hail.io.vcf

import java.util

import htsjdk.variant.variantcontext.VariantContext
import htsjdk.variant.vcf.VCFConstants
import is.hail.annotations._
import is.hail.expr._
import is.hail.expr.types._
import is.hail.utils._
import is.hail.variant._

class BufferedLineIterator(bit: BufferedIterator[String]) extends htsjdk.tribble.readers.LineIterator {
  override def peek(): String = bit.head

  override def hasNext: Boolean = bit.hasNext

  override def next(): String = bit.next()

  override def remove() {
    throw new UnsupportedOperationException
  }
}

class HtsjdkRecordReader(val callFields: Set[String]) extends Serializable {

  import HtsjdkRecordReader._

  def readVariantInfo(vc: VariantContext, rvb: RegionValueBuilder, infoType: TStruct) {
    // locus, alleles added via VCFLine

    // rsid
    val vcID = vc.getID
    val rsid = if (vcID == ".")
      rvb.setMissing()
    else
      rvb.addString(vcID)

    rvb.addDouble(vc.getPhredScaledQual)

    // filters
    if (!vc.filtersWereApplied)
      rvb.setMissing()
    else {
      if (vc.isNotFiltered) {
        rvb.startArray(0)
        rvb.endArray()
      } else {
        rvb.startArray(vc.getFilters.size())
        var fi = vc.getFilters.iterator()
        while (fi.hasNext)
          rvb.addString(fi.next())
        rvb.endArray()
      }
    }

    // info
    rvb.startStruct()
    infoType.fields.foreach { f =>
      val a = vc.getAttribute(f.name)
      addAttribute(rvb, a, f.typ, -1)
    }
    rvb.endStruct() // info
  }

  def readRecord(vc: VariantContext, rvb: RegionValueBuilder, infoType: TStruct, gType: TStruct, dropSamples: Boolean, canonicalFlags: Int): Unit = {
    readVariantInfo(vc, rvb, infoType)

    if (dropSamples) {
      rvb.startArray(0) // gs
      rvb.endArray()
      return
    }

    val nAlleles = vc.getNAlleles
    val nGenotypes = Variant.nGenotypes(nAlleles)
    val haploidPL = new Array[Int](nGenotypes)

    val nCanonicalFields = Integer.bitCount(canonicalFlags)

    rvb.startArray(vc.getNSamples) // gs
    val it = vc.getGenotypes.iterator
    while (it.hasNext) {
      val g = it.next()

      val alleles = g.getAlleles
      assert(alleles.size() == 1 || alleles.size() == 2,
        s"expected 1 or 2 alleles in genotype, but found ${ alleles.size() }")

      rvb.startStruct() // g

      if ((canonicalFlags & 1) != 0) {
        val a0 = alleles.get(0)
        val a1 = if (alleles.size() == 2)
          alleles.get(1)
        else
          a0

        assert(a0.isCalled || a0.isNoCall)
        assert(a1.isCalled || a1.isNoCall)
        assert(a0.isCalled == a1.isCalled)

        val hasGT = a0.isCalled
        if (hasGT) {
          val i = vc.getAlleleIndex(a0)
          val j = vc.getAlleleIndex(a1)
          rvb.addInt(Call2(i, j))
        } else
          rvb.setMissing()
      }

      if ((canonicalFlags & 2) != 0) {
        if (g.hasAD) {
          val ad = g.getAD
          rvb.startArray(ad.length)
          var i = 0
          while (i < ad.length) {
            rvb.addInt(ad(i))
            i += 1
          }
          rvb.endArray()
        } else
          rvb.setMissing()
      }

      if ((canonicalFlags & 4) != 0) {
        if (g.hasDP)
          rvb.addInt(g.getDP)
        else
          rvb.setMissing()
      }

      if ((canonicalFlags & 8) != 0) {
        if (g.hasGQ)
          rvb.addInt(g.getGQ)
        else
          rvb.setMissing()
      }

      if ((canonicalFlags & 16) != 0) {
        if (g.hasPL) {
          var pl = g.getPL

          // handle haploid
          if (alleles.size() == 1) {
            assert(pl.length == nAlleles)
            util.Arrays.fill(haploidPL, haploidNonsensePL)

            var i = 0
            while (i < pl.length) {
              haploidPL(triangle(i + 1) - 1) = pl(i)
              i += 1
            }

            pl = haploidPL
          }

          rvb.startArray(pl.length)
          var i = 0
          while (i < pl.length) {
            rvb.addInt(pl(i))
            i += 1
          }
          rvb.endArray()
        } else
          rvb.setMissing()
      }

      var i = nCanonicalFields
      while (i < gType.fields.length) {
        val f = gType.fields(i)
        val a = g.getAnyAttribute(f.name)
        addAttribute(rvb, a, f.typ, nAlleles)
        i += 1
      }

      rvb.endStruct() // g
    }
    rvb.endArray() // gs
  }
}

object HtsjdkRecordReader {
  val haploidNonsensePL = 1000

  private val haploidRegex = """^[0-9]+$""".r
  private val diploidRegex = """^([0-9]+)([|/])([0-9]+)$""".r

  def parseCall(gt: String, nAlleles: Int): BoxedCall = {
    gt match {
      case diploidRegex(a0, phase, a1) =>
        val c = Call2(a0.toInt, a1.toInt, phased = (phase == "|"))
        Call.check(c, nAlleles)
        c
      case VCFConstants.EMPTY_GENOTYPE => null
      case VCFConstants.EMPTY_ALLELE => null
      case haploidRegex() =>
        val i = gt.toInt
        val c = Call1(i)
        Call.check(c, nAlleles)
        c
      case _ => fatal(s"Invalid input format for Call type. Found `$gt'.")
    }
  }

  def addAttribute(rvb: RegionValueBuilder, attr: Any, t: Type, nAlleles: Int) {
    ((attr, t): @unchecked) match {
      case (null, _) =>
        rvb.setMissing()
      case (".", _) =>
        rvb.setMissing()
      case (s: String, TArray(_: TInt32, _)) =>
        val xs = s.split(",")
        rvb.startArray(xs.length)
        xs.foreach { x =>
          if (x == ".")
            rvb.setMissing()
          else
            rvb.addInt(x.toInt)
        }
        rvb.endArray()

      case (s: String, TArray(_: TFloat64, _)) =>
        val xs = s.split(",")
        rvb.startArray(xs.length)
        xs.foreach { x =>
          if (x == ".")
            rvb.setMissing()
          else
            rvb.addDouble(x.toDouble)
        }
        rvb.endArray()
      case (s: String, TArray(_: TString, _)) =>
        val xs = s.split(",")
        rvb.startArray(xs.length)
        xs.foreach { x =>
          if (x == ".")
            rvb.setMissing()
          else
            rvb.addString(x)
        }
        rvb.endArray()
      case (s: String, _: TBoolean) =>
        rvb.addBoolean(s.toBoolean)
      case (b: Boolean, _: TBoolean) =>
        rvb.addBoolean(b)
      case (s: String, _: TString) =>
        rvb.addString(s)
      case (s: String, _: TInt32) =>
        rvb.addInt(s.toInt)
      case (s: String, _: TFloat64) =>
        val d = s match {
          case "nan" => Double.NaN
          case "inf" => Double.PositiveInfinity
          case "-inf" => Double.NegativeInfinity
          case _ => s.toDouble
        }
        rvb.addDouble(d)
      case (i: Int, _: TInt32) =>
        rvb.addInt(i)
      case (d: Double, _: TInt32) =>
        rvb.addInt(d.toInt)

      case (d: Double, _: TFloat64) =>
        rvb.addDouble(d)
      case (f: Float, _: TFloat64) =>
        rvb.addDouble(f.toDouble)

      case (f: Float, _: TFloat32) =>
        rvb.addFloat(f)
      case (d: Double, _: TFloat32) =>
        rvb.addFloat(d.toFloat)

      case (l: java.util.List[_], TArray(_: TInt32, _)) =>
        rvb.startArray(l.size())
        var it = l.iterator()
        while (it.hasNext) {
          it.next() match {
            case "." => rvb.setMissing()
            case s: String => rvb.addInt(s.toInt)
            case i: Int => rvb.addInt(i)
          }
        }
        rvb.endArray()
      case (l: java.util.List[_], TArray(_: TFloat64, _)) =>
        rvb.startArray(l.size())
        var it = l.iterator()
        while (it.hasNext) {
          it.next() match {
            case "." => rvb.setMissing()
            case s: String => rvb.addDouble(s.toDouble)
            case i: Int => rvb.addDouble(i.toDouble)
            case d: Double => rvb.addDouble(d)
          }
        }
        rvb.endArray()
      case (l: java.util.List[_], TArray(_: TString, _)) =>
        rvb.startArray(l.size())
        var it = l.iterator()
        while (it.hasNext) {
          it.next() match {
            case "." => rvb.setMissing()
            case s: String => rvb.addString(s)
            case i: Int => rvb.addString(i.toString)
            case d: Double => rvb.addString(d.toString)
          }
        }
        rvb.endArray()
      case (s: String, TCall(_)) if nAlleles > 0 =>
        val call = parseCall(s, nAlleles)
        if (call == null)
          rvb.setMissing()
        else
          rvb.addInt(call)
    }
  }
}
