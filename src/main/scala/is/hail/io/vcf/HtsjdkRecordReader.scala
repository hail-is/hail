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

  def readVariantInfo(vc: VariantContext, rvb: RegionValueBuilder, infoType: TStruct, infoFlagFieldNames: Set[String]) {
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
        val fi = vc.getFilters.iterator()
        while (fi.hasNext)
          rvb.addString(fi.next())
        rvb.endArray()
      }
    }

    // info
    rvb.startStruct()
    infoType.fields.foreach { f =>
      val a = vc.getAttribute(f.name)
      addAttribute(rvb, a, f.typ, -1, isFlag = infoFlagFieldNames.contains(f.name))
    }
    rvb.endStruct() // info
  }
}

object HtsjdkRecordReader {
  def addAttribute(rvb: RegionValueBuilder, attr: Any, t: Type, nAlleles: Int, isFlag: Boolean = false) {
    ((attr, t): @unchecked) match {
      case (null, _) =>
        if (isFlag)
          rvb.addBoolean(false)
        else
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
          case "-nan" => Double.NaN
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
        val it = l.iterator()
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
        val it = l.iterator()
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
        val it = l.iterator()
        while (it.hasNext) {
          it.next() match {
            case "." => rvb.setMissing()
            case s: String => rvb.addString(s)
            case i: Int => rvb.addString(i.toString)
            case d: Double => rvb.addString(d.toString)
          }
        }
        rvb.endArray()
      case _ => fatal(s"data/type mismatch: $t / $attr (${ attr.getClass.getName }")
    }
  }
}
