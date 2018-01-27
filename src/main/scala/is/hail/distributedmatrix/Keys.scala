package is.hail.distributedmatrix

import is.hail.annotations.Annotation
import is.hail.expr.{JSONAnnotationImpex, Parser}
import is.hail.expr.types.Type
import is.hail.utils._
import org.apache.spark.SparkContext
import org.json4s.jackson.Serialization
import org.json4s.JValue
import org.json4s.jackson.JsonMethods.parse

case class KeysImpex(typeStr: String, values: Array[JValue])

object Keys {
  def unify(left: Option[Keys], right: Option[Keys], msg: String): Option[Keys] = {
    (left, right) match {
      case (Some(l), Some(r)) =>
        l.assertSame(r, msg)
        left
      case (Some(_), _) => left
      case _ => right
    }
  }
  
  def read(sc: SparkContext, uri: String): Keys = {
    val KeysImpex(typeStr, values) =  sc.hadoopConfiguration.readTextFile(uri)(in =>
      try {
        val json = parse(in)
        json.extract[KeysImpex]
      } catch {
        case e: Exception => fatal(
          s"""corrupt or outdated matrix key data.
             |  Recreate with current version of Hail.
             |  Detailed exception:
             |  ${ e.getMessage }""".stripMargin)
      })
    
    val typ = Parser.parseType(typeStr)
   
    new Keys(typ, values.map(JSONAnnotationImpex.importAnnotation(_, typ)))
  }
}

class Keys(val typ: Type, val values: Array[Annotation]) {
  def length: Int = values.length
  
  def assertSame(that: Keys, msg: String = "") {
    if (typ != that.typ)
      fatal(msg + "Keys have different types: " +
        s"${ typ.toString }, ${ that.typ.toString }")

    if (values.length != that.values.length)
      fatal(msg + s"Differing number of keys: $length, ${ that.length }")

    var i = 0
    while (i < length) {
      if (values(i) != that.values(i))
        fatal(msg + s"Key mismatch at index $i: ${typ.str(values(i))}, ${typ.str(that.values(i))}")
      i += 1
    }
  }
  
  def filter(pred: Annotation => Boolean): Keys = new Keys(typ, values.filter(pred))
  
  def filter(keep: Array[Int]): Keys = {
    require(keep.isEmpty || (keep.isIncreasing && keep.head >= 0 && keep.last < length))
    new Keys(typ, keep.map(values))
  }
  
  def filterAndIndex(pred: Annotation => Boolean): (Keys, Array[Int]) = {
    val (newValues, indices) = values.zipWithIndex.filter(vi => pred(vi._1)).unzip
    (new Keys(typ, newValues), indices)
  }
  
  def write(sc: SparkContext, uri: String) {
    val typeStr = typ.toString

    sc.hadoopConfiguration.writeTextFile(uri)(out =>
      Serialization.write(KeysImpex(typeStr, values.map(typ.toJSON)), out))
  }
}
