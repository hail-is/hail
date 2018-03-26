package is.hail.expr.ir.functions

import is.hail.expr.types._

object StringFunctions extends RegistryFunctions {

  def upper(s: String): String = s.toUpperCase

  def lower(s: String): String = s.toLowerCase

  def strip(s: String): String = s.trim()

  def contains(s: String, t: String): Boolean = s.contains(t)

  def startswith(s: String, t: String): Boolean = s.startsWith(t)

  def endswith(s: String, t: String): Boolean = s.endsWith(t)

  def firstMatchIn(s: String, regex: String): IndexedSeq[String] = regex.r.findFirstMatchIn(s).map(_.subgroups.toIndexedSeq).orNull

  def registerAll(): Unit = {
    val thisClass = getClass
    registerScalaFunction("upper", TString(), TString())(thisClass, "upper")
    registerScalaFunction("lower", TString(), TString())(thisClass, "lower")
    registerScalaFunction("strip", TString(), TString())(thisClass, "strip")
    registerScalaFunction("contains", TString(), TString(), TBoolean())(thisClass, "contains")
    registerScalaFunction("startswith", TString(), TString(), TBoolean())(thisClass, "startswith")
    registerScalaFunction("endswith", TString(), TString(), TBoolean())(thisClass, "endswith")
    registerScalaFunction("firstMatchIn", TString(), TString(), TArray(TString()))(thisClass, "firstMatchIn")
  }
}
