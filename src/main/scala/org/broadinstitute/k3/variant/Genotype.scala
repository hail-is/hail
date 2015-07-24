package org.broadinstitute.k3.variant

import scala.language.implicitConversions

case class Genotype(private val GT: Int,
  val AD: (Int, Int),
  val DP: Int,
  private val PL1: Int,
  private val PL2: Int,
  formatOther: Map[String, String]) {

  def PL(): (Int, Int, Int) = {
    GT match {
      case 0 => (0, PL1, PL2)
      case 1 => (PL1, 0, PL2)
      case _ =>
        assert(GT == 2)
        (PL1, PL2, 0)
    }
  }

  def call(): Option[Call] = {
    if (GT == -1)
      None
    else
      Some(Call(GT, PL1 min PL2 min 99, PL))
  }

  override def toString(): String = {
    val b = new StringBuilder
    val c = call
    call match {
      case Some(c) =>
        c.GT match {
          case 0 => b.append("0/0")
          case 1 => b.append("0/1")
          case _ =>
            assert(c.GT == 2)
            b.append("1/1")
        }
      case None =>
        b.append("./.")
      case _ =>
        assert(false)
    }
    b += ':'
    b.append(AD._1)
    b += ','
    b.append(AD._2)
    b += ':'
    b.append(DP)
    call match {
      case Some(c) =>
        b += ':'
        b.append(c.GQ)
        b += ':'
        b.append(c.PL._1)
        b += ','
        b.append(c.PL._2)
        b += ','
        b.append(c.PL._3)
      case None =>
    }

    return b.result
  }
}
