package org.broadinstitute.k3.variant

import scala.language.implicitConversions
import scala.collection.mutable.ArrayBuilder
import org.broadinstitute.k3.Utils._
import org.broadinstitute.k3.utils.ByteStream

case class Genotype(private val gt: Int,
                    ad: (Int, Int),
                    dp: Int,
                    private val pl: (Int, Int, Int)) {

  require(gt >= -1 && gt <= 2)
  require(dp >= ad._1 + ad._2)
  require(gt == -1 || pl.at(gt + 1) == 0)
  require(gt != -1 || pl == null)

  private def minPl: (Int, Int) = {
    gt match {
      case 0 => (pl._2, pl._3)
      case 1 => (pl._1, pl._3)
      case 2 => (pl._1, pl._2)
    }
  }

  def write(b: ArrayBuilder[Byte]) {
    val writeDp = ad._1 + ad._2 != dp
    val writeAd2 = (gt != 0 || ad._2 != 0)
    b += ((if (writeDp) 0x08 else 0)
      | (if (writeAd2) 0x10 else 0)
      | (gt & 7)).toByte
    if (gt != -1) {
      val (pl1, pl2) = minPl
      writeULEB128(b, pl1)
      writeULEB128(b, pl2)
    }
    writeULEB128(b, ad._1)
    if (writeAd2)
      writeULEB128(b, ad._2)
    if (writeDp)
      writeULEB128(b, dp - (ad._1 + ad._2))
  }

  def isHomRef: Boolean = gt == 0

  def isHet: Boolean = gt == 1

  def isHomVar: Boolean = gt == 2

  def notCalled: Boolean = gt == -1

  def called: Boolean = gt != -1

  def call: Option[Call] = {
    if (gt == -1)
      None
    else {
      val (pl1, pl2) = minPl
      Some(Call(gt, pl1 min pl2 min 99, pl))
    }
  }

  override def toString: String = {
    val b = new StringBuilder
    call match {
      case Some(c) =>
        c.gt match {
          case 0 => b.append("0/0")
          case 1 => b.append("0/1")
          case 2 => b.append("1/1")
        }
      case None => b.append("./.")
      case _ => fail()
    }
    b += ':'
    b.append(ad._1)
    b += ','
    b.append(ad._2)
    b += ':'
    b.append(dp)
    call match {
      case Some(c) =>
        b += ':'
        b.append(c.gq)
        b += ':'
        b.append(c.pl._1)
        b += ','
        b.append(c.pl._2)
        b += ','
        b.append(c.pl._3)
      case None =>
    }

    b.result()
  }
}

object Genotype {
  def read(a: ByteStream): Genotype = {
    val b = a.readByte()

    val gt = (b << 29) >> 29
    val writeDp = (b & 0x08) != 0
    val writeAd2 = (b & 0x10) != 0

    val pl =
      if (gt != -1) {
        val pl1 = a.readULEB128()
        val pl2 = a.readULEB128()

        gt match {
          case 0 => (0, pl1, pl2)
          case 1 => (pl1, 0, pl2)
          case 2 => (pl1, pl2, 0)
        }
      } else
        null

    val ad1: Int = a.readULEB128()
    val ad2: Int =
      if (writeAd2)
        a.readULEB128()
      else
        0

    val dpDelta =
      if (writeDp)
        a.readULEB128()
      else
        0

    Genotype(gt, (ad1, ad2), ad1 + ad2 + dpDelta, pl)
  }
}
