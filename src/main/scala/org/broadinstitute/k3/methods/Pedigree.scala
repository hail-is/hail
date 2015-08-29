package org.broadinstitute.k3.methods

import java.io.{File, FileWriter}

import scala.io.Source

object TryOut {
  def main(args: Array[String]) {
    val ped = Pedigree.read("/Users/jbloom/sample.fam")
    println(ped.pedMap.get("MIGFI_1014"))
    ped.write("/Users/jbloom/sample_output.ped")
  }
}

object Pedigree {

  def read(file: String): Pedigree = {
    require(file.endsWith(".fam"))

    new Pedigree(Source.fromFile(new File(file))
      .getLines()
      .filter(line => !line.isEmpty)
      .map{ line =>
      val fields: Array[String] = line.split(" ")
      assert(fields.length == 6)
      val famID = fields(0)
      val kidID = fields(1)
      val dadID = fields(2) // '0' if father not in dataset  // FIXME: use Option
      val momID = fields(3) // '0' if mother not in dataset
      val sex = fields(4).toInt // male=1, female=2, unknown=0
      val phenotype = fields(5).toInt // FIXME: assuming binary phenotype: control=1, case=2, missing=0
      (kidID, (famID, dadID, momID, sex, phenotype))}
      .toMap)
  }
}

class Pedigree(val pedMap: Map[String, (String, String, String, Int, Int)]) {

  def write(file: String) {
    val fw = new FileWriter(new File(file))

    pedMap
      .foreach { case (kidID, (famID, dadID, momID, sex, phenotype)) =>
      fw.write(famID + " " + kidID + " " + dadID + " " + momID + " " + sex.toString + " " + phenotype.toString + "\n")
    }

    // FIXME
    fw.close()
  }
}
