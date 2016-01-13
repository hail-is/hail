package org.broadinstitute.hail.annotations

import java.io.{DataInputStream, DataOutputStream}
import org.broadinstitute.hail.Utils._

abstract class AnnotationSignature {
  def emitUtilities: String

  def emitConversionIdentifier: String

  def emitType: String

}

object AnnotationSiganture {
  implicit def writableAnnotationSignature: DataWritable[AnnotationSignature] =
    new DataWritable[AnnotationSignature] {
      def write(dos: DataOutputStream, t: AnnotationSignature) {
        t match {
          case ss: SimpleSignature =>
            writeData[String](dos, "SimpleSignature")
            writeData[String](dos, ss.emitType)
            writeData[String](dos, ss.emitConversionIdentifier)

          case vcfs: VCFSignature =>
            writeData[String](dos, "VCFSignature")
            writeData[String](dos, vcfs.vcfType)
            writeData[String](dos, vcfs.emitType)
            writeData[String](dos, vcfs.number)
            writeData[String](dos, vcfs.emitConversionIdentifier)
            writeData[String](dos, vcfs.description)
        }
      }
    }

  implicit def readableAnnotationSignature: DataReadable[AnnotationSignature] =
    new DataReadable[AnnotationSignature] {
      def read(dis: DataInputStream): AnnotationSignature = {
        val t = readData[String](dis)
        t match {
          case "SimpleSignature" =>
            SimpleSignature(readData[String](dis),
              readData[String](dis))

          case "VCFSignature" =>
            VCFSignature(readData[String](dis),
              readData[String](dis),
              readData[String](dis),
              readData[String](dis),
              readData[String](dis))
        }
      }
    }
}

case class SimpleSignature(emitType: String, emitConversionIdentifier: String) extends AnnotationSignature {

  def emitUtilities = ""

}